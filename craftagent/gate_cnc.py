# -*- coding: utf-8 -*-
"""gate.py -- 安全门：一条程序能不能放行，由一组独立的规则回答，不由计划器回答。

这是 robot-spine 的 spine/gate.py 换了一个对象：那边管一条关节指令，
这边管一行 G 代码。结构照抄，因为它已经被证明是对的形状：

    Verdict(ok, reasons, detail)
    detail 里每条规则给出自己的理由字符串，所以一次拒绝永远能说是**哪一条**、
    **哪一行**。理由字符串里带行号，这是这个仓对"可归因"的最低要求。

六条规则，每条都能被单独关掉：

    gcode_syntax   解析器读不懂（未知代码、几何自相矛盾的圆弧）
    axis_limit     走到机床行程之外
    feed_limit     进给超过这台机床的额定
    accel_limit    规划之后仍然存在超过额定加速度的段
    radius_limit   圆弧半径小于刀具能走的最小值
    spindle        切削段上没有主轴转速，或者转速超额定

**门的数字必须成对报**：合法程序放行率必须 100%，植入故障拦截率也必须 100%。
只报第二个数，一个"全部拒绝"的门能拿满分，而它什么也没做。
"""
import math

from . import gcode as gcode_mod
from . import plan as plan_mod

RULES = ('gcode_syntax', 'axis_limit', 'feed_limit', 'accel_limit',
         'radius_limit', 'spindle', 'quality_predict')

# 撞刀那四条与前七条正交：前七条问"这条指令有没有越界"，这四条问
# "刀会不会撞到不是要切的东西"。分开列，是因为它们需要的是几何
# （毛坯/压板/刀柄），前七条只需要指令和规划表。混在一起以后，
# 想只跑前七条就得假装没有毛坯——那正是最容易骗自己的一步。
COLLIDE_RULES = ('table_crash', 'clamp_crash', 'holder_crash', 'rapid_into_stock')
ALL_RULES = RULES + COLLIDE_RULES


class Limits(object):
    """一台机床的硬边界。默认值写得偏保守，是给一台中档三轴立加用的。"""

    def __init__(self, envelope=None, feed_max_mm_min=6000.0,
                 a_max_mm_s2=1500.0, r_min_mm=0.5,
                 spindle_max_rpm=12000.0, require_spindle=True,
                 spec_um=8.5, quality_margin=1.0):
        # envelope: ((xmin,xmax),(ymin,ymax),(zmin,zmax))
        #
        # Z 上限原来是 10.0，而默认夹具的压板顶面在 12.0、安全 Z 在 14.0。
        # 也就是说这台机床按默认参数**退不出夹具**：任何一次正确的抬刀都会被
        # axis_limit 拒绝。这不是测试的问题，是这两个默认值互相矛盾，
        # 而且矛盾的方向正好是"看起来门很严"。
        # 一台能装下这块料的三轴立式加工中心，Z 上限至少要在夹具之上留出行程：
        # 16 是"刚好够退刀"，这里取 100 是给真实主轴鼻子留位置。
        self.envelope = envelope or ((-10.0, 800.0), (-10.0, 450.0), (-400.0, 100.0))
        self.feed_max_mm_min = feed_max_mm_min
        self.a_max_mm_s2 = a_max_mm_s2
        self.r_min_mm = r_min_mm
        self.spindle_max_rpm = spindle_max_rpm
        self.require_spindle = require_spindle
        # 质量预测门：放行之前先让虚拟机床走一遍，预测的最差圆度超过
        # spec_um * quality_margin 就拒绝。
        #
        # 这一条与前五条不同：前五条检查的是**指令是否越界**，
        # 这一条检查的是**越界之前就已经做不到**。进给在额定内、
        # 行程也在内，但拐角处的跟随误差让圆度超差——这在真实车间里
        # 是最常见的报废原因，也是"事后抽检"唯一能发现的那一类。
        # 把它放到门里，就是把"事后抽检"变成"事前拦截"。
        self.spec_um = spec_um
        self.quality_margin = quality_margin


class Verdict(object):
    __slots__ = ('ok', 'reasons', 'detail', 'lines')

    def __init__(self, ok, reasons, detail, lines=None):
        self.ok, self.reasons, self.detail = ok, reasons, detail
        self.lines = lines or {}

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return '<Verdict %s %s>' % ('OK' if self.ok else 'REFUSED',
                                    ','.join(self.reasons))


class Gate(object):
    def __init__(self, limits=None, rules=None, fixture=None):
        self.lim = limits or Limits()
        # 规则可以被有意识地关掉，好让 bench 说明每条规则到底买到了什么，
        # 而不是假定它买到了什么
        self.rules = set(rules or ALL_RULES)
        # fixture 为 None 时用 collide.Fixture() 的默认工件。撞刀规则的
        # 正确性依赖这块几何，所以它可以被换掉，而不是写死在规则里。
        self.fixture = fixture

    def _fixture(self):
        if self.fixture is not None:
            return self.fixture
        from . import collide as collide_mod
        return collide_mod.Fixture()

    # -- 各条规则 ------------------------------------------------------
    def _axis_limit(self, segs):
        out = {}
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = self.lim.envelope
        for s in segs:
            for i, (lo, hi) in enumerate(((xlo, xhi), (ylo, yhi), (zlo, zhi))):
                for pt in (s.start, s.end):
                    if pt[i] < lo - 1e-9 or pt[i] > hi + 1e-9:
                        axis = 'XYZ'[i]
                        out.setdefault('axis_limit',
                                       'line %d: %s=%.3f outside [%.1f, %.1f]'
                                       % (s.line, axis, pt[i], lo, hi))
                        return out
        return out

    def _feed_limit(self, segs):
        for s in segs:
            if s.feed and s.feed > self.lim.feed_max_mm_min + 1e-9:
                return {'feed_limit': 'line %d: F%.1f above %.1f mm/min'
                        % (s.line, s.feed, self.lim.feed_max_mm_min)}
        return {}

    @staticmethod
    def _seg_accel(r):
        """这一段内部真正用到的加速度。

        必须用规划器算出来的 d_acc（加速距离），不能用 L/2 近似：
        三角形速度曲线里 d_acc 可以远小于 L/2，拿 L/2 去反推加速度
        会把一个完全合规的段算成恰好等于上限、再因为浮点误差判成超限。
        原来就是这么误报的，而且是**只在上限处**误报——最难看见的那一种。
        """
        d = r.get('d_acc')
        if d is None or d <= 1e-12:
            return 0.0
        return (r['v_peak'] ** 2 - r['v_entry'] ** 2) / (2.0 * d)

    def _accel_limit(self, pl):
        """不存在"这一段用到的加速度超过上限"这回事——规划器本身就是按
        a_max 规划的，它不可能产出超过自己上限的曲线。

        所以这条规则检查的不是"规划器算错了"，而是**规划器的上限和门认定的
        上限不是同一个数**：规划用的 a_max 比门的额定大，程序就可能在门的
        眼里超载。这才是这条规则真正能抓到的东西，也是它唯一该抓的东西。
        写成"重算加速度再比一次"会退化成比对浮点误差，只会误报。
        """
        cap = self.lim.a_max_mm_s2
        used = float((pl.get('params') or {}).get('a_max_mm_s2', cap))
        if used > cap * (1.0 + 1e-9):
            return {'accel_limit': 'planned at %.1f mm/s^2, above the machine rating %.1f'
                    % (used, cap)}
        return {}

    def _radius_limit(self, segs):
        for s in segs:
            if s.kind == 'arc' and s.radius is not None:
                if s.radius < self.lim.r_min_mm - 1e-12:
                    return {'radius_limit': 'line %d: R=%.4f below %.4f mm'
                            % (s.line, s.radius, self.lim.r_min_mm)}
        return {}

    def _spindle(self, segs, meta):
        if not self.lim.require_spindle:
            return {}
        rpm = meta.get('spindle_rpm')
        if rpm is not None and rpm > self.lim.spindle_max_rpm + 1e-9:
            return {'spindle': 'S%.0f above rated %.0f rpm'
                    % (rpm, self.lim.spindle_max_rpm)}
        cutting = [s for s in segs if (s.s or 1) == 1 and s.feed and s.feed > 0]
        if cutting and rpm is None:
            return {'spindle': 'line %d: cutting move with no S word anywhere in the program'
                    % cutting[0].line}
        return {}

    def _quality_predict(self, pl):
        from . import plant as plant_mod
        passes = plant_mod.machine_pass(pl['rows'], pl['params'])
        limit = self.lim.spec_um * self.lim.quality_margin
        worst = None
        for m in passes:
            if m.get('roundness_um') is None:      # 空行程不参与质量评判
                continue
            if worst is None or m['roundness_um'] > worst['roundness_um']:
                worst = m
        if worst is not None and worst['roundness_um'] > limit:
            return {'quality_predict':
                    'line %d: predicted roundness %.2f um above %.2f um (feed %.0f mm/min)'
                    % (worst['line'], worst['roundness_um'], limit, worst['feed_mm_min'])}, worst['line']
        return {}, 0

    # -- 唯一的入口 ----------------------------------------------------
    def check_text(self, text, name='<program>', plan_params=None):
        """检查一段程序文本。返回 Verdict。"""
        try:
            segs, meta = gcode_mod.parse(text, name=name)
        except gcode_mod.GcodeError as e:
            return Verdict(False, ['gcode_syntax'],
                           {'gcode_syntax': str(e)},
                           {'gcode_syntax': e.line or 0})
        return self.check_segments(segs, meta, plan_params)

    def check_segments(self, segs, meta, plan_params=None):
        reasons, detail, lines = {}, {}, {}
        # 顺序是刻意的：先报能拦住这条程序的最"硬"的那一条。
        # 一条会撞行程的程序同时也缺主轴转速时，报告应当先说撞行程——
        # 那是能动的手，缺转速不是。
        if 'axis_limit' in self.rules:
            got = self._axis_limit(segs)
            if got:
                reasons.update(got); detail.update(got)
                lines['axis_limit'] = segs[0].line
        if 'feed_limit' in self.rules:
            got = self._feed_limit(segs)
            if got:
                reasons.update(got); detail.update(got)
                right = [s.line for s in segs if s.feed and s.feed > self.lim.feed_max_mm_min]
                lines['feed_limit'] = right[0] if right else 0
        if 'radius_limit' in self.rules:
            got = self._radius_limit(segs)
            if got:
                reasons.update(got); detail.update(got)
                right = [s.line for s in segs
                         if s.kind == 'arc' and s.radius is not None
                         and s.radius < self.lim.r_min_mm - 1e-12]
                lines['radius_limit'] = right[0] if right else 0
        # 主轴规则排最后：前面任何一条已经拒绝时，它不再追加上去，
        # 否则一条超出行程的程序会被报成"缺主轴"，真正该修的那行反而被埋掉。
        if 'spindle' in self.rules and not reasons:
            got = self._spindle(segs, meta)
            if got:
                reasons.update(got); detail.update(got)
                lines['spindle'] = 0
        pl = None
        if 'accel_limit' in self.rules:
            p = dict(plan_mod.DEFAULT)
            p['a_max_mm_s2'] = self.lim.a_max_mm_s2
            p.update(plan_params or {})
            pl = plan_mod.plan(segs, p)
            got = self._accel_limit(pl)
            if got:
                reasons.update(got); detail.update(got)
                lines['accel_limit'] = 0
        if 'quality_predict' in self.rules and not reasons:
            # 前五条已经拒绝时不再追加上去：一条会撞行程的程序，
            # 报告应当先说撞行程。质量预测只在"指令本身都合规、
            # 但结果仍不合格"时才有话可说——那正是它的存在理由。
            if pl is None:
                pp = dict(plan_mod.DEFAULT)
                pp.update(plan_params or {})
                pl = plan_mod.plan(segs, pp)
            got, ln = self._quality_predict(pl)
            if got:
                reasons.update(got); detail.update(got)
                lines['quality_predict'] = ln
        # 撞刀四条。排在圆度之后、但优先级上更硬：圆度不好是废品，
        # 撞刀是废机床。所以它即使前面已经有理由也要追加，并且
        # reason 排序时 collision 这类会自然排到前面（'c' < 'q'）。
        # 唯一被跳过的情况是前七条已经拒绝——那时报告该先说那个，
        # 否则一条超行程的程序会被报成"撞压板"，真正该修的行被埋掉。
        collide_rules = [r for r in COLLIDE_RULES if r in self.rules]
        if collide_rules and not reasons:
            from . import collide as collide_mod
            got, glines = collide_mod.check(segs, self._fixture())
            for rule in collide_rules:
                if rule in got:
                    reasons[rule] = got[rule]
                    detail[rule] = got[rule]
                    lines[rule] = glines.get(rule, 0)
        ok = not reasons
        return Verdict(ok, sorted(reasons), detail, lines)


def check_text(text, limits=None, rules=None, name='<program>', plan_params=None):
    return Gate(limits, rules).check_text(text, name=name, plan_params=plan_params)
