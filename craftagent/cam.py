# -*- coding: utf-8 -*-
"""cam.py -- 轮廓 + 层深 + 材料 -> 一条能被门检查的 G 代码程序。

这是整条链里唯一"生成"东西的地方，所以它要守住两件事：

  一、**生成的每一行，参数都要说得出从哪来。**
      S 与 F 不是写死的，也不是模型随口给的，是 feeds.choose() 查表查出来的，
      并把材料、工序、刀具一起写进程序的注释里。一条没有出处的 S3000 与
      一条查表来的 S3000 在语法上一模一样，而它们的可信度不是一回事。

  二、**生成器自己不能先犯规。**
      bench 里的 legal_programs 曾经造出"比毛坯还宽"的零件，
      于是那些"合法程序"实际上会撞压板。同一个错误在这里的代价更大：
      生成器造出一条会撞的程序，门拦住了，看起来是门的功劳，
      实际是生成器坏了。所以 build() 的最后一步总是**自检再返回**。

层深写在图层名里（见 drawing.py）。这里把每个轮廓变成：
    下刀 -> 沿轮廓走一圈 -> 抬刀
多层（深度大于一次切深）时按 ap 分多次走，这才是车间里的做法，
一次扎到 -6 mm 的孔和三刀下到 -6 mm 是两件事，而门只看后者合规。
"""
import math

from . import drawing as drawing_mod
from . import feeds as feeds_mod


class CamError(Exception):
    pass


class Plan(object):
    """一次生成的结果：程序文本、用到的参数、以及它的自检结论。"""

    def __init__(self, text, choices, meta):
        self.text = text
        self.choices = choices
        self.meta = meta

    def __repr__(self):
        return '<Plan %d lines, %d contours, %d choices>' % (
            self.text.count('\n'), self.meta.get('contours', 0), len(self.choices))


def _safe_z(bounds, margin=5.0):
    return margin


def _fmt(v):
    return ('%.3f' % v).rstrip('0').rstrip('.') or '0'


def build(dr, material='P', tool=None, machine=None, stock=None,
          op='rough', ap_per_pass=1.0, safe_z=None, conservative=1.0,
          plan_ok=True):
    """把一张读出来的图纸变成一条程序。

    stock: (x0, y0, x1, y1, top_z, thickness)。给 None 时按轮廓包围盒 + 边距推。
    ap_per_pass: 每一刀的轴向切深。深度大于它就要分多次走。
    """
    if not dr.contours:
        raise CamError('图纸里没有可切的闭合轮廓，没有东西可生成')
    tool = tool or feeds_mod.Tool(6.0, 3, 'carbide')
    machine = machine or feeds_mod.Machine()
    bx0, by0, bx1, by1 = dr.bounds()

    if stock is None:
        pad = tool.diameter_mm + 10.0
        stock = (bx0 - pad, by0 - pad, bx1 + pad, by1 + pad, 0.0,
                 max(10.0, ap_per_pass * 3 + 2.0))
    sx0, sy0, sx1, sy1, top_z, thick = stock
    if safe_z is None:
        safe_z = top_z + 5.0

    # 排序：外层先、型腔后，同深度按面积从大到小。
    # 顺序写错的具体后果是：先切了小腔，再切外轮廓时件已经松了，
    # 最后一道把件弹起来——那是废件，而每条指令单独看都合规。
    def key(c):
        return (c.depth, -abs(c.area()))
    contours = sorted(dr.contours, key=key)

    L = ['G21 G90', 'G0 Z%s' % _fmt(safe_z)]
    choices = []
    head = []

    for c in contours:
        target = c.depth
        if abs(target) <= 1e-9:
            raise CamError('轮廓 %s 的层深是 0：没有材料要去除' % c.layer)
        passes = [min(ap_per_pass, abs(target))]
        # 分刀：每刀不超过 ap_per_pass，最后一刀切剩下的
        dep = passes[0]
        while dep < abs(target) - 1e-9:
            nxt = min(dep + ap_per_pass, abs(target))
            passes.append(nxt - dep)
            dep = nxt
        z_here = 0.0
        first = True
        for dp in passes:
            z_here -= dp
            ch = feeds_mod.choose(material, op, tool, machine,
                                  ap_mm=dp, conservative=conservative)
            if first:
                choices.append(ch)
                # 注释用 ';' 不用 '#'：G 代码里 '#' 是参数变量前缀，
                # 这个仓的解析器不认识它，会带行号拒绝。自检第一次就抓到了这个。
                head.append('; %s  %s  ap=%.2f  n=%.0f rpm  vf=%.0f mm/min  '
                            'Pc=%.2f/%.2f kW  %s'
                            % (c.layer, ch.op, dp, ch.rpm, ch.feed_mm_min,
                               ch.power_kw, ch.power_available_kw,
                               ('夹: ' + ','.join(ch.clamped)) if ch.clamped else '未夹'))
                L.append('M3 S%.0f' % round(ch.rpm))
                L.append('G0 Z%s' % _fmt(safe_z))
                if c.pts:
                    L.append('G0 X%s Y%s' % (_fmt(c.pts[0][0]), _fmt(c.pts[0][1])))
                L.append('G1 Z%s F%.0f' % (_fmt(z_here), round(ch.feed_mm_min * 0.3)))
                first = False
            else:
                choices.append(ch)
                L.append('G1 Z%s F%.0f' % (_fmt(z_here), round(ch.feed_mm_min * 0.3)))
            # 沿轮廓走一圈。进给用查表值；首点是 G0 定位，所以从第二点开始走。
            n = len(c.pts)
            for i in range(1, n + 1):
                x, y = c.pts[i % n]
                L.append('G1 X%s Y%s F%.0f'
                         % (_fmt(x), _fmt(y), round(ch.feed_mm_min)))
            L.append('G0 Z%s' % _fmt(safe_z))
    L.append('M5')
    text = '\n'.join(head + L) + '\n'
    plan = Plan(text, choices, {'contours': len(contours), 'stock': stock,
                                'material': material, 'op': op,
                                'tool': tool.name, 'safe_z': safe_z,
                                'ap_per_pass': ap_per_pass})
    if plan_ok:
        self_check(plan, dr, stock)
    return plan


def self_check(plan, dr, stock):
    """生成器自己的自检：用**同一个解析器**把文本读回来。

    只检查三件生成器最常犯错的事：
      1. 程序里的坐标是否都在毛坯之上（不是机床行程——那是门的事；
         这里是"这个件本身"的事）。
      2. 层深是否真的达到了图要求的最深那一层。
      3. 是否有任何一刀超过了毛坯厚度（会直接切到台面）。
    三条都是生成器能自己发现的，而不是等门替它发现。
    """
    from . import gcode as gcode_mod
    segs, meta = gcode_mod.parse(plan.text)
    sx0, sy0, sx1, sy1, top_z, thick = stock
    problems = []
    zmin = None
    for s in segs:
        for pt in (s.start, s.end):
            x, y, z = pt
            if not (sx0 - 1e-6 <= x <= sx1 + 1e-6 and sy0 - 1e-6 <= y <= sy1 + 1e-6):
                problems.append('line %d: (%.2f,%.2f) 在毛坯之外' % (s.line, x, y))
            if zmin is None or z < zmin:
                zmin = z
    want = min(c.depth for c in dr.contours)
    if zmin is not None and zmin > want + 1e-6:
        problems.append('最深只到 Z=%.3f，图纸要求 %.3f' % (zmin, want))
    if zmin is not None and zmin < top_z - thick - 1e-6:
        problems.append('Z=%.3f 已超过毛坯厚度 %.1f' % (zmin, thick))
    if problems:
        raise CamError('生成器自检失败：' + '; '.join(problems[:4]))
    return True


# ---------------------------------------------------------------- 闭环回退
class Attempt(object):
    """一次尝试：用哪个保守系数、生成了什么、门怎么说。"""

    def __init__(self, conservative, plan, verdict, reason):
        self.conservative = conservative
        self.plan = plan
        self.verdict = verdict
        self.reason = reason

    def __repr__(self):
        return '<Attempt k=%.2f %s%s>' % (
            self.conservative, 'OK' if self.verdict.ok else 'REFUSED',
            '' if self.verdict.ok else ' ' + ','.join(self.verdict.reasons))


def solve(dr, material='P', tool=None, machine=None, stock=None, op='rough',
          ap_per_pass=1.0, conservative=1.0, limits=None, max_tries=8,
          floor=0.15, steps=1.35):
    """查到被门拒绝为止，然后降参数再来一遍。

    这是这条链里唯一"闭环"的地方，也是它值得存在的理由：

        查表定参数 -> 生成 -> 门 -> 若拒绝, 把保守系数乘小 -> 再来

    为什么是降**进给**而不是降转速：门拒绝的理由是圆度（quality_predict），
    而圆度由跟随误差主导，跟随误差随进给线性涨（见 plant.py 的闭式解）。
    降转速不改变进给，它只改变切口速度和每齿厚度——那是另一件事。
    所以回退的方向由**拒绝理由**决定，不是由"哪个参数看起来更安全"决定。

    迭代次数与每一次的拒绝理由都会被留下来。一个"通过了"的结论若说不出
    它试过什么、为什么放弃那些参数，那它与随机搜索没有区别。
    """
    from . import gate_cnc as gate_mod
    from . import collide as collide_mod
    tool = tool or feeds_mod.Tool(6.0, 3, 'carbide')
    machine = machine or feeds_mod.Machine()
    limits = limits or gate_mod.Limits()
    fx = collide_mod.Fixture()
    gate = gate_mod.Gate(limits, fixture=fx)

    k = conservative
    attempts = []
    for i in range(max_tries):
        plan = build(dr, material, tool, machine, stock=stock, op=op,
                     ap_per_pass=ap_per_pass, conservative=k)
        v = gate.check_text(plan.text)
        why = ','.join(v.reasons)
        attempts.append(Attempt(k, plan, v, why))
        if v.ok:
            return plan, attempts
        # 只对**可以由降进给解决**的理由重试。
        # 撞刀、超行程、缺主轴这几类是几何/程序结构问题，
        # 降进给不会让刀绕开压板——把它们也拿去重试，只会
        # 白跑八轮之后报"试过了都不行"，把真正的原因埋掉。
        retryable = {'quality_predict', 'accel_limit'}
        if not set(v.reasons) <= retryable:
            return plan, attempts
        k /= steps
        if k < floor:
            break
    return (attempts[-1].plan if attempts else None), attempts
