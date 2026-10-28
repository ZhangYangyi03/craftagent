# -*- coding: utf-8 -*-
"""collide.py -- 工件周围那些**不是要切的东西**，以及撞上去的四条规则。

gate_cnc.py 原来那六条规则回答的是"这条指令本身越界没有"：行程、进给、
加速度、圆弧半径、主轴、圆度。它们全都是关于**被加工的那块料**的。

可是车间里废掉机床的，往往不是料。是压板、是台面、是刀柄。
这是四条与那六条正交的规则，也是这个文件存在的理由：

    table_crash        刀尖落到台面以下
    clamp_crash        刀尖或刀柄进了压板
    holder_crash       刀柄进了料（伸出量不够，也就是"刀具长度这一栏填的是假的"）
    rapid_into_stock   快速移动从料里穿过去

为什么单独立一个文件而不是并进 gate_cnc：这四条需要的是几何（毛坯、压板、
刀柄），前六条需要的是指令和规划表。混在一起以后，想只跑前六条就得假装
没有毛坯，而那正是最容易骗自己的一步。

三条写在这里的取舍，都是刻意的：

  * 高度场而不是体素。三轴机的刀轴是竖直的，材料只能从上面被削短，
    所以"每个 (x,y) 一个 z"是**精确**的，不是近似。五轴就不是了，
    那需要体素场，而这个仓没有，所以它不假装支持五轴。

  * 快速移动只在**横移**时判。每个程序的结尾都有 G0 Z5 直着抬出工件，
    把那条判成"穿过材料"会让门拒绝世界上每一份程序。抬刀不切东西，
    因为刀已经在那个柱子里了。这一条是实测出来的，不是想出来的。

  * 圆弧真的要沿弧采样。只取两端点（或者取弦）会漏掉鼓出来的那一段：
    弦在弧的内侧，撞到压板的那一点恰好不在弦上。这个文件按圆心/半径/
    方向把弧采样出来，宁可贵一点。
"""
import math


class Stock(object):
    """一块料。默认值就是 craftagent 的 demo 程序切的那一块。"""

    def __init__(self, x0=0.0, y0=0.0, x1=60.0, y1=84.0, top_z=0.0,
                 stock_z=28.0):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.top_z = top_z
        self.stock_z = stock_z
        self.table_z = top_z - stock_z

    def footprint(self, x, y, margin=0.0):
        return (self.x0 - margin <= x <= self.x1 + margin and
                self.y0 - margin <= y <= self.y1 + margin)


class Clamp(object):
    """一块压板：一个盒子，顶面用 Z 表示。"""

    def __init__(self, x0, y0, x1, y1, top_z, name=None):
        self.x0, self.y0, self.x1, self.y1, self.top_z = x0, y0, x1, y1, top_z
        self.name = name

    def holds(self, x, y, z):
        return (self.x0 - 1e-9 <= x <= self.x1 + 1e-9 and
                self.y0 - 1e-9 <= y <= self.y1 + 1e-9 and
                z <= self.top_z + 1e-9)

    def top(self):
        return self.top_z

    def __repr__(self):
        return '<Clamp %s x[%.0f,%.0f] y[%.0f,%.0f] top %.0f>' % (
            self.name or '?', self.x0, self.x1, self.y0, self.y1, self.top_z)


class Tool(object):
    """刀尖半径、伸出量、刀柄。撞人的从来是刀柄那一截。"""

    def __init__(self, diameter=6.0, stickout=20.0, holder_diameter=30.0):
        self.radius = diameter / 2.0
        self.stickout = stickout
        self.holder_radius = holder_diameter / 2.0

    def holder_bottom(self, tip_z):
        """刀尖在 tip_z 时，刀柄最低点的 Z。

        这一个数就是真车间里最常见的一次撞刀：刀装得太短，刃还没到底
        刀柄已经坐在工件上了。CAM 把它叫"伸出量"，而向大模型要 G 代码时，
        这个数从来不在提示词里。
        """
        return tip_z + self.stickout

    def __repr__(self):
        return '<Tool D%.1f stickout %.1f>' % (2 * self.radius, self.stickout)


class Fixture(object):
    """料 + 四块压板 + 刀。默认压板在料四角外，顶面比料高 12 mm。

    压板比料高这件事不是装饰：正因为高，一条"在 Z=0 横着飞过去"的快速
    移动才是撞刀。如果压板与料齐平，这条规则就永远抓不到东西，
    而它看起来还是通过了测试。
    """

    def __init__(self, stock=None, clamps=None, tool=None, clamp_rise=12.0,
                 clamp_width=10.0):
        self.stock = stock or Stock()
        self.tool = tool or Tool()
        if clamps is not None:
            self.clamps = list(clamps)
        else:
            s = self.stock
            w = clamp_width
            z = s.top_z + clamp_rise
            # 压板必须整体落在机床行程之内。原来 clamp0 占到 x=-10，
            # 而默认行程下界也是 -10 —— 于是所有撞压板的故障都被
            # axis_limit 先拦下，"由指定的那条规则拦住"就成了假的：
            # 那四条撞刀规则一次也没被真正考过。压板往里收 4 mm。
            self.clamps = [
                Clamp(s.x0 - w, s.y0 + 2.0, s.x0 - 1.0, s.y0 + 10.0, z, 'clamp0'),
                Clamp(s.x0 - w, s.y1 - 10.0, s.x0 - 1.0, s.y1 - 2.0, z, 'clamp1'),
                Clamp(s.x1 + 1.0, s.y0 + 2.0, s.x1 + w, s.y0 + 10.0, z, 'clamp2'),
                Clamp(s.x1 + 1.0, s.y1 - 10.0, s.x1 + w, s.y1 - 2.0, z, 'clamp3'),
            ]

    def safe_z(self):
        """横着飞之前必须在的 Z：料顶与所有压板顶里更高的那个，再加一点。"""
        tops = [self.stock.top_z] + [c.top() for c in self.clamps]
        return max(tops) + 2.0

    def __repr__(self):
        return '<Fixture %r %r %d clamps safe_z %.1f>' % (
            self.stock, self.tool, len(self.clamps), self.safe_z())


# --------------------------------------------------------------- 采样
def sample_seg(seg, step=0.5, max_pts=4000):
    """把一段（直线或圆弧）采样成点列，端点一定在里面。

    圆弧按圆心/半径/方向真的走一遍，不是取弦：弦在弧内侧，撞到压板的
    那一点恰好可能不在弦上——那样门就会放过一次真撞。
    """
    a, b = seg.start, seg.end
    if seg.kind == 'arc' and seg.centre is not None and seg.radius:
        cx, cy = seg.centre[0], seg.centre[1]   # 圆心是二维的, Z 由端点线性插值
        r = seg.radius
        a0 = math.atan2(a[1] - cy, a[0] - cx)
        a1 = math.atan2(b[1] - cy, b[0] - cx)
        if seg.ccw:
            while a1 <= a0:
                a1 += 2 * math.pi
            sweep = a1 - a0
        else:
            while a1 >= a0:
                a1 -= 2 * math.pi
            sweep = a1 - a0
        n = max(2, min(max_pts, int(abs(sweep) * r / max(step, 1e-6)) + 1))
        out = []
        for k in range(n + 1):
            t = k / n
            ang = a0 + sweep * t
            out.append((cx + r * math.cos(ang), cy + r * math.sin(ang),
                        a[2] + (b[2] - a[2]) * t))
        out[0] = (a[0], a[1], a[2])
        out[-1] = (b[0], b[1], b[2])
        return out
    L = math.dist(a, b)
    n = max(1, min(max_pts, int(L / max(step, 1e-6)) + 1))
    return [(a[0] + (b[0] - a[0]) * k / n,
             a[1] + (b[1] - a[1]) * k / n,
             a[2] + (b[2] - a[2]) * k / n) for k in range(n + 1)]



# --------------------------------------------------------------- 四条规则
def check(segs, fixture, eps=1e-9):
    """四条撞刀规则。返回 ({rule: 理由}, {rule: 行号})。

    理由字符串一律带行号，理由见 gate_cnc.py 顶部：一次拒绝永远要说得出
    是哪一条、哪一行。
    """
    fx = fixture
    st = fx.stock
    tool = fx.tool
    reasons, lines = {}, {}

    def note(rule, line, msg):
        if rule not in reasons:
            reasons[rule] = msg
            lines[rule] = line

    for seg in segs:
        pts = sample_seg(seg)

        # 1. 台面 —— 刀尖落到台面以下。任何一段都判，不等它是不是切削。
        for (x, y, z) in (pts if len(pts) <= 3 else (pts[0], pts[-1], pts[len(pts) // 2])):
            if z < st.table_z - eps:
                note('table_crash', seg.line,
                     'line %d: 刀尖 Z=%.3f 低于台面 %.3f' % (seg.line, z, st.table_z))
                break
        if 'table_crash' in reasons:
            continue

        # 2. 压板 —— 只查刀尖。
        #
        # 我一开始还写了一条"刀柄撞压板"，加完才发现它是死代码：
        # 刀柄与刀尖同轴，且 holder_bottom = 刀尖 + 伸出量 恒在刀尖之上。
        # 所以只要刀柄底面进了压板，刀尖必然**更**在压板里 —— 刀尖检查
        # 先就报了，那条分支永远轮不到。留着它只会让人以为门管了四件事
        # 而实际只管了三件，所以删掉。
        #
        # 真要区分"刀柄蹭到压板侧面而刀尖没进"，得把刀柄半径算进 x-y：
        # 刀尖在压板之外两个身位、刀柄的粗脖子却扫到了压板边缘。
        # 那是真几何，这个文件没实现，所以它不声称有这条规则。
        #
        # 刀柄真正独立的一条是"刀柄进料"（下面第 3 条）：料是有限厚的块，
        # 刀尖切在里面、刀柄也扎进来，那是刀装得太短——与压板无关。
        for c in fx.clamps:
            hit_tip = None
            for (x, y, z) in pts:
                if c.holds(x, y, z):
                    hit_tip = (x, y, z)
                    break
            if hit_tip:
                note('clamp_crash', seg.line,
                     'line %d: 刀尖 (%.1f,%.1f,%.1f) 落在压板 %s 里'
                     % (seg.line, hit_tip[0], hit_tip[1], hit_tip[2], c.name))
                break

        # 3. 刀柄进料 —— 刃用完了
        if 'holder_crash' not in reasons:
            for (x, y, z) in pts:
                if st.footprint(x, y) and z < st.top_z - eps:
                    if tool.holder_bottom(z) < st.top_z - eps:
                        note('holder_crash', seg.line,
                             'line %d: 刀柄底面 Z=%.2f 已在料里（料顶 %.2f，'
                             '刀尖 Z=%.2f，伸出量 %.1f mm 不够）'
                             % (seg.line, tool.holder_bottom(z), st.top_z, z,
                                tool.stickout))
                        break

        # 4. 快速穿过原料 —— 只判横移
        if 'rapid_into_stock' not in reasons and (seg.s or 0) == 0:
            a, b = seg.start, seg.end
            lateral = math.hypot(b[0] - a[0], b[1] - a[1])
            if lateral > 1e-9:
                for (x, y, z) in pts:
                    if st.footprint(x, y, margin=tool.radius) and z < st.top_z - eps:
                        note('rapid_into_stock', seg.line,
                             'line %d: 快速移动在 Z=%.2f 横向穿过料（料顶 %.2f）'
                             '，从 (%.1f,%.1f) 到 (%.1f,%.1f)'
                             % (seg.line, z, st.top_z, a[0], a[1], b[0], b[1]))
                        break
    return reasons, lines


# --------------------------------------------------------------- 植入故障
def planted_faults(fixture=None):
    """会撞的四条，各自必须由**指定的那一条规则**拦下来。

    与 gate_cnc.py 里那六条同一个理由：说"门拒绝了"没有意义，
    要说的是"由哪一条拒绝的"。别的规则顺手拦下不算。

    所以每条故障都带 M3 S8000：不带的话它们会先被 spindle 规则
    （"切削段上没有主轴转速"）拦下，撞刀那四条就一次也考不到。
    这条纪律和 bench 里"每条故障只碰一个字段"是同一条，只是字段在两条链上。
    """
    fx = fixture or Fixture()
    st, tool = fx.stock, fx.tool
    sz = fx.safe_z()
    c0 = fx.clamps[0]
    cx, cy = (c0.x0 + c0.x1) / 2.0, (c0.y0 + c0.y1) / 2.0
    ctop = fx.clamps[0].top()
    mid_x = (st.x0 + st.x1) / 2.0
    mid_y = (st.y0 + st.y1) / 2.0

    # 料厚这一条要说清楚：默认 28 mm 不是为了好看。刀柄进料要在
    # 切深 > 伸出量(20) 且 切深 < 料厚 时才成立，料薄于 22 mm 就会先破台面，
    # 于是拦它的是 table_crash，而"由指定的那条规则拦住"就不算数了。
    # 28 mm 同时给两条规则留了空间：holder 在 -22，table 在 -29。
    return [
        # 刀尖直接扎到台面下面
        ('table', 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\n'
         % (sz, mid_x, mid_y, st.table_z - 1.0), 'table_crash'),
        # 刀尖进了压板
        ('clamp_tip', 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\n'
         % (sz, cx, cy, ctop - 1.0), 'clamp_crash'),
        # 刀柄进了料：切到比伸出量还深
        ('holder_in_stock',
         'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\n'
         % (sz, mid_x, mid_y,
            st.top_z - (tool.stickout + 2.0)), 'holder_crash'),
        # 快速从料里横着穿过去
        ('rapid_through',
         'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\nG0 X%.1f Y%.1f\n'
         % (sz, st.x0 + 3, st.y0 + 3, st.top_z - 2.0, st.x1 - 3, st.y1 - 3),
         'rapid_into_stock'),
    ]


def legal_programs(fixture=None):
    """几条**必须放行**的程序。误拒数是门的另一半点，不能只报拦截率。

    这四条覆盖了四种容易把门写坏的地方：直着抬刀、沿料边、圆弧扫到
    压板外侧但不进压板、以及一条压板之间的空地。
    """
    fx = fixture or Fixture()
    st = fx.stock
    sz = fx.safe_z()
    mid_x = (st.x0 + st.x1) / 2.0
    mid_y = (st.y0 + st.y1) / 2.0
    r = fx.tool.radius
    return [
        ('lift_out', 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\n'
         'G1 Z-1.0 F150\nG1 X%.1f F600\nG0 Z%.1f\nM5\n'
         % (sz, mid_x, mid_y, mid_x + 10.0, sz)),
        ('along_edge', 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\n'
         'G1 Z-1.0 F150\nG1 X%.1f Y%.1f F600\nG0 Z%.1f\nM5\n'
         % (sz, st.x0 + r, st.y0 + r, st.x1 - r, st.y0 + r, sz)),
        # 半圆弧：起点 (mx-5, my)，圆心 (mx, my)，所以 I=5 J=0。
        # 写成 I0 J5 会得到一个"终点不在圆上"的几何，解析器会带行号拒绝它
        # ——这次是它先抓到我的，不是反过来。
        ('arc_inside_clearance', 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\n'
         'G1 Z-0.5 F150\nG2 X%.1f Y%.1f I5.0 J0 F400\nG0 Z%.1f\nM5\n'
         % (sz, mid_x - 5.0, mid_y, mid_x + 5.0, mid_y, sz)),
        ('between_clamps', 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\n'
         'G1 Z-2.0 F150\nG1 X%.1f F500\nG1 Y%.1f\nG1 X%.1f\nG0 Z%.1f\nM5\n'
         % (sz, st.x0 + 5.0, st.y0 + 5.0, st.x1 - 5.0, st.y1 - 5.0,
            st.x0 + 5.0, sz)),
    ]
