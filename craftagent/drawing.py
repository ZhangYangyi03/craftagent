# -*- coding: utf-8 -*-
"""drawing.py -- 从 2D 图纸（DXF）读出"要切出一个什么形状"。

为什么自己写 DXF 解析器，而不是用 ezdxf：

  * 本机 numpy 导入直接 MemoryError（这不是猜的，是把 traceback 打出来看的），
    而 ezdxf 依赖 numpy，所以它在这台机器上根本起不来。
  * 更要紧的是这个仓从第一天起就只用标准库：工厂那台电脑上少一个依赖，
    就少一个装不上的理由。一个"必须先 pip install 才能看图纸"的工具，
    在车间里等于不能用。

这个解析器只认它要量的东西：LINE / ARC / CIRCLE / LWPOLYLINE / POLYLINE。
读不懂的实体不是静默跳过，而是**记进 unsupported**——计划器看不见的轮廓，
后面的门也拦不住，那"它跑起来了"就什么也不意味着。

    DXF 是"组码 + 值"成对出现的纯文本：
        0
        LWPOLYLINE
        8
        CUT
        10
        0.0
        20
        0.0
    所以解析它不需要几何库，只需要一个老老实实的两行一读的循环。

图纸到 3D 的那一步，这里的做法是 2.5D：每个闭合轮廓是一个"层深"，
层深写在**图层名**里（CUT_-1.5 就是切到 -1.5 mm）。这不是唯一约定，
但它是一条可以被机械检查的约定：图层名里没有深度的轮廓，
这个模块会报出来，而不是猜一个 0。
"""
import math
import re


class DrawingError(Exception):
    pass


# ---------------------------------------------------------------- DXF 底层
def pairs(text):
    """把 DXF 文本读成 (组码, 值) 序列。

    组码行必须是整数；不是的话说明这不是 ASCII DXF（可能是二进制 DXF，
    或者根本不是 DXF）。两种情况都该在第一时间拒绝，而不是读出一堆垃圾之后
    再报一个看不懂的错。
    """
    lines = text.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i + 1 < n:
        code_s = lines[i].strip()
        val = lines[i + 1].rstrip('\r\n')
        i += 2
        if not code_s:
            continue
        try:
            code = int(code_s)
        except ValueError:
            raise DrawingError('第 %d 行不是组码（%r）：这不是 ASCII DXF'
                               % (i - 1, code_s[:40]))
        out.append((code, val))
    return out


def sections(ps):
    """按 SECTION/ENDSEC 切开，返回 {section_name: [(code, value), ...]}。

    段**没有 ENDSEC 结尾**时也收下（见末尾的 flush）。理由：真实图纸
    会被截断——从 U 盘拷一半、从邮件附件里断掉。截断的文件里 ENTITIES
    仍然可读，而"一个段都没闭合所以整张图没有 ENTITIES"是一句
    比事实更严重的谎。真正的空图纸说"没有 ENTITIES"才是对的。
    """
    out, cur, name = {}, None, None
    for code, val in ps:
        if code == 0 and val.strip() == 'SECTION':
            cur = []
            name = None
        elif code == 0 and val.strip() == 'ENDSEC':
            if name is not None and cur is not None:
                out.setdefault(name, []).extend(cur)
            cur, name = None, None
        elif code == 2 and cur is not None and name is None:
            name = val.strip()
        elif cur is not None:
            cur.append((code, val))
    if cur is not None and name is not None:
        out.setdefault(name, []).extend(cur)
    return out


# ---------------------------------------------------------------- 实体
class Contour(object):
    """一条闭合轮廓：若干线段（直线或圆弧），加一个层深。"""

    def __init__(self, layer, depth, pts, source=''):
        self.layer = layer
        self.depth = depth
        self.pts = pts              # [(x, y), ...]，首尾相接，不重复首点
        self.source = source
        self.assumed_junctions = 0  # 拼环时按"最小转角"猜过几次

    def area(self):
        """鞋带公式。符号说明绕向，绝对值是面积。"""
        a = 0.0
        n = len(self.pts)
        for i in range(n):
            x1, y1 = self.pts[i]
            x2, y2 = self.pts[(i + 1) % n]
            a += x1 * y2 - x2 * y1
        return 0.5 * a

    def perimeter(self):
        return sum(math.dist(self.pts[i], self.pts[(i + 1) % len(self.pts)])
                   for i in range(len(self.pts)))

    def bounds(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)

    def __repr__(self):
        extra = (' guess=%d' % self.assumed_junctions) if self.assumed_junctions else ''
        return '<Contour %s z=%.2f %d pts area=%.1f%s>' % (
            self.layer, self.depth, len(self.pts), self.area(), extra)


# 图层名里带深度的三种常见写法：CUT_-1.5 / CUT-1.5 / CUT_D1.5
_DEPTH_RE = re.compile(r'([-_]D?)(-?\d+(?:\.\d+)?)\s*$')


def depth_from_layer(layer):
    """从图层名读层深。读不出就是 None——**不猜 0**。

    猜 0 的后果很具体：一条深度不明的轮廓会被当成"在表面上划一刀"，
    门会放行，件是废的，而所有报表都是绿的。
    """
    m = _DEPTH_RE.search(layer or '')
    if not m:
        return None
    return float(m.group(2))


def chord_steps(radius, sweep_rad, tolerance_mm=0.01):
    """按弦高容差定采样数，而不是按固定角度。

    固定 5 度对 D8 的孔是 73 个点，对 D200 的圆弧也是 73 个点——
    前者浪费、后者超差。弦高 h = r(1 - cos(dtheta/2))，
    所以 dtheta = 2*acos(1 - tol/r)，采样数由**容差**决定。

    tolerance_mm 取 0.01：比这台机床的圆度规格（8.5 um）宽一点，
    也就是"折线化带来的误差不主导质量"。这个关系写在这里，
    因为把容差取成 0.5 mm 会省一半代码行，同时让所有孔变成多边形。
    """
    if radius <= 0 or sweep_rad <= 0:
        return 1
    if tolerance_mm >= radius:
        return 1
    dth = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - tolerance_mm / radius)))
    if dth <= 1e-9:
        return 1
    return max(2, int(math.ceil(abs(sweep_rad) / dth)))


def _arc_points(cx, cy, r, a0_deg, a1_deg, tolerance_mm=0.01):
    """把圆弧采样成折线。角度按 DXF 的逆时针。"""
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    while a1 <= a0:
        a1 += 2 * math.pi
    sweep = a1 - a0
    n = chord_steps(r, sweep, tolerance_mm)
    return [(cx + r * math.cos(a0 + sweep * k / n),
             cy + r * math.sin(a0 + sweep * k / n)) for k in range(n + 1)]


def _bulge_arc(p0, p1, bulge, tolerance_mm=0.01):
    """LWPOLYLINE 的 bulge 段 -> 折线。bulge = tan(转角/4)。"""
    if abs(bulge) < 1e-12:
        return [p0, p1]
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    chord = math.hypot(dx, dy)
    if chord <= 1e-12:
        return [p0]
    theta = 4.0 * math.atan(bulge)          # 包含角
    r = chord / (2.0 * math.sin(abs(theta) / 2.0))
    # 圆心在弦的中垂线上
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    h = r * math.cos(abs(theta) / 2.0) * (1.0 if abs(theta) > math.pi else -1.0)
    nx, ny = -dy / chord, dx / chord
    sign = 1.0 if bulge > 0 else -1.0
    cx, cy = mx + sign * h * nx, my + sign * h * ny
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    if bulge > 0:
        while a1 <= a0:
            a1 += 2 * math.pi
    else:
        while a1 >= a0:
            a1 -= 2 * math.pi
    sweep = a1 - a0
    n = chord_steps(r, sweep, tolerance_mm)
    out = [(cx + r * math.cos(a0 + sweep * k / n),
            cy + r * math.sin(a0 + sweep * k / n)) for k in range(n + 1)]
    out[0] = (x0, y0)
    out[-1] = (x1, y1)
    return out


class Drawing(object):
    """一张图纸读出来的结果：轮廓、毛坯、以及读不懂的东西。"""

    def __init__(self):
        self.contours = []
        self.unsupported = []       # [(entity_type, layer, 原因)]
        self.unknown_depth = []     # [(layer, 点数)]
        self.entities_seen = {}
        self.open_edges = []        # 待拼环的开链（LINE/ARC）
        self.stitch = None          # StitchReport，拼过才有

    def bounds(self):
        xs, ys = [], []
        for c in self.contours:
            b = c.bounds()
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def summary(self):
        d = {'contours': len(self.contours),
             'entities': dict(self.entities_seen),
             'unsupported': list(self.unsupported),
             'unknown_depth': list(self.unknown_depth),
             'bounds': self.bounds()}
        if self.stitch is not None:
            d['stitch'] = self.stitch.as_dict()
        return d


def read_dxf(text, stitch_tol_mm=0.01):
    """读 ASCII DXF，返回 Drawing。

    只读 ENTITIES 段。BLOCKS 里的东西不展开——块引用（INSERT）被记成
    unsupported，因为这需要一张块表加变换栈，而这个仓现在不做。
    不展开就说没展开，而不是假装图纸是空的。
    """
    ps = pairs(text)
    secs = sections(ps)
    dr = Drawing()
    ents = secs.get('ENTITIES')
    if ents is None:
        raise DrawingError('DXF 里没有 ENTITIES 段')

    # 按实体切开
    cur_type, cur = None, []
    items = []
    for code, val in ents:
        if code == 0:
            if cur_type is not None:
                items.append((cur_type, cur))
            cur_type, cur = val.strip(), []
        else:
            cur.append((code, val))
    if cur_type is not None:
        items.append((cur_type, cur))

    for etype, body in items:
        dr.entities_seen[etype] = dr.entities_seen.get(etype, 0) + 1
        layer = ''
        nums = {}
        verts = []
        bulges = []
        # 10/20 与 11/21 是两组不同的点：LINE 的起点与终点、CIRCLE 的圆心
        # 与（不用的）第二点。第一版把 11/21 塞进了 nums，于是每条 LINE
        # 只读到起点，报成"少于两个端点"——报错是对的，原因说的是假的。
        second = [None, None]
        # LWPOLYLINE 的顶点是 (10,20) 成对**按出现顺序**交替出现的，
        # 不是"20 配最近一个还缺 y 的 10"。第一版按后者写，
        # 于是顶点 (30,25) 被读成 (30,None)+(25,None) 的错配：
        # 三个顶点的型腔报成"少于 3 个点"。报错是对的，原因说的是假的。
        # 这里用一个开关：10 开一个新顶点，紧跟着的 20 填它的 y。
        pending_y = False
        for code, val in body:
            if code == 8:
                layer = val.strip()
            elif code == 10:
                verts.append([_f(val), None])
                pending_y = True
            elif code == 20 and pending_y and verts:
                verts[-1][1] = _f(val)
                pending_y = False
            elif code == 11:
                second[0] = _f(val)
            elif code == 21:
                second[1] = _f(val)
                pending_y = False
            elif code == 42:
                bulges.append(_f(val))
            elif code in (40, 41, 50, 51):
                nums[code] = _f(val)
            elif code in (70, 90):
                nums[code] = int(_f(val))
        verts = [(x, y) for x, y in verts if y is not None]
        if second[0] is not None and second[1] is not None:
            end_pt = (second[0], second[1])
        else:
            end_pt = None

        if etype == 'LINE':
            pts = []
            if verts:
                pts.append(verts[0])
            if end_pt is not None:
                pts.append(end_pt)
            if len(pts) < 2:
                dr.unsupported.append((etype, layer, 'LINE 少于两个端点'))
                continue
            _add_open(dr, layer, pts, etype)
        elif etype == 'ARC':
            if len(verts) < 1 or 40 not in nums:
                dr.unsupported.append((etype, layer, 'ARC 缺圆心或半径'))
                continue
            (cx, cy) = verts[0]
            a0 = nums.get(50, 0.0)
            a1 = nums.get(51, 360.0)
            pts = _arc_points(cx, cy, nums[40], a0, a1)
            _add_open(dr, layer, pts, etype)
        elif etype == 'CIRCLE':
            if len(verts) < 1 or 40 not in nums:
                dr.unsupported.append((etype, layer, 'CIRCLE 缺圆心或半径'))
                continue
            (cx, cy) = verts[0]
            pts = _arc_points(cx, cy, nums[40], 0.0, 360.0)
            pts = pts[:-1]          # 闭合轮廓不重复首点
            _add_closed(dr, layer, pts, etype)
        elif etype == 'LWPOLYLINE':
            if len(verts) < 2:
                dr.unsupported.append((etype, layer, 'LWPOLYLINE 少于两个顶点'))
                continue
            closed = bool(nums.get(70, 0) & 1)
            pts = []
            n = len(verts)
            for i in range(n):
                p0 = verts[i]
                p1 = verts[(i + 1) % n] if closed else verts[(i + 1) % n if i + 1 < n else i]
                if i + 1 >= n and not closed:
                    pts.append(p0)
                    break
                b = bulges[i] if i < len(bulges) else 0.0
                seg = _bulge_arc(p0, p1, b)
                pts.extend(seg[:-1])
            if closed:
                _add_closed(dr, layer, pts, etype)
            else:
                _add_open(dr, layer, pts, etype)
        elif etype in ('INSERT', 'SPLINE', 'ELLIPSE', '3DFACE', 'HATCH', 'TEXT', 'MTEXT',
                       'DIMENSION', 'LEADER', 'POINT', 'SOLID'):
            if etype in ('TEXT', 'MTEXT', 'DIMENSION', 'LEADER', 'POINT'):
                continue        # 标注与文字不参与切削，跳过是对的
            dr.unsupported.append((etype, layer, '这个实体类型没有实现'))
        else:
            dr.unsupported.append((etype, layer, '不认识的实体类型'))

    if dr.open_edges:
        contours, rep = stitch(dr.open_edges, stitch_tol_mm)
        dr.contours.extend(contours)
        dr.stitch = rep
    return dr


def _f(s):
    try:
        return float(s)
    except ValueError:
        raise DrawingError('期望一个数，读到 %r' % (s[:40],))


def _layer_depth(dr, layer):
    d = depth_from_layer(layer)
    if d is None:
        dr.unknown_depth.append((layer, 0))
    return d


def _add_closed(dr, layer, pts, etype):
    d = _layer_depth(dr, layer)
    if d is None:
        return
    if len(pts) < 3:
        dr.unsupported.append((etype, layer, '闭合轮廓少于 3 个点'))
        return
    dr.contours.append(Contour(layer, d, pts, etype))


def _add_open(dr, layer, pts, etype):
    """LINE/ARC 是开的：单独一条切不出件，但一堆首尾相接的能。

    攒完才收（见本文件末尾的 stitch()）：这里只把它放进待拼的筐里。
    """
    dr.open_edges.append(Edge(pts, layer, etype))


# ---------------------------------------------------------------- 开链拼环
# 真实图纸里绝大多数轮廓不是 LWPOLYLINE，而是一堆 LINE/ARC 首尾相接。
# 不拼环就等于"这张图纸没有可切的轮廓"——而那通常是假的。
#
# 拼环这件事有一个必须做出的决定，写在这里而不是藏在代码里：
#
#     分叉怎么办？三条边共用一个端点时，走哪条？
#
# 这台机器上做的是"按最小转角继续"（也就是车间里人眼会做的那个选择），
# 并且**把每一次这样的猜测都数出来**（Contour.assumed_junctions）。
# 理由：一个悄悄挑了一条边的拼接器，会生出一条看起来完全正常的轮廓，
# 而它是错的；报告里必须有一行说"这条轮廓经过了 N 个分叉，每个都是猜的"。
# 不猜（遇到分叉就放弃）在这里更糟：真实图纸会因为一个无关的标注端点
# 就整张拼不出来，而"拼不出来"和"这里分叉了"是两条不同的信息。
class StitchReport(object):
    """拼接的结果，包括它没能拼成什么。"""

    def __init__(self):
        self.layers = {}
        self.open_chains = []       # [(layer, 点数, 两端坐标, 缺多少)]
        self.junctions = []          # [(layer, x, y, 边数)]
        self.self_intersect_unchecked = True

    def note_layer(self, layer):
        d = self.layers.setdefault(layer, {'edges': 0, 'cycles': 0,
                                           'open': 0, 'junctions': 0,
                                           'assumed': 0})
        return d

    def as_dict(self):
        return {'layers': self.layers,
                'open_chains': [{'layer': l, 'points': n, 'gap_mm': round(g, 6),
                                 'ends': a} for l, n, a, g in self.open_chains],
                'junctions': [{'layer': l, 'at': [round(x, 4), round(y, 4)],
                               'edges': k} for l, x, y, k in self.junctions],
                'self_intersect_unchecked': self.self_intersect_unchecked}


class Edge(object):
    """一条开链：有序的点，两端是端点。"""

    def __init__(self, pts, layer, etype):
        self.pts = pts
        self.layer = layer
        self.etype = etype

    def first_dir(self, from_start):
        """从某一端出发的方向（单位向量），用来算转角。"""
        pts = self.pts if from_start else list(reversed(self.pts))
        for q in pts[1:]:
            dx = q[0] - pts[0][0]
            dy = q[1] - pts[0][1]
            n = math.hypot(dx, dy)
            if n > 1e-12:
                return (dx / n, dy / n)
        return (0.0, 0.0)


class _Nodes(object):
    """在容差内把端点归并成节点。

    只按量化格子查会把"恰好跨在格子边界上的同一个点"拆成两个，
    于是本该闭合的环报成开链——而那个报错的样子非常像图纸有问题。
    所以查 3x3 邻域，用真实距离判。
    """

    def __init__(self, tol):
        self.tol = tol
        self.pts = []
        self.grid = {}

    def get(self, pt):
        kx, ky = int(round(pt[0] / self.tol)), int(round(pt[1] / self.tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.grid.get((kx + dx, ky + dy), ()):
                    if math.dist(self.pts[idx], pt) <= self.tol:
                        return idx
        idx = len(self.pts)
        self.pts.append(pt)
        self.grid.setdefault((kx, ky), []).append(idx)
        return idx


def stitch(edges, tol_mm=0.01, report=None):
    """把一批开链拼成闭合轮廓。返回 (contours, report)。

    只在不跨图层的前提下拼：不同图层是不同的层深，跨层拼接会悄悄把
    半条轮廓的深度改掉，而改完之后它看起来完全正常。
    """
    report = report or StitchReport()
    by_layer = {}
    for e in edges:
        if len(e.pts) < 2:
            continue
        by_layer.setdefault(e.layer, []).append(e)

    out = []
    for layer in sorted(by_layer):
        es = by_layer[layer]
        note = report.note_layer(layer)
        note['edges'] = len(es)
        depth = depth_from_layer(layer)
        if depth is None:
            # 深度不明就不拼：拼出来一条深度是错的轮廓，比拼不出来更糟。
            report.open_chains.append((layer, sum(len(e.pts) for e in es), None, 0.0))
            note['open'] = len(es)
            continue

        nodes = _Nodes(tol_mm)
        ea, eb = [], []
        for e in es:
            ea.append(nodes.get(e.pts[0]))
            eb.append(nodes.get(e.pts[-1]))

        # 邻接：节点 -> [(边号, 是否从起点进)]
        adj = {}
        for i in range(len(es)):
            adj.setdefault(ea[i], []).append((i, True))
            adj.setdefault(eb[i], []).append((i, False))

        used = [False] * len(es)
        for i in range(len(es)):
            if used[i]:
                continue
            used[i] = True
            chain = list(es[i].pts)
            start_node = ea[i]
            cur_node = eb[i]
            cur_pt = es[i].pts[-1]
            assumed = 0
            closed = False
            while True:
                if cur_node == start_node:
                    closed = True
                    break
                cands = [(j, from_start) for (j, from_start) in adj.get(cur_node, ())
                         if not used[j]]
                if not cands:
                    break
                if len(cands) > 1:
                    # 分叉：按最小转角继续，并**数出来**。见文件头的说明。
                    note['junctions'] += 1
                    report.junctions.append((layer, nodes.pts[cur_node][0],
                                             nodes.pts[cur_node][1], len(cands) + 1))
                    assumed += 1
                    prev = chain[-2] if len(chain) >= 2 else chain[-1]
                    din = (chain[-1][0] - prev[0], chain[-1][1] - prev[1])
                    n = math.hypot(din[0], din[1]) or 1.0
                    din = (din[0] / n, din[1] / n)
                    scored = []
                    for (j, from_start) in cands:
                        d = es[j].first_dir(from_start)
                        scored.append((din[0] * d[0] + din[1] * d[1], j, from_start))
                    scored.sort(reverse=True)
                    cands = [(scored[0][1], scored[0][2])]
                j, from_start = cands[0]
                used[j] = True
                seg = es[j].pts if from_start else list(reversed(es[j].pts))
                chain.extend(seg[1:])
                cur_node = eb[j] if from_start else ea[j]
                cur_pt = chain[-1]
            if closed and len(chain) >= 4:
                if math.dist(chain[0], chain[-1]) <= tol_mm:
                    chain = chain[:-1]
                c = Contour(layer, depth, chain, 'stitched')
                c.assumed_junctions = assumed
                out.append(c)
                note['cycles'] += 1
                note['assumed'] += assumed
            else:
                report.open_chains.append(
                    (layer, len(chain),
                     (tuple(round(v, 4) for v in chain[0]),
                      tuple(round(v, 4) for v in chain[-1])),
                     math.dist(chain[0], chain[-1])))
                note['open'] += 1
    return out, report


def _add_open(dr, layer, pts, etype):
    """LINE/ARC 是开的：单独一条切不出件，但一堆首尾相接的能。

    攒完才收（见本文件末尾的 stitch()）：这里只把它放进待拼的筐里。
    """
    dr.open_edges.append(Edge(pts, layer, etype))


# ---------------------------------------------------------------- 开链拼环
# 真实图纸里绝大多数轮廓不是 LWPOLYLINE，而是一堆 LINE/ARC 首尾相接。
# 不拼环就等于"这张图纸没有可切的轮廓"——而那通常是假的。
#
# 拼环这件事有一个必须做出的决定，写在这里而不是藏在代码里：
#
#     分叉怎么办？三条边共用一个端点时，走哪条？
#
# 这台机器上做的是"按最小转角继续"（也就是车间里人眼会做的那个选择），
# 并且**把每一次这样的猜测都数出来**（Contour.assumed_junctions）。
# 理由：一个悄悄挑了一条边的拼接器，会生出一条看起来完全正常的轮廓，
# 而它是错的；报告里必须有一行说"这条轮廓经过了 N 个分叉，每个都是猜的"。
# 不猜（遇到分叉就放弃）在这里更糟：真实图纸会因为一个无关的标注端点
# 就整张拼不出来，而"拼不出来"和"这里分叉了"是两条不同的信息。
class StitchReport(object):
    """拼接的结果，包括它没能拼成什么。"""

    def __init__(self):
        self.layers = {}
        self.open_chains = []       # [(layer, 点数, 两端坐标, 缺多少)]
        self.junctions = []          # [(layer, x, y, 边数)]
        self.self_intersect_unchecked = True

    def note_layer(self, layer):
        d = self.layers.setdefault(layer, {'edges': 0, 'cycles': 0,
                                           'open': 0, 'junctions': 0,
                                           'assumed': 0})
        return d

    def as_dict(self):
        return {'layers': self.layers,
                'open_chains': [{'layer': l, 'points': n, 'gap_mm': round(g, 6),
                                 'ends': a} for l, n, a, g in self.open_chains],
                'junctions': [{'layer': l, 'at': [round(x, 4), round(y, 4)],
                               'edges': k} for l, x, y, k in self.junctions],
                'self_intersect_unchecked': self.self_intersect_unchecked}


class Edge(object):
    """一条开链：有序的点，两端是端点。"""

    def __init__(self, pts, layer, etype):
        self.pts = pts
        self.layer = layer
        self.etype = etype

    def first_dir(self, from_start):
        """从某一端出发的方向（单位向量），用来算转角。"""
        pts = self.pts if from_start else list(reversed(self.pts))
        for q in pts[1:]:
            dx = q[0] - pts[0][0]
            dy = q[1] - pts[0][1]
            n = math.hypot(dx, dy)
            if n > 1e-12:
                return (dx / n, dy / n)
        return (0.0, 0.0)


def _key(pt, tol):
    return (int(round(pt[0] / tol)), int(round(pt[1] / tol)))


def _find_node(nodes, pt, tol):
    """在容差内找一个已有节点。先在量化格子里找，再找 3x3 邻域。

    只查同一个格子会把"恰好跨在格子边界上的同一个点"拆成两个，于是
    一个本该闭合的环报成开链——那个报错的样子非常像图纸有问题。
    """
    kx, ky = _key(pt, tol)
    for dx in (0, -1, 1):
        for dy in (0, -1, 1):
            for cand in nodes.get((kx + dx, ky + dy), ()):
                if math.dist(cand[0], pt) <= tol:
                    return cand[0]
    return None
