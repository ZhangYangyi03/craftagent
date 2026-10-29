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


def _signed_area(pts):
    a = 0.0
    n = len(pts)
    for k in range(n):
        x1, y1 = pts[k]
        x2, y2 = pts[(k + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


# 拐角最大放大倍数。4.0 对应转折角约 151°，再尖就宁可留缺口也不把
# 刀心甩到很远的地方——甩出去的那一段会切进件里，而程序完全合法。
MITER_LIMIT = 4.0


def offset_contour(pts, dist, clamped=None):
    """把闭合轮廓沿法向偏 dist（正数向外，负数向内）。

    这里做的是**逐点法向偏移**，不是真正的刀具半径补偿。区别必须说清：

      逐点法向偏移  每个顶点沿"前后两段的平均法向"挪 dist，再把新点连起来。
                    优点：五行程二十行代码，任何形状都能算，不会失败。
                    缺点：**凸角会被拉长、凹角会自交**。偏得越多越明显。

      真正的偏置    需要求等距曲线（直线偏移 + 圆弧偏移 + 自交裁剪），
                    是一个真正的几何算法，最后那步"裁剪"容易写错，
                    而写错的表现是"轮廓多出一小段回头路"，程序仍然合法。

    所以：**这个函数用于"让刀心轨迹离开图纸线"这件事足够，用于
    "保证不残留"不够。** 报告里 offset_kind 会写明是近似，
    不写成 G41/G42 是因为它不是控制器在做，做完之后程序里没有补偿指令，
    机床也就没有机会替我们修正。这条区别写在返回值里，不藏在注释里。
    """
    n = len(pts)
    if n < 3 or abs(dist) < 1e-12:
        return list(pts)
    out = []
    for k in range(n):
        p0 = pts[(k - 1) % n]
        p1 = pts[k]
        p2 = pts[(k + 1) % n]
        d0 = (p1[0] - p0[0], p1[1] - p0[1])
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        n0 = (d0[0] ** 2 + d0[1] ** 2) ** 0.5 or 1.0
        n1 = (d1[0] ** 2 + d1[1] ** 2) ** 0.5 or 1.0
        # 逆时针轮廓的外法向在右手边：(-dy, dx) 是指向左边的法向，
        # 所以向外用 (dy, -dx)。绕向由有符号面积定，不靠假设。
        sgn = 1.0 if _signed_area(pts) > 0 else -1.0
        nu = ((d0[1] / n0 + d1[1] / n1) * sgn, (-d0[0] / n0 - d1[0] / n1) * sgn)
        nn = (nu[0] ** 2 + nu[1] ** 2) ** 0.5 or 1.0
        nu = (nu[0] / nn, nu[1] / nn)
        # 拐角处要放大，否则偏出来的两条边之间会有缺口。
        # miter 系数 = 1/cos(δ/2)，δ 是两段的**转折角**（点积给的是 cos δ）。
        # 第一版把余弦本身夹了下限 0.2，于是直角处算成 1.291 而不是 1.414：
        # 矩形偏 5 mm 得到 4.564，件会小 0.44 mm——一个看不出来、
        # 量出来才知道、而且每次都在的错。夹要夹在**系数**上，不是夹余弦上。
        cos_turn = (d0[0] * d1[0] + d0[1] * d1[1]) / (n0 * n1)
        cos_turn = max(-1.0, min(1.0, cos_turn))
        half_cos = max(1e-6, (0.5 * (1.0 + cos_turn)) ** 0.5)
        scale = 1.0 / half_cos
        if scale > MITER_LIMIT:
            # 极尖的角：不放大到无限远，宁可留一个缺口并把它报出来。
            # 悄悄留缺口会变成"这一角残留一小块料"，而那是最难查的废品。
            scale = MITER_LIMIT
            clamped.append(k)
        out.append((p1[0] + nu[0] * dist * scale, p1[1] + nu[1] * dist * scale))
    return out

def _ramp_in(pts, start_idx, drop, ramp_deg, feed):
    """斜插下刀：沿轮廓走一小段，同时下降到目标 Z。

    直插（G1 Z 直着下去）的问题很具体：立铣刀的端面中心没有切削速度，
    直插就是拿中心去顶料。软材料能顶，钢会顶出一声闷响然后崩刃。
    斜插让刀刃在下降时始终有侧向速度，这是车间里的标准做法。

    返回值是 [(x, y, z, feed)]，z 是**相对本刀起点**的增量，
    最后一行一定落在 -abs(drop) 上。角度为 0 时返回空列表，也就是直插。
    """
    if ramp_deg <= 0 or abs(drop) <= 1e-12 or len(pts) < 3:
        return []
    horizon = abs(drop) / math.tan(math.radians(max(1.0, min(89.0, ramp_deg))))
    n = len(pts)
    out = []
    traveled = 0.0
    k = 0
    x, y = pts[start_idx % n]
    while traveled < horizon - 1e-12 and k <= n:
        nx, ny = pts[(start_idx + k + 1) % n]
        seg = math.hypot(nx - x, ny - y)
        if seg <= 1e-12:
            k += 1
            continue
        take = min(seg, horizon - traveled)
        r = take / seg
        x2 = x + (nx - x) * r
        y2 = y + (ny - y) * r
        z2 = -abs(drop) * min(1.0, (traveled + take) / horizon)
        # 斜插时进给要比切削慢：刀刃同时吃侧面和下面
        out.append((x2, y2, z2, feed * 0.5))
        traveled += take
        x, y = x2, y2
        if take < seg - 1e-12:
            break
        k += 1
    if not out or abs(out[-1][2] + abs(drop)) > 1e-9:
        out.append((x, y, -abs(drop), feed * 0.3))
    return out


def _point_in_poly(pt, poly):
    inside = False
    n = len(poly)
    x, y = pt
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xx:
                inside = not inside
    return inside


def _poly_span(poly):
    xs = [q[0] for q in poly]
    ys = [q[1] for q in poly]
    return (max(xs) - min(xs), max(ys) - min(ys))


def _max_inner_radius(poly, samples=17):
    """这个区域里最大能塞进多大的圆。

    清腔的停止判据必须是它，不是包围盒短边：
    一个 60x15 的窄条型腔短边只有 15，但中心处内切半径能到 7.5，
    用短边判会直接认为"没料了"，而那条窄槽其实还得清。
    第一版就是拿短边判的，20x15 的型腔报 0 圈。
    """
    xs = [q[0] for q in poly]
    ys = [q[1] for q in poly]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    cands = [tuple(q) for q in poly] + [(cx, cy)]
    # 再在包围盒里撒一层网格，避免"最大内切圆不在顶点也不在中心"的形状漏掉
    for i in range(samples):
        for j in range(samples):
            x = min(xs) + (max(xs) - min(xs)) * (i + 0.5) / samples
            y = min(ys) + (max(ys) - min(ys)) * (j + 0.5) / samples
            cands.append((x, y))
    best = 0.0
    for q in cands:
        if not _point_in_poly(q, poly):
            continue
        d = _poly_dist_to_segments(q, poly)
        if d > best:
            best = d
    return best


def _poly_dist_to_segments(pt, poly):
    """点到多边形边界的最短距离。清腔的间距就是靠它判的。"""
    best = float('inf')
    n = len(poly)
    x, y = pt
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 <= 1e-18:
            d = math.hypot(x - x1, y - y1)
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
            d = math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        if d < best:
            best = d
    return best


def pocket_clear_paths(pts, tool_dia, stepover=None, allow=0.0):
    """型腔清腔：走一圈边界之后，把中间那坨料也去掉。

    只沿边界走一圈的问题很具体：型腔中间留着一根柱子，件是废的，
    而程序里每一行都合法、门也放行——因为它只看刀有没有撞夹具、
    有没有超限，它不知道"这块料应该被掏空"。

    做法（水平等距环）：
      1. 边界先往里让一个刀半径（刀心不能出边界）；
      2. 再一层层往里让 stepover，直到剩下的洞小到没有料；
      3. 每一圈都要**验一遍它真的在里面**——逐点法向偏移在凹处会自交，
         自交出去的那一小段会切进型腔壁，而那是不可修的废。

    第 3 步不是保险，是必须的：这个仓的偏置本来就是近似（见
    offset_contour 的说明），近似的东西必须被检查，不能靠"应该没问题"。
    """
    r = tool_dia / 2.0
    step = stepover if stepover else 0.7 * tool_dia
    if step <= 0:
        raise CamError('清腔行距必须为正')
    rings = []
    region = list(pts)
    guard = 0
    while guard < 400:
        guard += 1
        # 收敛判据必须是"再往里让还剩多少地方"。
        # 只判"环上有几个点"不够：两点退化之后 offset_contour 会把它
        # 补成一个 4 点小方块，于是这个循环永远退不出来——
        # 第一版就是这样，跑出 400 圈完全重合的轨迹，而每一圈都"合法"。
        # 停止判据：这一圈之后再往里让，刀心已经放不下了。
        # 用"最大内切半径 <= 刀半径 + 留量"来判，因为那正是
        # "刀心还能不能进到更里面"的定义。
        base_inner = _max_inner_radius(region)
        if base_inner <= r + allow + 1e-9:
            break
        ring = offset_contour(region, -step)
        if len(ring) < 3:
            break
        # 自交/翻出去的一圈不能用：宁可少清一层，也不要过切
        bad = sum(1 for q in ring if not _point_in_poly(q, pts))
        if bad:
            break
        if _max_inner_radius(ring) >= base_inner - 1e-6:
            break           # 没往里走，说明已经到底
        rings.append(ring)
        region = ring
        # 这一圈离**原始型腔壁**还剩多少：不够一个半径说明中心料已去掉
        inner = min(_poly_dist_to_segments(q, pts) for q in ring)
        if inner <= r + 1e-9:
            break
    return list(reversed(rings))  # 由内向外走，最后贴壁

def default_stock(dr, tool=None, ap_per_pass=1.0):
    """推一块够大的毛坯：轮廓包围盒 + 边距，厚度够分刀。

    这个推导只有**一份实现**，build 与 solve 都调它。
    第一版各有各的默认值：build 按包围盒推，solve 建门时用了
    collide.Stock() 的固定 60x84。于是同一次生成里，
    生成器认为料是 (-20,-20,80,60)，门认为料是 (0,0,60,84)，
    一条完全正常的刀路被报成"落在压板里"。拒绝是对的，理由是假的，
    而假理由会把闭环回退引到完全错误的参数上去。
    """
    tool = tool or feeds_mod.Tool(6.0, 3, 'carbide')
    bx0, by0, bx1, by1 = dr.bounds()
    # 边距必须吃下刀心偏置，否则偏出来的轨迹落在料外面：
    # 那里没有料，只有夹具——门会报撞压板，而那是对的。
    pad = tool.diameter_mm + 10.0
    return (bx0 - pad, by0 - pad, bx1 + pad, by1 + pad, 0.0,
            max(10.0, ap_per_pass * 3 + 2.0))


def build(dr, material='P', tool=None, machine=None, stock=None,
          op='rough', ap_per_pass=1.0, safe_z=None, conservative=1.0,
          plan_ok=True, ramp_deg=0.0, offset='none', finish_allow=0.0,
          pocket_stepover=None, clear_pockets=True, offset_engine='approx'):
    """把一张读出来的图纸变成一条程序。

    stock: (x0, y0, x1, y1, top_z, thickness)。给 None 时按轮廓包围盒 + 边距推。
    ap_per_pass: 每一刀的轴向切深。深度大于它就要分多次走。
    ramp_deg: 下刀斜插角度。0 = 直插（钻孔用），3~10 是铣削常用。
    offset: 'none' | 'tool_radius' | 'tool_radius_minus_allow'。
        'tool_radius' 让刀心离开图纸线一个半径，切出来的件才是图纸尺寸。
        注意这是**刀心轨迹的近似偏置**，不是 G41/G42 —— 见 offset_contour 的说明。
        offset='none' 意味着程序里的坐标就是图纸线，也就是刀的**中心**走在
        轮廓上，实际切出来会比图纸小一个半径。这个差别很大，所以它是个
        必须显式选的参数，而不是一个默认值。
    finish_allow: 留量。offset='tool_radius_minus_allow' 时刀心只离开
        (半径 - 留量)，于是留下 finish_allow 的料给精铣。
    clear_pockets: 型腔是否清腔（把中间那坨料也去掉）。
        只沿边界走一圈的话，型腔中间留着一根柱子：件是废的，
        而每一行指令都合法、门也放行——门只看撞不撞夹具、
        超不超限，它不知道"这块料应该被掏空"。
    pocket_stepover: 清腔行距，默认 0.7 倍刀径。
    """
    if not dr.contours:
        raise CamError('图纸里没有可切的闭合轮廓，没有东西可生成')
    tool = tool or feeds_mod.Tool(6.0, 3, 'carbide')
    machine = machine or feeds_mod.Machine()
    bx0, by0, bx1, by1 = dr.bounds()

    if stock is None:
        stock = default_stock(dr, tool, ap_per_pass)
    sx0, sy0, sx1, sy1, top_z, thick = stock
    if safe_z is None:
        safe_z = top_z + 5.0
    if offset_engine not in ('approx', 'exact'):
        # 名字写错不能被当成 approx 悄悄放过：报告里的 kind 会说谎，
        # 而"报告说了是近似"正是这条路唯一的安全阀。
        raise CamError('offset_engine 只能是 approx / exact，收到 %r'
                       % (offset_engine,))

    # 排序：外层先、型腔后，同深度按面积从大到小。
    # 顺序写错的具体后果是：先切了小腔，再切外轮廓时件已经松了，
    # 最后一道把件弹起来——那是废件，而每条指令单独看都合规。
    def key(c):
        return (c.depth, -abs(c.area()))
    contours = sorted(dr.contours, key=key)

    L = ['G21 G90', 'G0 Z%s' % _fmt(safe_z)]
    choices = []
    head = []
    offsets_applied = []
    pockets_cleared = []

    for c in contours:
        target = c.depth
        if abs(target) <= 1e-9:
            raise CamError('轮廓 %s 的层深是 0：没有材料要去除' % c.layer)
        # 刀心轨迹 = 轮廓 + 偏置
        clamped_pts = []
        # 偏置的方向由**这条轮廓的角色**决定，不是由绕向决定。
        # 外轮廓：刀心往外让一个半径，切出来的外缘才是图纸线。
        # 型腔：刀心往**里**让，切出来的内壁才是图纸线。
        # 用绕向来定方向是错的：DXF 里同一条型腔的绕向谁写的都有，
        # 而偏错方向是过切——件报废，程序每一条都合法。
        inward = -1.0 if c.role == 'inner' else 1.0

        def _shift(pts, d):
            """按 offset_engine 选偏置实现。两条路都必须说出自己是哪一条。"""
            if offset_engine == 'exact':
                from . import offset as off_mod
                res = off_mod.offset_polygon(pts, d)
                if not res['ok']:
                    # 报出来，不硬凑：拓扑变了的轮廓（窄槽被腐蚀断）用一条
                    # 闭合折线表示不了，凑出来的那条每一点距离都对、少一半。
                    raise CamError('轮廓 %s 的精确偏置失败（%s，%d 个点，'
                                   '删环 %d 次）—— 这条轮廓偏 %.2f 之后不再是'
                                   '一条闭合曲线，改用 offset_engine=approx '
                                   '或改刀径' % (c.layer, res['reason'],
                                                 len(res['pts']),
                                                 res['loops_removed'], d))
                return res['pts'], 'exact_offset'
            return offset_contour(pts, d, clamped_pts), 'point_normal_approx'

        if offset == 'none':
            path = list(c.pts)
            off = 0.0
            kind = 'none'
        elif offset == 'tool_radius':
            off = tool.diameter_mm / 2.0
            path, kind = _shift(c.pts, inward * off)
        elif offset == 'tool_radius_minus_allow':
            off = tool.diameter_mm / 2.0 - finish_allow
            if off <= 0:
                raise CamError('留量 %.3f 不小于刀半径 %.3f，刀心在件里了'
                               % (finish_allow, tool.diameter_mm / 2.0))
            path, kind = _shift(c.pts, inward * off)
        else:
            raise CamError('offset 只能是 none / tool_radius / tool_radius_minus_allow，'
                           '收到 %r' % (offset,))
        offsets_applied.append({'layer': c.layer, 'role': c.role,
                                'dist': off, 'inward': inward < 0,
                                'kind': kind if off else 'none',
                                'miter_clamped_vertices': clamped_pts})
        # 型腔要清腔：边界那一圈之外，中间还要一圈圈往里走，否则留柱子。
        # 清腔的每一圈都必须已经让到"刀心不出边界"的位置，所以
        # pocket_clear_paths 自己会再往里让一个刀半径——这里传原始轮廓，
        # 不传已经偏过的那条（传偏过的会让出两倍半径，型腔被切大一圈）。
        extra_rings = []
        if clear_pockets and c.role == 'inner' and len(c.pts) >= 3:
            try:
                extra_rings = pocket_clear_paths(
                    c.pts, tool.diameter_mm, pocket_stepover, finish_allow)
            except CamError:
                extra_rings = []
        pocket_meta = {'layer': c.layer, 'clear_rings': len(extra_rings)}
        if c.role == 'inner':
            pockets_cleared.append(pocket_meta)
        passes = [min(ap_per_pass, abs(target))]
        dep = passes[0]
        while dep < abs(target) - 1e-9:
            nxt = min(dep + ap_per_pass, abs(target))
            passes.append(nxt - dep)
            dep = nxt
        z_here = 0.0
        first = True
        for dp in passes:
            prev_z = z_here
            z_here -= dp
            ch = feeds_mod.choose(material, op, tool, machine,
                                  ap_mm=dp, conservative=conservative)
            if first:
                choices.append(ch)
                # 注释用 ';' 不用 '#'：G 代码里 '#' 是参数变量前缀，
                # 这个仓的解析器不认识它，会带行号拒绝。自检第一次就抓到了这个。
                head.append('; %s  %s  ap=%.2f  n=%.0f rpm  vf=%.0f mm/min  '
                            'Pc=%.2f/%.2f kW  刀心偏置=%.2f(%s)  %s'
                            % (c.layer, ch.op, dp, ch.rpm, ch.feed_mm_min,
                               ch.power_kw, ch.power_available_kw, off,
                               '近似' if off else '无',
                               ('夹: ' + ','.join(ch.clamped)) if ch.clamped else '未夹'))
                L.append('M3 S%.0f' % round(ch.rpm))
                L.append('G0 Z%s' % _fmt(safe_z))
                if path:
                    L.append('G0 X%s Y%s' % (_fmt(path[0][0]), _fmt(path[0][1])))
                first = False
            else:
                choices.append(ch)
                if path:
                    L.append('G0 X%s Y%s' % (_fmt(path[0][0]), _fmt(path[0][1])))
            # 同一个 Z 层上要走的环：清腔的先由内向外一圈圈走，
            # 最后才是贴壁那一圈（也就是 path）。
            # 顺序不能反：先贴壁的话，后面几圈会把已经切开的壁再刮一次，
            # 而且中间那坨料还挂在壁上，容易把刀挤偏。
            for ring in list(extra_rings) + [path]:
                if len(ring) < 2:
                    continue
                L.append('G0 X%s Y%s' % (_fmt(ring[0][0]), _fmt(ring[0][1])))
                # 下刀：直插或斜插。斜插会吃掉环开头的一截，
                # 那一截不能重复走——从斜插结束的位置接着走。
                n = len(ring)
                start_idx = 0
                if ramp_deg > 0 and n >= 3:
                    ramp = _ramp_in(ring, 0, dp, ramp_deg, ch.feed_mm_min)
                    # _ramp_in 给的 z 是相对本刀起点（0 -> -dp），
                    # 加上 prev_z 才是机床的绝对坐标。
                    for rx, ry, rz, rf in ramp:
                        L.append('G1 X%s Y%s Z%s F%.0f'
                                 % (_fmt(rx), _fmt(ry), _fmt(prev_z + rz), round(rf)))
                    endpt = (ramp[-1][0], ramp[-1][1]) if ramp else ring[0]
                    start_idx = min(range(n), key=lambda k: math.dist(ring[k], endpt))
                else:
                    L.append('G1 Z%s F%.0f'
                             % (_fmt(z_here), round(ch.feed_mm_min * 0.3)))
                for k in range(start_idx + 1, start_idx + n + 1):
                    x, y = ring[k % n]
                    L.append('G1 X%s Y%s F%.0f'
                             % (_fmt(x), _fmt(y), round(ch.feed_mm_min)))
                # 一圈走完抬到安全高度，再去下一个环的起点。
                # 不抬起来的后果是"贴着件横穿过去"，那正是撞夹具的那一刀。
                L.append('G0 Z%s' % _fmt(safe_z))
    L.append('M5')
    text = '\n'.join(head + L) + '\n'
    plan = Plan(text, choices, {'contours': len(contours), 'stock': stock,
                                'material': material, 'op': op,
                                'tool': tool.name, 'safe_z': safe_z,
                                'ap_per_pass': ap_per_pass,
                                'ramp_deg': ramp_deg, 'offset': offset,
                                'finish_allow': finish_allow,
                                'pocket_stepover': pocket_stepover,
                                'clear_pockets': clear_pockets,
                                'pockets': pockets_cleared,
                                'offsets': offsets_applied})
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
          floor=0.15, steps=1.35, offset='none', finish_allow=0.0,
          ramp_deg=0.0):
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
    # 门用的料和生成器用的料**必须是同一块**。第一版这里各自用默认值：
    # build 按轮廓包围盒 + 边距推毛坯，门用 Stock() 的固定 60x84。
    # 于是"刀在毛坯外面"被报成 clamp_crash——拒绝是对的，理由是假的，
    # 而假理由会把回退逻辑引到完全错误的参数上去。
    if stock is None:
        stock = default_stock(dr, tool, ap_per_pass)
    sx0, sy0, sx1, sy1, top_z, thick = stock
    fx = collide_mod.Fixture(stock=collide_mod.Stock(
        sx0, sy0, sx1, sy1, top_z, thick))
    gate = gate_mod.Gate(limits, fixture=fx)

    k = conservative
    attempts = []
    for i in range(max_tries):
        plan = build(dr, material, tool, machine, stock=stock, op=op,
                     ap_per_pass=ap_per_pass, conservative=k,
                     offset=offset, finish_allow=finish_allow,
                     ramp_deg=ramp_deg)
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
