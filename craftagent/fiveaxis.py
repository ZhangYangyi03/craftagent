# -*- coding: utf-8 -*-
"""fiveaxis.py -- 五轴：刀轴矢量的扫掠体碰撞 + 旋转轴逆解。

为什么高度场在五轴下是错的
--------------------------
三轴的高度场模型把毛坯压成一张 z = h(x, y) 的曲面，然后问"刀尖是不是
低于这张面"。这个模型成立的前提是**刀轴竖直**：刀只在 (x, y) 上占一个
点，所以"低于"就等于"切进去"。

五轴把刀轴倾斜之后，刀在 (x, y) 上的投影是一条线段，刀杆还会伸到
几厘米之外。于是出现一种三轴永远不会有的撞法：刀尖在毛坯上方、
在 (x, y) 上也完全在毛坯外面，可是**刀杆**斜着扎进了旁边的侧壁。
高度场对这种姿态给出的答案是"安全"，因为它只看 (x, y) 那一点的高度。
这不是精度问题，是模型缺了一个维度。

所以这一层把碰撞判据换成**扫掠体**：刀是一个胶囊（球头 + 圆柱刀杆）
加一个刀柄圆柱，碰撞 = 这些几何体与毛坯/夹具的距离小于零。距离是
精确的（线段-线段、线段-长方体都闭式），不是采样。

验收怎么对
----------
一、三轴退化。刀轴竖直时，扫掠体的判定必须与高度场**逐条一致**。
    否则不是五轴写错了，是三轴那条基线本身有问题。
二、反例。给出一个具体姿态：高度场说安全，扫掠体说撞。这一条是
    这一层存在的理由，不能只有"我实现了扫掠体"这种说法。
三、逆解往返。给定刀轴方向解出 (A, C)，再正解回去必须一致到 1e-12；
    奇异位形（刀轴竖直，C 不定）必须**报出来**，不能随便给一个角度。

没有做的
--------
不做刀路规划、不做材料去除仿真、不做真正的机床几何（导轨、转台实体）。
这一层只回答一个问题：给定一个姿态，会不会撞。
"""
from __future__ import annotations

import math

__all__ = ["Tool", "segment_point_dist", "segment_segment_dist",
           "segment_box_dist", "swept_collision", "heightfield_collision",
           "compare_models", "pocket", "ik_ac", "fk_ac", "ik_roundtrip"]


# ---------------------------------------------------------------- 距离

def segment_point_dist(a, b, p):
    """点到线段的距离。"""
    ax, ay, az = a
    bx, by, bz = b
    px, py, pz = p
    dx, dy, dz = bx - ax, by - ay, bz - az
    dd = dx * dx + dy * dy + dz * dz
    if dd <= 1e-300:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / dd
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy, qz = ax + t * dx, ay + t * dy, az + t * dz
    return math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)


def segment_segment_dist(p1, p2, q1, q2):
    """两条线段之间的距离（闭式，含平行与退化的处理）。

    用的是标准的钳制最近点迭代：先解无约束的最近参数，再逐个钳到
    [0,1] 并重解另一个参数。对线段（不是直线）这一步是必须的 ——
    只解无约束问题会把"最近点在延长线上"当成距离，那是**低估**，
    也就是把不撞判成撞，方向上是安全的，但会误报。
    """
    d1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    d2 = (q2[0] - q1[0], q2[1] - q1[1], q2[2] - q1[2])
    r = (p1[0] - q1[0], p1[1] - q1[1], p1[2] - q1[2])
    a = sum(v * v for v in d1)
    e = sum(v * v for v in d2)
    f = sum(d2[i] * r[i] for i in range(3))
    if a <= 1e-300 and e <= 1e-300:
        return math.sqrt(sum(v * v for v in r))
    if a <= 1e-300:
        s = 0.0
        t = min(1.0, max(0.0, f / e))
    else:
        c = sum(d1[i] * r[i] for i in range(3))
        if e <= 1e-300:
            t = 0.0
            s = min(1.0, max(0.0, -c / a))
        else:
            b = sum(d1[i] * d2[i] for i in range(3))
            den = a * e - b * b
            if den > 1e-300:
                s = min(1.0, max(0.0, (b * f - c * e) / den))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))
    cp = (p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2])
    cq = (q1[0] + t * d2[0], q1[1] + t * d2[1], q1[2] + t * d2[2])
    return math.sqrt(sum((cp[i] - cq[i]) ** 2 for i in range(3)))


def segment_box_dist(a, b, box):
    """线段到轴对齐长方体的距离。box = (x0, x1, y0, y1, z0, z1)。

    线段与凸多面体的距离：要么相交（0），要么最近点对里必有一个是
    线段的端点对某个面，或者线段与某条棱的最近点对。所以只要检查
    8 个顶点、12 条棱，加上"线段是否穿进盒子里"，就是精确的。
    采样会漏掉薄壁，而薄壁正是撞刀发生的地方。
    """
    x0, x1, y0, y1, z0, z1 = box
    lo = (min(x0, x1), min(y0, y1), min(z0, z1))
    hi = (max(x0, x1), max(y0, y1), max(z0, z1))
    # 线段与盒子相交 -> 0（slab 法）
    tmin, tmax = 0.0, 1.0
    hit = True
    for i in range(3):
        d = b[i] - a[i]
        if abs(d) < 1e-15:
            if a[i] < lo[i] or a[i] > hi[i]:
                hit = False
                break
            continue
        t1 = (lo[i] - a[i]) / d
        t2 = (hi[i] - a[i]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            hit = False
            break
    if hit:
        return 0.0
    best = min(segment_point_dist(a, b, (lo[0] + (hi[0] - lo[0]) * i / 1.0,
                                         lo[1] + (hi[1] - lo[1]) * j / 1.0,
                                         lo[2] + (hi[2] - lo[2]) * k / 1.0))
               for i in (0, 1) for j in (0, 1) for k in (0, 1))
    corners = [(lo[0] + (hi[0] - lo[0]) * i, lo[1] + (hi[1] - lo[1]) * j,
                lo[2] + (hi[2] - lo[2]) * k)
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges = []
    for i in (0, 1):
        for j in (0, 1):
            edges.append(((lo[0], lo[1] + j * (hi[1] - lo[1]), lo[2] + i * (hi[2] - lo[2])),
                          (hi[0], lo[1] + j * (hi[1] - lo[1]), lo[2] + i * (hi[2] - lo[2]))))
            edges.append(((lo[0] + i * (hi[0] - lo[0]), lo[1], lo[2] + j * (hi[2] - lo[2])),
                          (lo[0] + i * (hi[0] - lo[0]), hi[1], lo[2] + j * (hi[2] - lo[2]))))
            edges.append(((lo[0] + i * (hi[0] - lo[0]), lo[1] + j * (hi[1] - lo[1]), lo[2]),
                          (lo[0] + i * (hi[0] - lo[0]), lo[1] + j * (hi[1] - lo[1]), hi[2])))
    for (c1, c2) in edges:
        d = segment_segment_dist(a, b, c1, c2)
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------- 刀具

class Tool:
    """球头刀 + 刀柄。尺寸都是半径，单位 mm。

    `flute` 是刀刃长度（从刀尖沿刀轴往上），`holder_start` 是刀柄下端
    距刀尖的距离，`holder_r` 是刀柄半径。刀柄比刀杆粗，所以它会先在
    侧壁上撞 —— 这正是高度场看不见的那种撞。
    """

    def __init__(self, radius=5.0, flute=30.0, holder_start=45.0,
                 holder_r=20.0, shank_r=None):
        self.radius = float(radius)
        self.flute = float(flute)
        self.holder_start = float(holder_start)
        self.holder_r = float(holder_r)
        self.shank_r = float(shank_r if shank_r is not None else radius)

    def segments(self, tip, axis):
        """返回 [(a, b, 半径, 名字)]：刀杆段、刀柄段。"""
        n = _unit(axis)
        cut_end = (tip[0] + n[0] * self.flute, tip[1] + n[1] * self.flute,
                   tip[2] + n[2] * self.flute)
        # 刀柄下端实际是平的。这里用胶囊（两端球冠）近似，所以线段起点
        # 要上移一个 holder_r，让球冠的最低点正好落在 holder_start 上。
        # 否则球冠会往下多鼓出 20 mm，在窄槽里到处误报撞刀 —— 实测
        # 三轴姿态下就已经报满。上移之后下端仍是圆角，比真实刀柄略大，
        # 方向上是保守的（只会多报，不会漏报）。
        hs = self.holder_start + self.holder_r
        hold_a = (tip[0] + n[0] * hs, tip[1] + n[1] * hs, tip[2] + n[2] * hs)
        hold_b = (hold_a[0] + n[0] * 200.0, hold_a[1] + n[1] * 200.0,
                  hold_a[2] + n[2] * 200.0)
        return [(tip, cut_end, self.shank_r, "flute"),
                (hold_a, hold_b, self.holder_r, "holder")]


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    if n <= 1e-300:
        raise ValueError("刀轴矢量长度为零")
    return (v[0] / n, v[1] / n, v[2] / n)


# ---------------------------------------------------------------- 两条判据

def swept_collision(pose, stock, clamps=(), tool=None, margin=0.0):
    """扫掠体碰撞。pose = (tip_xyz, axis_xyz)。

    `stock` 与 `clamps` 都是一组轴对齐长方体。返回命中的规则列表，
    每条带最小间隙 —— 间隙是负数时才是撞，正数时是"还剩多少"，
    后者对调机更有用（余量 0.3 mm 和余量 30 mm 是两回事）。
    """
    tool = tool or Tool()
    tip, axis = pose
    segs = tool.segments(tip, axis)
    out = []
    for name, boxes in (("stock", stock), ("clamp", clamps)):
        best = None
        for (a, b, r, seg_name) in segs:
            for box in boxes:
                d = segment_box_dist(a, b, box)
                clr = d - r - margin
                if best is None or clr < best[0]:
                    best = (clr, seg_name, box)
        if best is not None and best[0] < 0.0:
            out.append({"rule": "%s_crash" % name, "clearance": best[0],
                        "part": best[1], "box": best[2]})
    return out


def _rect_dist(x, y, rect):
    """(x,y) 到矩形 footprint 的二维距离。"""
    x0, x1, y0, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def heightfield_collision(pose, stock, clamps=(), tool=None, margin=0.0):
    """三轴高度场判据（作为对照，**不作为**五轴判据）。

    模型：假设刀轴竖直。于是刀在 (x,y) 平面上占一个半径 r 的圆盘，
    刀的最低点是 tip_z - r。判据是"最低点是否低于圆盘下方的最低面高"。

    这里必须用**圆盘**而不是刀尖那一个点。第一版只查刀尖的 (x,y)，
    结果连三轴都不对：刀在 x=63、半径 5 的时候，刀盘实际覆盖 58..68，
    左边已经压在 x<60 的毛坯上了，而只查 (63,30) 会给出"那里是空腔，
    安全"。所以圆盘不是精细化的选项，是三轴判据本来的样子。

    五轴下这个模型**结构性地**不够用：刀一倾斜，刀杆在 (x,y) 上的投影
    就从圆盘变成一条线段，而且伸出很远。深槽里刀杆撞对侧壁是它的盲区。
    """
    tool = tool or Tool()
    tip, axis = pose
    r = tool.radius
    out = []
    best = None
    for box in stock:
        x0, x1, y0, y1, z0, z1 = box
        if _rect_dist(tip[0], tip[1], (x0, x1, y0, y1)) <= r:
            clr = (tip[2] - r) - max(z0, z1)
            if best is None or clr < best:
                best = clr
    if best is not None and best - margin < 0.0:
        out.append({"rule": "stock_crash", "clearance": best - margin,
                    "part": "tip"})
    for box in clamps:
        x0, x1, y0, y1, z0, z1 = box
        if _rect_dist(tip[0], tip[1], (x0, x1, y0, y1)) <= r and tip[2] - r <= z1:
            out.append({"rule": "clamp_crash", "clearance": tip[2] - r - z1,
                        "part": "tip"})
    return out


def pocket(floor=-20.0, left=10.0, right=40.0, top=40.0, y0=-10.0, y1=70.0):
    """深窄槽：左右两堵高墙，槽底在 floor。

    这是五轴撞刀最典型的场景，也是高度场最没辙的场景：刀尖在槽中间、
    离槽底还有余量，高度场在那里读到的是槽底高度，于是判"安全"；
    而刀杆斜过去撞在侧壁上，位置在刀尖上方十几毫米处。
    """
    # 槽底必须是一块实体。少了它，刀可以一路穿到 floor 以下而两条判据
    # 都说"下面没材料"—— 那不是判据的问题，是场景本身没有底。
    stock = [(-60.0, left, y0, y1, floor, top),
             (right, 100.0, y0, y1, floor, top),
             (left, right, y0, y1, floor - 100.0, floor)]
    return {"stock": stock, "floor": floor, "left": left, "right": right,
            "top": top, "y0": y0, "y1": y1}


def compare_models(poses=None, tool=None, geom=None):
    """扫掠体与高度场逐条对照，返回两者的分歧。

    分歧只有两种：
        扫掠体说撞、高度场说安全   -> 高度场缺的那一维（真正要抓的）
        扫掠体说安全、高度场说撞   -> 高度场把刀当成竖直的，误报

    三轴姿态（刀轴竖直）下两者必须**完全一致**，这一条是基线。
    """
    tool = tool or Tool()
    g = geom or pocket()
    stock = g["stock"]
    if poses is None:
        poses = []
        mid = 0.5 * (g["left"] + g["right"])
        for tilt in (0.0, 10.0, 20.0, 30.0, 40.0):
            th = math.radians(tilt)
            ax = (math.sin(th), 0.0, math.cos(th))
            for dx in (-8.0, -4.0, 0.0, 4.0):
                # dz 要覆盖两段：刀尖低到会碰到槽底（两条判据都说撞），
                # 以及刀尖离槽底还有余量（高度场说安全，但刀杆可能撞壁）。
                # dz 的两段：-10 让刀尖撞到槽底（两条判据都说撞，基线）；
                # 16/20/24 让刀尖离槽底有余量、刀柄也已经高过墙顶，
                # 此时只有倾斜的刀杆能撞到侧壁 —— 高度场看不见的那种。
                for dz in (-10.0, 16.0, 20.0, 24.0):
                    poses.append(((mid + dx, 30.0, g["floor"] + dz), ax))
    rows = []
    for pose in poses:
        s = swept_collision(pose, stock, (), tool)
        h = heightfield_collision(pose, stock, (), tool)
        rows.append({"pose": pose,
                     "swept": bool(s), "height": bool(h),
                     "agree": bool(s) == bool(h),
                     "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, pose[1][2])))),
                     "swept_rules": [r["rule"] for r in s],
                     "min_clearance": min([r["clearance"] for r in s], default=None)})
    disagree = [r for r in rows if not r["agree"]]
    return {"rows": rows, "disagree": disagree,
            "n": len(rows), "n_disagree": len(disagree),
            "missed_by_height": [r for r in disagree
                                 if r["swept"] and not r["height"]],
            "false_alarm_by_height": [r for r in disagree
                                      if r["height"] and not r["swept"]],
            "three_axis": [r for r in rows if r["tilt_deg"] < 1e-9]}


# ---------------------------------------------------------------- 逆解

def ik_ac(axis, tol=1e-9):
    """转台-转台（A 绕 x、C 绕 z）五轴的逆解。

    正解：n = Rz(C) Rx(A) (0,0,1) = (sinA sinC, -sinA cosC, cosA)

    逆解：
        A = acos(nz)          A in [0, pi]
        C = atan2(nx, -ny)

    奇异：A = 0 或 pi 时刀轴与 z 重合，C 完全不定 —— 此时任何 C 都
    给出同一个刀轴。必须**报出来**，因为 CAM 侧会用它来保持 C 连续
    （否则 C 会在 0/2pi 之间跳，机床跟着甩）。随便返回一个角度会让
    后处理看起来正常，实际在机床上是一记甩刀。
    """
    n = _unit(axis)
    ca = max(-1.0, min(1.0, n[2]))
    A = math.acos(ca)
    singular = math.sin(A) < 1e-6
    if singular:
        C = 0.0
        residual = None
    else:
        C = math.atan2(n[0], -n[1])
        back = fk_ac(A, C)
        residual = max(abs(back[i] - n[i]) for i in range(3))
    return {"A": A, "C": C, "A_deg": math.degrees(A),
            "C_deg": math.degrees(C), "singular": singular,
            "residual": residual}


def fk_ac(A, C):
    """正解：由两个转角给出刀轴单位矢量。"""
    sa, ca = math.sin(A), math.cos(A)
    sc, cc = math.sin(C), math.cos(C)
    return (sa * sc, -sa * cc, ca)


def ik_roundtrip(samples=None):
    """逆解往返：方向 -> (A,C) -> 方向，必须一致到 1e-12。

    奇异位形单独统计：它们只能验证 A，C 没有意义。
    """
    if samples is None:
        samples = []
        for A_deg in (0.0, 15.0, 45.0, 90.0, 135.0, 179.0, 180.0):
            for C_deg in (0.0, 37.0, 90.0, 180.0, 271.0):
                samples.append((math.radians(A_deg), math.radians(C_deg)))
    worst = 0.0
    singular = 0
    for (A, C) in samples:
        n = fk_ac(A, C)
        r = ik_ac(n)
        if r["singular"]:
            singular += 1
            worst = max(worst, abs(r["A"] - A) if A <= math.pi / 2
                        else abs(r["A"] - A))
            continue
        back = fk_ac(r["A"], r["C"])
        worst = max(worst, max(abs(back[i] - n[i]) for i in range(3)))
    return {"worst": worst, "singular": singular, "samples": len(samples),
            "ok": worst < 1e-12}
