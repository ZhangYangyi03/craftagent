# -*- coding: utf-8 -*-
"""真正的等距曲线偏置：直线偏移 + 圆弧连接 + 凹角求交 + 自交环删除。

为什么要有这个文件
------------------
`cam.offset_contour` 做的是**逐点法向偏移**：每个顶点沿前后两段平均法向挪 dist。
它便宜、不会失败，但有两处结构性缺陷，而且两处都"错得看不出来"：

  凸角被拉长      miter 系数超过上限就被夹住，于是那一角留一条缝。缝在报告里
                  叫 `clamped`，但缝本身不是废品——它是"这一角残留一小块料"
                  的种子，而残留只有量出来才知道。
  凹角会自交      偏得比当地曲率半径还多时，偏出来的线会折回来，轮廓多出一小段
                  回头路。程序每一行都合法，门也放行，件是废的。

这个文件补的就是这两处。

一个必须先说清的区别：斜接不是等距曲线
--------------------------------------
把两条偏移线延长到交点（miter，也就是 `cam.offset_contour` 在做的事），
交点**到原轮廓的距离不是 dist**。矩形向外偏 5，斜接顶点在 (-5,-5)，
它到轮廓边界的距离是 7.07 而不是 5。所以斜接是 CAM 里的一个约定
（留给控制器补角），不是等距曲线。等距曲线在凸角处**必须是圆弧**。

默认 `join='round'`，就是为了让返回值真的是等距曲线。`join='miter'`
保留下来是因为 `cam` 的刀路要它（斜接出来的程序更短，且 G41 由控制器补角），
但那时 `max_dist_err` 会如实变大，报告里看得出来。

验收对着什么
------------
三条，都不是它自己：

  1. 矩形向外偏 r：面积等于 Steiner 公式 A + P*r + pi*r^2。凸集的数学事实，
     与本文件的实现无关。（只有圆角连接才满足；斜接会偏大，测试里分开断言。）
  2. 每个偏移顶点到**原轮廓边界**的距离等于 |dist|（容差 1e-9）。这条是等距
     曲线的定义本身。
  3. 面积对着**网格采样**（`area_grid`）比。网格是另一套完全不同的算法：
     它不知道什么是偏移线、什么是圆弧、什么是自交裁剪，只知道点到边的距离。
     第 3 条是唯一能抓住"方向对、量错了"的检查。

关于拓扑变化：不假装能表示
--------------------------
窄槽被偏置填平（向外）或腐蚀断成两截（向内）时，结果**不是一条轮廓**。
槽宽 4、向内偏 3，腐蚀后左右两块互不相连 —— 单条闭合折线表示不了这件事。
本文件不硬凑一条：它删环之后会检查自己（`ok` / `suspect` / `reason`），
删过环就置 `suspect=True`，理由写 `loop_removed`，调用方必须拿 `verify()`
的网格数去核对，核不上就拒绝用这条轮廓。

没有做的
--------
不做变半径偏置（dist 是常数）；不做开放折线的偏置（只接受闭合轮廓）；
不做 square join / 变半径圆角。这三样都是别的算法，不是这一个的开关。
"""
from __future__ import annotations

import math

EPS = 1e-12


# ---------------------------------------------------------------- 基础几何

def signed_area(pts):
    """有符号面积。>0 逆时针。绕向由它定，不靠假设。"""
    s = 0.0
    n = len(pts)
    for k in range(n):
        x1, y1 = pts[k]
        x2, y2 = pts[(k + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def area(pts):
    return abs(signed_area(pts))


def perimeter(pts):
    n = len(pts)
    return sum(math.hypot(pts[(k + 1) % n][0] - pts[k][0],
                          pts[(k + 1) % n][1] - pts[k][1]) for k in range(n))


def clean(pts, eps=1e-9):
    """去掉重复点与共线点。

    共线点不删的后果很具体：两条方向几乎相同的边求交时给出一个远处的交点，
    偏移轮廓会突然甩出一个尖刺，而尖刺上每一点到原轮廓的距离仍然等于 dist
    —— 距离检查抓不住它。所以宁可先删。
    """
    out = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append((float(p[0]), float(p[1])))
    while len(out) > 1 and math.hypot(out[0][0] - out[-1][0],
                                      out[0][1] - out[-1][1]) <= eps:
        out.pop()
    if len(out) < 3:
        return out
    keep = []
    n = len(out)
    for k in range(n):
        a, b, c = out[(k - 1) % n], out[k], out[(k + 1) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) > 1e-12:
            keep.append(b)
    return keep if len(keep) >= 3 else out


def _unit(dx, dy):
    L = math.hypot(dx, dy)
    if L <= EPS:
        return None
    return dx / L, dy / L


def point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= EPS:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def distance_to_boundary(p, poly):
    """点到闭合折线**边界**的最短距离。不是到区域的距离。"""
    n = len(poly)
    return min(point_segment_distance(p, poly[k], poly[(k + 1) % n])
               for k in range(n))


def point_in_polygon(p, poly):
    inside = False
    n = len(poly)
    x, y = p
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xx:
                inside = not inside
    return inside


def signed_distance(p, poly):
    """到边界的距离，内部取负。偏置区域就是 {p : signed_distance(p) <= dist}。"""
    d = distance_to_boundary(p, poly)
    return -d if point_in_polygon(p, poly) else d


# ---------------------------------------------------------------- 自交

def _seg_cross(p1, p2, p3, p4):
    """两段严格内部相交才返回交点。

    端点相接不算 —— 相邻段永远共用一个端点，把它们判成相交会让删环永远
    删不完（每删一次又生成一个新的相邻交点）。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-15:
        return None
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / den
    u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / den
    if 1e-12 < t < 1.0 - 1e-12 and 1e-12 < u < 1.0 - 1e-12:
        return (x1 + t * d1x, y1 + t * d1y)
    return None


def first_self_intersection(pts):
    """返回 (i, j, X)：第 i 段与第 j 段在 X 处相交。没有就 None。"""
    n = len(pts)
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            X = _seg_cross(a1, a2, pts[j], pts[(j + 1) % n])
            if X is not None:
                return i, j, X
    return None


def is_simple(pts):
    return first_self_intersection(pts) is None


def remove_loops(pts, max_loops=64):
    """反复删掉自交产生的小环，直到没有自交。

    为什么删**面积小的**那个：自交一定把轮廓分成两块，一块是真正要的偏置
    边界，另一块是折回来的赘余，赘余总更小。但这不是定理，所以在返回值里
    把删了几次报出来（`loops_removed`），让读的人知道这条轮廓被改过。
    """
    removed = 0
    for _ in range(max_loops):
        hit = first_self_intersection(pts)
        if hit is None:
            break
        i, j, X = hit
        A = [X] + pts[i + 1:j + 1]
        B = [X] + pts[j + 1:] + pts[:i + 1]
        if len(A) < 3 or len(B) < 3:
            break
        pts = A if area(A) >= area(B) else B
        removed += 1
    return pts, removed


# ---------------------------------------------------------------- 偏置

def offset_polygon(pts, dist, join="round", miter_limit=4.0, arc_tol=0.005):
    """把闭合轮廓沿法向精确偏 dist（正数向外，负数向内）。

    join='round'（默认）凸角补圆弧 —— 这才是等距曲线。
    join='miter'  凸角走斜接（cam 的刀路要它），交点距离会大于 |dist|。

    返回 dict：
        pts            偏移后的闭合轮廓
        kind           'exact_offset' / 'degenerate'
        ok             距离与简单性都过（不含拓扑检查）
        suspect        True 表示删过环，拓扑可能变了，必须用 verify() 核对
        reason         suspect 的理由；空串表示没有
        arcs/miter/trimmed/loops_removed   各类关节用了几个
        max_dist_err   每个顶点到原轮廓距离与 |dist| 的最大偏差
    """
    src = clean(pts)
    if len(src) < 3 or abs(dist) < EPS:
        return {"pts": list(src), "kind": "degenerate", "ok": False,
                "suspect": False, "reason": "degenerate_input", "arcs": 0,
                "miter": 0, "trimmed": 0, "loops_removed": 0, "simple": True,
                "max_dist_err": 0.0}

    n = len(src)
    s = 1.0 if signed_area(src) > 0 else -1.0
    # 逆时针时外法向在右手边：(dy, -dx)。顺时针取反。绕向由面积定，不靠假设。
    dirs = []
    for k in range(n):
        a, b = src[k], src[(k + 1) % n]
        dirs.append(_unit(b[0] - a[0], b[1] - a[1]))
    normals = [(s * d[1], -s * d[0]) for d in dirs]

    out = []
    arcs = miters = trimmed = clamped = 0
    for k in range(n):
        u_prev, u_next = dirs[(k - 1) % n], dirs[k]
        n_prev, n_next = normals[(k - 1) % n], normals[k]
        A = (src[k][0] + dist * n_prev[0], src[k][1] + dist * n_prev[1])
        B = (src[k][0] + dist * n_next[0], src[k][1] + dist * n_next[1])

        cross = u_prev[0] * u_next[1] - u_prev[1] * u_next[0]
        turn = cross * s                      # >0 逆时针意义下的凸角
        dot = max(-1.0, min(1.0, u_prev[0] * u_next[0] + u_prev[1] * u_next[1]))

        if abs(cross) < 1e-12:
            out.append(A)                     # 直行
            continue

        # 偏移线张开缺口 <-> turn 与 dist 同号；交叉 <-> 异号。
        if turn * dist > 1e-12:
            if join == "miter":
                half_cos = math.sqrt(max(1e-12, 0.5 * (1.0 + dot)))
                scale = 1.0 / half_cos
                if scale <= miter_limit:
                    bis = _unit(n_prev[0] + n_next[0], n_prev[1] + n_next[1])
                    out.append((src[k][0] + dist * bis[0] * scale,
                                src[k][1] + dist * bis[1] * scale))
                    miters += 1
                    continue
                clamped += 1
            # 圆角连接：这一角必须补圆弧，不能夹住留缝。
            out.extend(_arc(src[k], A, B, abs(dist), arc_tol))
            arcs += 1
        else:
            # 交叉：取两条偏移线的交点，把多出来的那一段裁掉。
            den = u_prev[0] * n_next[0] + u_prev[1] * n_next[1]
            if abs(den) < 1e-9:
                out.append(((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5))
                continue
            t = ((n_next[0] - n_prev[0]) * n_next[0] +
                 (n_next[1] - n_prev[1]) * n_next[1]) / den
            out.append((src[k][0] + dist * (n_prev[0] + t * u_prev[0]),
                        src[k][1] + dist * (n_prev[1] + t * u_prev[1])))
            trimmed += 1

    out = clean(out)
    if len(out) < 3:
        return {"pts": list(src), "kind": "degenerate", "ok": False,
                "suspect": False, "reason": "collapsed", "arcs": arcs,
                "miter": miters, "trimmed": trimmed, "loops_removed": 0,
                "simple": True, "max_dist_err": 0.0}

    out, removed = remove_loops(out)
    if len(out) < 3:
        return {"pts": list(src), "kind": "degenerate", "ok": False,
                "suspect": True, "reason": "all_loops_removed", "arcs": arcs,
                "miter": miters, "trimmed": trimmed, "loops_removed": removed,
                "simple": True, "max_dist_err": 0.0}

    err = max(abs(distance_to_boundary(p, src) - abs(dist)) for p in out)
    simple = is_simple(out)
    # 圆弧是用弦代替的，弦高容差 arc_tol 就是这条曲线的精度上限：
    # 两段圆弧相交时，交点取在两条**弦**的交点上，所以偏差不超过 arc_tol。
    # 把这个上限写进判据，而不是写一个更严的数假装更准。
    ok = simple and (join == "miter" or err <= max(arc_tol * 1.2, 1e-9))
    if join == "miter":
        # 斜接顶点到轮廓的距离本来就大于 |dist|（矩形外偏 5 得到 7.07），
        # 它不是等距曲线，所以距离这一项不参与判定，只有简单性有意义。
        ok = simple
    return {"pts": out, "kind": "exact_offset", "ok": ok,
            "suspect": removed > 0, "reason": "loop_removed" if removed else "",
            "arcs": arcs, "miter": miters, "trimmed": trimmed,
            "loops_removed": removed, "simple": simple, "max_dist_err": err,
            "miter_clamped": clamped}


def _arc(center, A, B, radius, arc_tol):
    """绕 center 从 A 到 B 的圆弧采样，走短的那一边。

    采样密度由**弦高**定，不由固定段数定：半径大角度小时用 40 段是浪费，
    半径小角度大时用 4 段会切进材料。弦高容差 0.02 mm 是图纸的量级。
    """
    ax, ay = A[0] - center[0], A[1] - center[1]
    bx, by = B[0] - center[0], B[1] - center[1]
    a0 = math.atan2(ay, ax)
    sweep = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
    if abs(sweep) < 1e-12:
        return [A, B]
    if radius <= arc_tol:
        steps = 1
    else:
        step = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - arc_tol / radius)))
        steps = max(1, min(64, int(math.ceil(abs(sweep) / max(step, 1e-6)))))
    out = [A]
    for k in range(1, steps):
        a = a0 + sweep * k / steps
        out.append((center[0] + radius * math.cos(a),
                    center[1] + radius * math.sin(a)))
    out.append(B)
    return out


# ---------------------------------------------------------------- 独立参照

def area_grid(pts, dist, n=400, pad=None):
    """用网格数格点算偏置区域面积 —— 和偏置算法完全无关的另一套东西。

    区域定义只有一句：{p : signed_distance(p) <= dist}。它不知道什么是直线
    偏移、什么是圆弧、什么是自交裁剪，所以它能抓住"轮廓偏对了方向但错了量"
    这一类错。

    精度 O(1/n)：边界那一圈格点被整格计入或整格不计，误差约为
    (周长 * 格子边长)/2。n=400 时对 100 mm 量级的件是 0.1% 量级。
    """
    if pad is None:
        pad = abs(dist) * 1.5 + 1.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    cell = max(x1 - x0, y1 - y0) / n
    cnt = 0
    y = y0 + cell * 0.5
    while y < y1:
        x = x0 + cell * 0.5
        while x < x1:
            if signed_distance((x, y), pts) <= dist:
                cnt += 1
            x += cell
        y += cell
    return cnt * cell * cell


def verify(src, dist, n=300, tol=0.02):
    """拿网格核对一条偏置轮廓：面积对不对、有没有自交、距离是不是处处等于 |dist|。

    返回 dict(ok, area, area_ref, rel_err, simple, max_dist_err)。
    `ok` 为 False 时**不要用这条轮廓** —— 它可能少了一块（窄槽被腐蚀断）。
    """
    r = offset_polygon(src, dist)
    off = r["pts"]
    a = area(off) if len(off) >= 3 else 0.0
    ref = area_grid(src, dist, n=n)
    rel = abs(a - ref) / ref if ref > 0 else 1.0
    return {"ok": rel <= tol and r["ok"],
            "area": a, "area_ref": ref, "rel_err": rel,
            "simple": r["simple"], "max_dist_err": r["max_dist_err"],
            "loops_removed": r["loops_removed"], "points": len(off),
            "suspect": r["suspect"], "reason": r["reason"]}
