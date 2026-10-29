# -*- coding: utf-8 -*-
"""csg.py -- 纯标准库的三角网格布尔（BSP）：并 / 交 / 差。

为什么要有这个，而不是直接用 OCC
--------------------------------
OCC 的布尔是 B-rep 上的精确运算，那是另一条路（见 booleans.py），而且
需要 OCP —— 这台机器上没有，CI 上装在另外的解释器里。于是出现一个
很实际的问题：**本机跑不了的东西，本机就没法验**。

所以这一层用 BSP 在三角网格上做布尔。它不精确（结果是折线逼近），
但它能在任何装了 Python 的地方跑，于是可以在本机对着**解析体积**验：

    两个 10 立方体错开 5 叠在一起
        并  1000 + 1000 - 125 = 1875
        交  5^3 = 125
        差  1000 - 125 = 875
    板挖通孔
        欧拉示性数必须是 0（一个通孔），体积 = 板 - 孔

这些数字与实现无关，所以它们能证明布尔真的做了布尔，而不是"返回了
输入的一个子集"。

算法
----
BSP 树 + 多边形裁剪，与 csg.js 同一套：把 A 的多边形插进 B 的 BSP 树里
裁剪掉被 B 挡住的部分，反过来再裁一次，然后合并。共面多边形单独处理
（coplanarFront / coplanarBack），否则两个盒子贴面时会出现缝隙。

这一层不假装精确：所有坐标都是浮点，共面判定带容差，结果是三角网格。
需要精确 B-rep 的场合走 booleans.py。
"""
from __future__ import annotations

import math

__all__ = ["Polygon", "Node", "box", "prism", "union", "subtract",
           "intersect", "volume", "area", "topology", "check"]

_EPS = 1e-9


class Polygon:
    """带平面的多边形（这里只用三角形和四边形，都是凸的）。"""

    __slots__ = ("verts", "plane")

    def __init__(self, verts, plane=None):
        self.verts = [tuple(float(c) for c in v) for v in verts]
        self.plane = plane or _plane_of(self.verts)

    def clone(self):
        return Polygon(self.verts, self.plane)

    def flipped(self):
        n, w = self.plane
        return Polygon(list(reversed(self.verts)), ((-n[0], -n[1], -n[2]), -w))


def _plane_of(verts):
    """三点定平面，返回 (单位法向, 常数项)。"""
    for i in range(1, len(verts) - 1):
        a, b, c = verts[0], verts[i], verts[i + 1]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        L = math.sqrt(nx * nx + ny * ny + nz * nz)
        if L > 1e-14:
            n = (nx / L, ny / L, nz / L)
            return (n, n[0] * a[0] + n[1] * a[1] + n[2] * a[2])
    raise ValueError("退化多边形：三点共线")


def _split(poly, plane, coplanar_front, coplanar_back, front, back):
    """按平面切一个多边形，分到前后两侧。"""
    n, w = plane
    COPLANAR, FRONT, BACK, SPANNING = 0, 1, 2, 3
    ptype = COPLANAR
    types = []
    for v in poly.verts:
        t = n[0] * v[0] + n[1] * v[1] + n[2] * v[2] - w
        ty = BACK if t < -_EPS else (FRONT if t > _EPS else COPLANAR)
        ptype |= ty
        types.append(ty)
    if ptype == COPLANAR:
        # 共面：按法向同向/反向分到两侧，否则贴面处会出现缝隙
        dot = n[0] * poly.plane[0][0] + n[1] * poly.plane[0][1] + n[2] * poly.plane[0][2]
        (coplanar_front if dot > 0 else coplanar_back).append(poly)
    elif ptype == FRONT:
        front.append(poly)
    elif ptype == BACK:
        back.append(poly)
    else:
        f, b = [], []
        k = len(poly.verts)
        for i in range(k):
            j = (i + 1) % k
            ti, tj = types[i], types[j]
            vi, vj = poly.verts[i], poly.verts[j]
            if ti != BACK:
                f.append(vi)
            if ti != FRONT:
                b.append(vi)
            if (ti | tj) == SPANNING:
                di = n[0] * vi[0] + n[1] * vi[1] + n[2] * vi[2] - w
                dj = n[0] * vj[0] + n[1] * vj[1] + n[2] * vj[2] - w
                t = di / (di - dj)
                v = (vi[0] + t * (vj[0] - vi[0]),
                     vi[1] + t * (vj[1] - vi[1]),
                     vi[2] + t * (vj[2] - vi[2]))
                f.append(v)
                b.append(v)
        if len(f) >= 3:
            front.append(Polygon(f, poly.plane))
        if len(b) >= 3:
            back.append(Polygon(b, poly.plane))


class Node:
    """BSP 节点。"""

    __slots__ = ("plane", "front", "back", "polygons")

    def __init__(self, polygons=None):
        self.plane = None
        self.front = None
        self.back = None
        self.polygons = []
        if polygons:
            self.build(polygons)

    def invert(self):
        for p in self.polygons:
            p.verts.reverse()
            n, w = p.plane
            p.plane = ((-n[0], -n[1], -n[2]), -w)
        if self.plane:
            n, w = self.plane
            # 翻平面必须**同时**翻法向和常数项。只翻 w 的话平面还是同一个
            # 平面，但前后两侧的定义反了 —— 布尔照样返回一个闭合网格，
            # 只是交集变成空的、差集少一块。实测：只翻 w 时交集体积为 0。
            self.plane = ((-n[0], -n[1], -n[2]), -w)
        if self.front:
            self.front.invert()
        if self.back:
            self.back.invert()
        self.front, self.back = self.back, self.front

    def clip_polygons(self, polygons):
        if not self.plane:
            return list(polygons)
        front, back = [], []
        for p in polygons:
            _split(p, self.plane, front, back, front, back)
        if self.front:
            front = self.front.clip_polygons(front)
        if self.back:
            back = self.back.clip_polygons(back)
        else:
            back = []
        return front + back

    def clip_to(self, other):
        self.polygons = other.clip_polygons(self.polygons)
        if self.front:
            self.front.clip_to(other)
        if self.back:
            self.back.clip_to(other)

    def all_polygons(self):
        out = list(self.polygons)
        if self.front:
            out.extend(self.front.all_polygons())
        if self.back:
            out.extend(self.back.all_polygons())
        return out

    def build(self, polygons):
        if not polygons:
            return
        if not self.plane:
            self.plane = polygons[0].plane
        front, back = [], []
        for p in polygons:
            _split(p, self.plane, self.polygons, self.polygons, front, back)
        if front:
            if not self.front:
                self.front = Node()
            self.front.build(front)
        if back:
            if not self.back:
                self.back = Node()
            self.back.build(back)


# ---------------------------------------------------------------- 图元

def box(x0, x1, y0, y1, z0, z1):
    """轴对齐长方体 -> 12 个三角形（外法向）。"""
    p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    out = []
    for q in quads:
        a, b, c, d = (p[i] for i in q)
        out.append(Polygon([a, b, c]))
        out.append(Polygon([a, c, d]))
    return out


def prism(cx, cy, r, z0, z1, n=24):
    """正 n 边棱柱，轴平行 z。用来近似圆柱孔。"""
    ring = [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    out = []
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        out.append(Polygon([(a[0], a[1], z0), (b[0], b[1], z0),
                            (b[0], b[1], z1), (a[0], a[1], z1)]))
    for (z, flip) in ((z0, True), (z1, False)):
        for i in range(n):
            a = ring[i]
            b = ring[(i + 1) % n]
            tri = [(cx, cy, z), (a[0], a[1], z), (b[0], b[1], z)]
            out.append(Polygon(list(reversed(tri)) if flip else tri))
    return out


# ---------------------------------------------------------------- 布尔

def _to_node(polys):
    return Node([p.clone() for p in polys])


def union(a, b):
    A, B = _to_node(a), _to_node(b)
    A.clip_to(B)
    B.clip_to(A)
    B.invert()
    B.clip_to(A)
    B.invert()
    A.build(B.all_polygons())
    return A.all_polygons()


def subtract(a, b):
    A, B = _to_node(a), _to_node(b)
    A.invert()
    A.clip_to(B)
    B.clip_to(A)
    B.invert()
    B.clip_to(A)
    B.invert()
    A.build(B.all_polygons())
    A.invert()
    return A.all_polygons()


def intersect(a, b):
    A, B = _to_node(a), _to_node(b)
    A.invert()
    B.clip_to(A)
    B.invert()
    A.clip_to(B)
    B.clip_to(A)
    A.build(B.all_polygons())
    A.invert()
    return A.all_polygons()


# ---------------------------------------------------------------- 验收

def volume(polys):
    """散度定理：V = (1/6) sum (v0 x v1) . v2。

    对闭合三角网格是精确的，所以它同时也在检查网格闭不闭合 ——
    少一个面，体积立刻不对。
    """
    v = 0.0
    for p in polys:
        vs = p.verts
        for i in range(1, len(vs) - 1):
            a, b, c = vs[0], vs[i], vs[i + 1]
            v += (a[0] * (b[1] * c[2] - b[2] * c[1])
                  - a[1] * (b[0] * c[2] - b[2] * c[0])
                  + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return v / 6.0


def area(polys):
    a = 0.0
    for p in polys:
        vs = p.verts
        for i in range(1, len(vs) - 1):
            ux, uy, uz = vs[i][0] - vs[0][0], vs[i][1] - vs[0][1], vs[i][2] - vs[0][2]
            vx, vy, vz = vs[i + 1][0] - vs[0][0], vs[i + 1][1] - vs[0][1], vs[i + 1][2] - vs[0][2]
            cx = uy * vz - uz * vy
            cy = uz * vx - ux * vz
            cz = ux * vy - uy * vx
            a += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    return a


def _key(v, q=1e-7):
    return (round(v[0] / q), round(v[1] / q), round(v[2] / q))


def topology(polys):
    """V / E / F 与欧拉示性数。

    闭合流形网格里每条边恰好被两个面共用。`edges_bad` 不为零说明结果
    网格**不是流形** —— 实际观察到的是 T 型接点：BSP 切出来的新顶点
    落在对面某个面的棱中间，于是那侧的棱被切成两段、这侧还是一段，
    计数变成 1 和 3。

    T 型接点不影响体积（散度定理仍成立，因为曲面在几何上还是闭合的），
    但下游拿去做网格生成就会在接点处漏掉材料。所以这里单独数它，
    而不是只报一个体积就完事 —— 体积对、网格不能用，是这一层真实的
    边界，不是可以省略的细节。
    """
    verts = {}
    edges = {}
    for p in polys:
        vs = p.verts
        k = len(vs)
        ids = []
        for v in vs:
            key = _key(v)
            if key not in verts:
                verts[key] = len(verts)
            ids.append(verts[key])
        for i in range(k):
            a, b = ids[i], ids[(i + 1) % k]
            e = (a, b) if a < b else (b, a)
            edges[e] = edges.get(e, 0) + 1
    V, E, F = len(verts), len(edges), len(polys)
    bad = sum(1 for c in edges.values() if c != 2)
    return {"V": V, "E": E, "F": F, "euler": V - E + F,
            "edges_bad": bad, "manifold": bad == 0}


def check():
    """三组已知答案的布尔，外加一个带通孔的件。

    判据分两层，因为它们回答的是不同的问题：

        体积     布尔到底做没做布尔（对解析体积）
        裂边     结果网格能不能给下游用（对流形性）

    轴对齐盒体三组都是 0 误差、0 裂边。带孔件体积精确到 1e-16（对
    棱柱的解析体积），但裂边随棱数线性增长（n=6 是 38 条，n=48 是 320
    条）—— BSP 在曲面逼近上会留 T 型接点。这不是"再调调就能好"的
    参数问题，是这条路线的边界；要无裂边的结果走 OCC（booleans.py）。
    """
    A = box(0, 10, 0, 10, 0, 10)
    B = box(5, 15, 5, 15, 5, 15)
    rows = [
        {"op": "union", "want": 1875.0, "got": volume(union(A, B))},
        {"op": "intersect", "want": 125.0, "got": volume(intersect(A, B))},
        {"op": "subtract", "want": 875.0, "got": volume(subtract(A, B))},
    ]
    for r in rows:
        r["rel_err"] = abs(r["got"] - r["want"]) / abs(r["want"])
        r["ok"] = r["rel_err"] < 1e-9
    plate = box(0, 40, 0, 20, 0, 6)
    drilled = {}
    for n in (12, 48):
        d = subtract(plate, prism(20, 10, 5, -1, 7, n=n))
        want = 40 * 20 * 6 - 0.5 * n * 25 * math.sin(2 * math.pi / n) * 6
        t = topology(d)
        drilled[n] = {"want": want, "got": volume(d),
                      "rel_err": abs(volume(d) - want) / want,
                      "euler": t["euler"], "edges_bad": t["edges_bad"],
                      "manifold": t["manifold"]}
    return {"rows": rows, "ok": all(r["ok"] for r in rows),
            "drilled": drilled,
            "drilled_volume_ok": all(v["rel_err"] < 1e-9
                                     for v in drilled.values()),
            "drilled_manifold": all(v["manifold"] for v in drilled.values())}
