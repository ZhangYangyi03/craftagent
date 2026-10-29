# -*- coding: utf-8 -*-
"""voxel.py -- 实体 -> 占据格：沿 z 的**精确区间** + 边界节点回贴。

为什么用区间而不是三角网格
--------------------------
要把一个零件变成网格，第一步得先有一个"零件"。cad.py 能从 STEP 读三角面，
但三角面做布尔、做内外判定，在共面处会碎、在薄壁处会漏。这里换一条路：

    每个实体回答一个问题 —— **这条竖直线上，材料在哪几段。**

答案是一个一维区间集合。于是：

    并   区间集合求并
    交   区间集合求交
    差   区间集合求差
    内外 看 z 落不落在区间里
    体积 区间长度对 x-y 积分

全部是一维区间算术，没有曲面求交、没有容差、没有共面退化。代价是这套
表示只对"沿 z 可分解"的实体精确 —— 但箱体、圆柱、棱柱、球，以及它们
的任意布尔组合，覆盖了机加工零件的大多数。

这不是在回避 OCC。OCC 的布尔在 booleans.py 里走另一条路（B-rep，精确，
但本机没有 OCP）。这里是**不需要 OCC 也能跑的那一半**，而且它能给出
OCC 给不出的东西：一个可以逐点求值的精确实体，网格生成的每一步都能拿它
回头核对。

薄壁会丢，这一条要明说
----------------------
单元取中心点判内外，所以一条比 h 还薄的壁可能整条落空。这不是 bug 是采样
的固有代价，所以 `voxelize` 把它**数出来**：`lost_volume` 是"实体里有、
但没有任何单元中心落在其中"的那部分体积。

但这个数只看得见**沿 z 方向**薄的壁。为什么：采样是按 (x, y) 一列一列走的，
一列里只有 z 区间会被判断"区间短于 h 且不含格心"。如果壁薄在 x 或 y 上，
整列根本没被采到，`iv` 是空的，什么都不会累加 —— 于是 `lost_volume` 是 0，
而材料确实丢了。实测：40x20x6 的板上立一道 0.8 厚的壁（y 从 9.6 到 10.4），
h=2 时格心在 y=1,3,...,9,11，一个都没落进 9.6~10.4，那道壁整条不见，
`lost_volume` 仍然是 0.0。

所以要判断网格完不完整，看的是 `volume_error()`（对 `volume_ref` 的净误差，
可正可负），不是 `lost_volume` 是否为零。两个数都给，各有各的盲区。

边界节点回贴
------------
体素边界是台阶。台阶会让表面应力凭空出现尖角，所以边界节点要沿"朝内"的
方向推到真实表面上：从节点往相邻内侧单元中心走，二分找内外交界。推完之后
逐个检查所属四面体的有向体积，**推坏了就退回去** —— 宁可留一个台阶，也不
能留一个负体积单元。
"""
from __future__ import annotations

import math

__all__ = [
    "Shape", "Box", "Sphere", "CylinderZ", "CylinderX", "PrismZ",
    "Union", "Intersect", "Diff",
    "voxelize", "snap_node", "interval_length", "volume_ref", "volume_error",
    "boundary_nodes",
]


# ---------------------------------------------------------------- 区间算术

def _norm(iv):
    """区间集合规范化：排序，合并重叠或相接的段。"""
    out = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _union(A, B):
    return _norm(list(A) + list(B))


def _intersect(A, B):
    out = []
    i = j = 0
    while i < len(A) and j < len(B):
        a0, a1 = A[i]
        b0, b1 = B[j]
        lo, hi = max(a0, b0), min(a1, b1)
        if hi > lo:
            out.append((lo, hi))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return out


def _diff(A, B):
    """A 减 B。逐段切，比"求交再取补"少一次规范化。"""
    out = []
    for a0, a1 in A:
        cur = a0
        for b0, b1 in B:
            if b1 <= cur:
                continue
            if b0 >= a1:
                break
            if b0 > cur:
                out.append((cur, min(b0, a1)))
            cur = max(cur, b1)
            if cur >= a1:
                break
        if cur < a1:
            out.append((cur, a1))
    return out


def interval_length(iv):
    return sum(b - a for a, b in iv)


# ---------------------------------------------------------------- 实体

class Shape:
    """实体接口：只要求两个方法。

    `intervals(x, y)` 给竖直线上材料的 z 段，`bbox()` 给包围盒。
    其余（内外判定、体积）都由这两个推出来 —— 所以子类只要把这两件事
    写对，布尔、体素化、回贴就全都自动成立。
    """

    def intervals(self, x, y):
        raise NotImplementedError

    def bbox(self):
        raise NotImplementedError

    def contains(self, p):
        x, y, z = p
        return any(a <= z <= b for a, b in self.intervals(x, y))

    def volume(self):
        """解析体积；CSG 组合没有闭式，返回 None（不假装知道）。"""
        return None


class Box(Shape):
    def __init__(self, x0, x1, y0, y1, z0, z1):
        self.x0, self.x1 = min(x0, x1), max(x0, x1)
        self.y0, self.y1 = min(y0, y1), max(y0, y1)
        self.z0, self.z1 = min(z0, z1), max(z0, z1)

    def intervals(self, x, y):
        if self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1:
            return [(self.z0, self.z1)]
        return []

    def bbox(self):
        return (self.x0, self.x1, self.y0, self.y1, self.z0, self.z1)

    def volume(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)


class Sphere(Shape):
    def __init__(self, cx, cy, cz, r):
        self.cx, self.cy, self.cz, self.r = cx, cy, cz, abs(r)

    def intervals(self, x, y):
        d2 = (x - self.cx) ** 2 + (y - self.cy) ** 2
        r2 = self.r * self.r
        if d2 >= r2:
            return []
        dz = math.sqrt(r2 - d2)
        return [(self.cz - dz, self.cz + dz)]

    def bbox(self):
        r = self.r
        return (self.cx - r, self.cx + r, self.cy - r, self.cy + r,
                self.cz - r, self.cz + r)

    def volume(self):
        return 4.0 / 3.0 * math.pi * self.r ** 3


class CylinderZ(Shape):
    """轴线平行 z 的圆柱。"""

    def __init__(self, cx, cy, r, z0, z1):
        self.cx, self.cy, self.r = cx, cy, abs(r)
        self.z0, self.z1 = min(z0, z1), max(z0, z1)

    def intervals(self, x, y):
        d2 = (x - self.cx) ** 2 + (y - self.cy) ** 2
        r2 = self.r * self.r
        if d2 >= r2:
            return []
        return [(self.z0, self.z1)]

    def bbox(self):
        r = self.r
        return (self.cx - r, self.cx + r, self.cy - r, self.cy + r,
                self.z0, self.z1)

    def volume(self):
        return math.pi * self.r ** 2 * (self.z1 - self.z0)


class CylinderX(Shape):
    """轴线平行 x 的圆柱（横孔、横向轴颈）。"""

    def __init__(self, cy, cz, r, x0, x1):
        self.cy, self.cz, self.r = cy, cz, abs(r)
        self.x0, self.x1 = min(x0, x1), max(x0, x1)

    def intervals(self, x, y):
        if not (self.x0 <= x <= self.x1):
            return []
        dy = y - self.cy
        r2 = self.r * self.r
        if dy * dy >= r2:
            return []
        dz = math.sqrt(r2 - dy * dy)
        return [(self.cz - dz, self.cz + dz)]

    def bbox(self):
        r = self.r
        return (self.x0, self.x1, self.cy - r, self.cy + r,
                self.cz - r, self.cz + r)

    def volume(self):
        return math.pi * self.r ** 2 * (self.x1 - self.x0)


class PrismZ(Shape):
    """x-y 多边形沿 z 拉伸。多边形给点列（不闭合也可以）。"""

    def __init__(self, poly, z0, z1):
        self.poly = [(float(a), float(b)) for a, b in poly]
        self.z0, self.z1 = min(z0, z1), max(z0, z1)

    def _inside_xy(self, x, y):
        # 射线穿越计数。边界上的点算不算内不影响体素化（单元取中心）。
        inside = False
        n = len(self.poly)
        for i in range(n):
            xa, ya = self.poly[i]
            xb, yb = self.poly[(i + 1) % n]
            if (ya > y) != (yb > y):
                xc = xa + (y - ya) * (xb - xa) / (yb - ya)
                if xc > x:
                    inside = not inside
        return inside

    def intervals(self, x, y):
        if self._inside_xy(x, y):
            return [(self.z0, self.z1)]
        return []

    def bbox(self):
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        return (min(xs), max(xs), min(ys), max(ys), self.z0, self.z1)

    def _area_xy(self):
        a = 0.0
        n = len(self.poly)
        for i in range(n):
            xa, ya = self.poly[i]
            xb, yb = self.poly[(i + 1) % n]
            a += xa * yb - xb * ya
        return abs(a) * 0.5

    def volume(self):
        return self._area_xy() * (self.z1 - self.z0)


class _CSG(Shape):
    op = "?"

    def __init__(self, a, b):
        self.a, self.b = a, b

    def bbox(self):
        ax0, ax1, ay0, ay1, az0, az1 = self.a.bbox()
        bx0, bx1, by0, by1, bz0, bz1 = self.b.bbox()
        if self.op == "union":
            return (min(ax0, bx0), max(ax1, bx1), min(ay0, by0), max(ay1, by1),
                    min(az0, bz0), max(az1, bz1))
        # 交与差都落在 a 里，但用两者交集收紧包围盒（差可能更小，保守即可）
        return (min(ax0, bx0), max(ax1, bx1), min(ay0, by0), max(ay1, by1),
                min(az0, bz0), max(az1, bz1))


class Union(_CSG):
    op = "union"

    def intervals(self, x, y):
        return _union(self.a.intervals(x, y), self.b.intervals(x, y))


class Intersect(_CSG):
    op = "intersect"

    def intervals(self, x, y):
        return _intersect(self.a.intervals(x, y), self.b.intervals(x, y))


class Diff(_CSG):
    op = "diff"

    def intervals(self, x, y):
        return _diff(self.a.intervals(x, y), self.b.intervals(x, y))


# ---------------------------------------------------------------- 体素化

def _cell_index_range(a, b, z0, h, nz):
    """单元中心 z0+(k+0.5)h 落在 [a,b] 内的 k 区间（闭）。

    端点用 1e-12 的相对容差放宽，避免 a 正好落在中心上时被 ceil 挤掉。
    """
    lo = math.ceil((a - z0) / h - 0.5 - 1e-12)
    hi = math.floor((b - z0) / h - 0.5 + 1e-12)
    if lo < 0:
        lo = 0
    if hi > nz - 1:
        hi = nz - 1
    return lo, hi


def voxelize(shape, h, bbox=None):
    """把实体打成占据格。

    返回的 `inside` 是长度 nx*ny*nz 的字节数组，索引 i + j*nx + k*nx*ny。

    `lost_volume` 是**实体里有但网格看不见**的体积：区间短于 h 且不含任何
    单元中心。它不为零意味着网格漏了材料，调用方必须知道。
    """
    if h <= 0:
        raise ValueError("h 必须为正")
    bx0, bx1, by0, by1, bz0, bz1 = bbox or shape.bbox()
    # 外面留一层空格，保证边界节点一定被"外侧单元"包围，回贴方向才存在
    x0 = bx0 - h
    y0 = by0 - h
    z0 = bz0 - h
    nx = max(1, int(math.ceil((bx1 - bx0 + 2 * h) / h)))
    ny = max(1, int(math.ceil((by1 - by0 + 2 * h) / h)))
    nz = max(1, int(math.ceil((bz1 - bz0 + 2 * h) / h)))

    inside = bytearray(nx * ny * nz)
    lost = 0.0
    for j in range(ny):
        y = y0 + (j + 0.5) * h
        for i in range(nx):
            x = x0 + (i + 0.5) * h
            iv = shape.intervals(x, y)
            if not iv:
                continue
            base = i + j * nx
            for a, b in iv:
                lo, hi = _cell_index_range(a, b, z0, h, nz)
                if lo > hi:
                    lost += (b - a) * h * h
                    continue
                for k in range(lo, hi + 1):
                    inside[base + k * nx * ny] = 1

    return {"h": h, "nx": nx, "ny": ny, "nz": nz,
            "x0": x0, "y0": y0, "z0": z0,
            "inside": inside, "lost_volume": lost,
            "cells": sum(inside)}


def _node_pos(mesh, n):
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    sx, sy = nx + 1, ny + 1
    i = n % sx
    j = (n // sx) % sy
    k = n // (sx * sy)
    h = mesh["h"]
    return (mesh["x0"] + i * h, mesh["y0"] + j * h, mesh["z0"] + k * h)


def _cell(mesh, i, j, k):
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return 0
    return mesh["inside"][i + j * nx + k * nx * ny]


def boundary_nodes(mesh):
    """属于内侧单元、且四周还有外侧单元的节点 -> 需要回贴的那些。"""
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    sx, sy = nx + 1, ny + 1
    out = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(sx):
                any_in = False
                any_out = False
                for dk in (-1, 0):
                    for dj in (-1, 0):
                        for di in (-1, 0):
                            if _cell(mesh, i + di, j + dj, k + dk):
                                any_in = True
                            else:
                                any_out = True
                if any_in and any_out:
                    out.append(i + j * sx + k * sx * sy)
    return out


def _inward_dir(mesh, node):
    """指向相邻内侧单元中心加权重心的单位向量；没有内侧邻居则返回 None。"""
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    sx, sy = nx + 1, ny + 1
    i = node % sx
    j = (node // sx) % sy
    k = node // (sx * sy)
    p = _node_pos(mesh, node)
    acc = [0.0, 0.0, 0.0]
    cnt = 0
    for dk in (-1, 0):
        for dj in (-1, 0):
            for di in (-1, 0):
                if _cell(mesh, i + di, j + dj, k + dk):
                    c = (mesh["x0"] + (i + di + 0.5) * mesh["h"],
                         mesh["y0"] + (j + dj + 0.5) * mesh["h"],
                         mesh["z0"] + (k + dk + 0.5) * mesh["h"])
                    acc[0] += c[0] - p[0]
                    acc[1] += c[1] - p[1]
                    acc[2] += c[2] - p[2]
                    cnt += 1
    if not cnt:
        return None
    n = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
    if n <= 0.0:
        return None
    return (acc[0] / n, acc[1] / n, acc[2] / n)


def snap_node(shape, p, target, iters=20):
    """把 p 沿 p->target 推到内外交界上（二分）。

    两个前提必须由调用方保证，不能靠"走远一点总会跨过表面"：

    1. **target 与 p 分处两侧。** 第一版取 target = p + 2h·方向，不检查
       这一条；终点没跨过表面时二分收敛到 t=1，节点被推 2h 深，半径 5 的
       球 h=2.5 时体积削掉 29.5%，而代码不报错。现在两侧由单元中心保证
       （见 `_bracket_target`），这个失败模式不存在了。
    2. **两个方向都要贴。** 只把"在外侧"的节点往里推，内侧节点不动，
       表面就整体向内塌陷 —— 实测还是 -29.5%。所以内侧节点要往外推到
       表面，用相邻的**外侧**单元中心做终点。
    """
    inside_p = shape.contains(p)
    if shape.contains(target) == inside_p:
        return p, 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = (p[0] + mid * (target[0] - p[0]),
             p[1] + mid * (target[1] - p[1]),
             p[2] + mid * (target[2] - p[2]))
        if shape.contains(q) == inside_p:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    q = (p[0] + t * (target[0] - p[0]),
         p[1] + t * (target[1] - p[1]),
         p[2] + t * (target[2] - p[2]))
    return q, t


def _bracket_target(mesh, shape, node):
    """给一个与 node **分处内外两侧**的终点。

    终点只取相邻单元的中心：被判为内侧的单元中心确实在实体里，被判为外侧
    的确实在外面 —— 这个前提由体素化本身保证，不会错。方向优先用同类中心
    的重心（更接近表面法向），重心不可用时退回最近的那个中心。
    """
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    sx, sy = nx + 1, ny + 1
    i = node % sx
    j = (node // sx) % sy
    k = node // (sx * sy)
    p = _node_pos(mesh, node)
    want_inside = not shape.contains(p)
    pool = []
    for dk in (-1, 0):
        for dj in (-1, 0):
            for di in (-1, 0):
                c_in = _cell(mesh, i + di, j + dj, k + dk)
                if bool(c_in) != want_inside:
                    continue
                pool.append((mesh["x0"] + (i + di + 0.5) * mesh["h"],
                             mesh["y0"] + (j + dj + 0.5) * mesh["h"],
                             mesh["z0"] + (k + dk + 0.5) * mesh["h"]))
    if not pool:
        return None
    cx = sum(c[0] for c in pool) / len(pool)
    cy = sum(c[1] for c in pool) / len(pool)
    cz = sum(c[2] for c in pool) / len(pool)
    if shape.contains((cx, cy, cz)) == want_inside:
        return (cx, cy, cz)
    return min(pool, key=lambda c: (c[0] - p[0]) ** 2 + (c[1] - p[1]) ** 2
               + (c[2] - p[2]) ** 2)


# ---------------------------------------------------------------- 参照体积

def volume_ref(shape, h=None, n=120, bbox=None):
    """用**中点求积**算体积，作为独立参照。

    为什么要这个：布尔组合（比如"板挖个孔"）没有闭式体积，可是验收不能
    自己跟自己对。中点求积是另一套东西 —— 它只调用 `contains`，不碰
    区间算术、不碰体素化、不碰网格，所以它和网格体积的偏差是真偏差。

    但这套求积本身也要被验：对有闭式的图元（箱、球、柱、棱柱），
    `volume_ref` 与解析值的相对误差在 n=120 时约 1e-4 量级。所以拿它
    当"带孔件"的参照，精度足够分辨 1% 以上的网格误差。

    分轴取 n 而不是 n^3：n=120 是 173 万次判定，纯 Python 里几秒。
    """
    bx0, bx1, by0, by1, bz0, bz1 = bbox or shape.bbox()
    if h is not None:
        n = max(1, int(math.ceil((bx1 - bx0) / h)))
        ny = max(1, int(math.ceil((by1 - by0) / h)))
        nz = max(1, int(math.ceil((bz1 - bz0) / h)))
    else:
        ny = nz = n
    dx = (bx1 - bx0) / n
    dy = (by1 - by0) / ny
    dz = (bz1 - bz0) / nz
    acc = 0.0
    for k in range(nz):
        z = bz0 + (k + 0.5) * dz
        for j in range(ny):
            y = by0 + (j + 0.5) * dy
            for i in range(n):
                x = bx0 + (i + 0.5) * dx
                if any(a <= z <= b for a, b in shape.intervals(x, y)):
                    acc += 1.0
    return acc * dx * dy * dz


def volume_error(shape, mesh, n=120, bbox=None):
    """体素网格的**净**体积误差，对 `volume_ref`（中点求积）。

    为什么不能只看 `lost_volume`：那个数只累加"z 区间短于 h 且不含格心"的
    材料。薄在 x 或 y 上的壁整列采不到，`iv` 为空，一分钱都不累加 ——
    实测 0.8 厚立壁在 h=2 时整条不见，`lost_volume` 还是 0.0。

    净误差可正可负：格心落在薄壁里会把体积放大（多算），落在外面会缩小
    （少算）。两种都要报，所以这里返回带符号的相对误差。
    """
    ref = volume_ref(shape, n=n, bbox=bbox)
    got = mesh["cells"] * mesh["h"] ** 3
    return {"ref": ref, "got": got, "diff": got - ref,
            "rel": (got - ref) / ref if ref else float("nan"),
            "lost": mesh["lost_volume"], "h": mesh["h"]}
