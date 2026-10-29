# -*- coding: utf-8 -*-
"""tetmesh.py -- 体素 -> 四面体网格，边界回贴，拓扑与体积自检。

核心：Freudenthal（Kuhn）六分
---------------------------
一个六面体切成 6 个四面体，**全部共用同一条主对角线** 000-111。
这不是随便切的：在结构化网格里，相邻两格在公共面上诱导出的二维三角化
必须一样，否则网格不连续（会出现"悬边"，装配出的刚度矩阵在缝上不相连，
解出来的位移场在缝上是断的，而 PCG 照样收敛 —— 那是最难查的一类错）。

取六个坐标增量顺序的排列，每条路径给出一个四面体：

    xyz: 000 100 110 111      xzy: 000 100 101 111
    yxz: 000 010 110 111      yzx: 000 010 011 111
    zxy: 000 001 101 111      zyx: 000 001 011 111

公共面 x=0 被切成 (000,010,011) 与 (000,001,011)，x=1 被切成
(100,110,111) 与 (100,101,111) —— 左边那格的 x=1 面与右边那格的 x=0 面
是同一条对角线。y、z 面同理。所以**只要每格都用同一个方向，网格天然连续**。
`check` 里的 `faces_interior == 2 * ...` 那一条就是在验这件事：内部面必须
恰好被两个四面体共用，边界面恰好被一个共用。任何一个面出现 3 次或 1 次
（在内部）都说明切法不一致。

有向体积
--------
6 个四面体里有一半按上面的顺序算出来是负的（坐标顺序决定手性）。所以
逐个体算有向体积、负了就换两个节点。**装配之前每个单元都必须为正**，
负体积单元的刚度矩阵是负定的，PCG 会不收敛 —— 或者更糟，收敛到一个
符号反了的解。
"""
from __future__ import annotations

import math

from . import voxel as _voxel

__all__ = ["build", "tet10", "volume", "topology", "check", "tet_volume"]

# 六面体局部编号 -> 二进制位码（x 是最低位）。与 fem3d._XI 同序：
# _XI 的顺序是 000,100,110,010,001,101,111,011
_HEX_BITS = (0, 1, 3, 2, 4, 5, 7, 6)
# 六个 Kuhn 四面体（位码）
_KUHN = ((0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
         (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7))


def _hex_tets_local():
    """把 Kuhn 四面体的位码翻成六面体局部编号。"""
    inv = [0] * 8
    for local, bit in enumerate(_HEX_BITS):
        inv[bit] = local
    return [tuple(inv[b] for b in t) for t in _KUHN]


_KUHN_LOCAL = _hex_tets_local()


def tet_volume(p0, p1, p2, p3):
    """有向体积（正 = 右手系）。"""
    ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    cx, cy, cz = p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]
    det = (ax * (by * cz - bz * cy)
           - ay * (bx * cz - bz * cx)
           + az * (bx * cy - by * cx))
    return det / 6.0


def build(shape, h, project=True, bbox=None, verbose=False):
    """实体 -> 四面体网格。

    project=True 时边界节点沿朝内方向回贴到真实表面；回贴后逐个检查所属
    四面体的有向体积，**变非正就退回原位置**。退回去的节点数记在
    `reverted` 里 —— 它不是零就说明这个 h 对这个形状太粗（回贴把单元
    压塌了），应该减小 h，而不是相信这份网格。
    """
    vm = _voxel.voxelize(shape, h, bbox)
    nx, ny, nz = vm["nx"], vm["ny"], vm["nz"]
    sx, sy = nx + 1, ny + 1
    nnode = sx * sy * (nz + 1)
    nodes = [_voxel._node_pos(vm, n) for n in range(nnode)]

    cells = []
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not _voxel._cell(vm, i, j, k):
                    continue
                n0 = i + j * sx + k * sx * sy
                hexn = (n0, n0 + 1, n0 + 1 + sx, n0 + sx,
                        n0 + sx * sy, n0 + 1 + sx * sy,
                        n0 + 1 + sx + sx * sy, n0 + sx + sx * sy)
                cells.append(hexn)
                for t in _KUHN_LOCAL:
                    q = (hexn[t[0]], hexn[t[1]], hexn[t[2]], hexn[t[3]])
                    if tet_volume(nodes[q[0]], nodes[q[1]], nodes[q[2]],
                                  nodes[q[3]]) < 0.0:
                        q = (q[0], q[1], q[3], q[2])
                    tets.append(q)

    reverted = 0
    projected = 0
    if project:
        # 节点 -> 所属四面体（只关心边界节点）
        incident = {}
        bset = set(_voxel.boundary_nodes(vm))
        if bset:
            for ti, t in enumerate(tets):
                for n in t:
                    if n in bset:
                        incident.setdefault(n, []).append(ti)
            for n in sorted(bset):
                target = _voxel._bracket_target(vm, shape, n)
                if target is None:
                    continue
                q, _t = _voxel.snap_node(shape, nodes[n], target)
                if q == nodes[n]:
                    continue
                old = nodes[n]
                # 已经在表面上的节点不该动。二分从内侧往外的第一步就能
                # 跨过表面，给出一个几乎为零的位移；照搬会引入 1e-7 量级
                # 的体积漂移，让"长方体体积必须精确"这条失效。
                if max(abs(q[i] - old[i]) for i in range(3)) <= 1e-6 * h:
                    continue
                nodes[n] = q
                bad = False
                for ti in incident.get(n, ()):
                    a, b, c, e = tets[ti]
                    if tet_volume(nodes[a], nodes[b], nodes[c], nodes[e]) <= 0.0:
                        bad = True
                        break
                if bad:
                    nodes[n] = old
                    reverted += 1
                else:
                    projected += 1

    return compact({"nodes": nodes, "tets": tets, "cells": cells,
                    "h": h, "nx": nx, "ny": ny, "nz": nz,
                    "lost_volume": vm["lost_volume"], "inside_cells": len(cells),
                    "projected": projected, "reverted": reverted,
                    "bbox": (vm["x0"], vm["x0"] + nx * h,
                             vm["y0"], vm["y0"] + ny * h,
                             vm["z0"], vm["z0"] + nz * h)})


def compact(mesh):
    """删掉没有被任何四面体用到的节点，重新编号。

    为什么非做不可：体素化在实体外面留了一层空格，那些格点一个单元都不
    属于，刚度矩阵里对应整行整列全是零。零行在 PCG 里是**静默**的 ——
    残差本来就是零，那些自由度永远不会被更新，于是解在那里取初值（0），
    而方程组的收敛判据完全不看它们。实测：单位立方体 h=0.5 的网格有
    125 个节点、只有 27 个真的被用到，TET10 的位移场在那 98 个点上
    是 0，而真值不为零 —— 分片试验因此报 10.6% 误差，看起来像单元写错了，
    实际是网格里躺着一堆没人管的自由度。
    """
    tets = mesh["tets"]
    used = set()
    for t in tets:
        used.update(t)
    order = sorted(used)
    remap = {old: new for new, old in enumerate(order)}
    out = dict(mesh)
    out["nodes"] = [mesh["nodes"][n] for n in order]
    out["tets"] = [tuple(remap[n] for n in t) for t in tets]
    out["dropped_nodes"] = len(mesh["nodes"]) - len(order)
    return out


def volume(mesh):
    """网格体积 = 所有四面体体积之和。"""
    nodes, tets = mesh["nodes"], mesh["tets"]
    return sum(tet_volume(nodes[a], nodes[b], nodes[c], nodes[e])
               for a, b, c, e in tets)


def topology(mesh):
    """V/E/F/T 与欧拉示性数。

    一个实心球（任何不含通孔、不含空腔的实体）满足 V - E + F - T = 1。
    通孔（比如一个带孔的板）每多一个孔，这个数减 1 —— 所以它同时是
    "网格有没有把孔丢掉"的判据。
    """
    tets = mesh["tets"]
    verts = set()
    edges = {}
    faces = {}
    for t in tets:
        verts.update(t)
        a, b, c, d = t
        for e in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            k = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            edges[k] = edges.get(k, 0) + 1
        for f in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            k = tuple(sorted(f))
            faces[k] = faces.get(k, 0) + 1
    V, E, F, T = len(verts), len(edges), len(faces), len(tets)
    boundary = sum(1 for v in faces.values() if v == 1)
    interior = sum(1 for v in faces.values() if v == 2)
    bad = sum(1 for v in faces.values() if v > 2)
    return {"V": V, "E": E, "F": F, "T": T, "euler": V - E + F - T,
            "faces_boundary": boundary, "faces_interior": interior,
            "faces_nonmanifold": bad,
            "edges_boundary": sum(1 for v in edges.values() if v == 1)}


def tet10(mesh):
    """TET4 -> TET10：每条边加中点，二次插值。

    线性四面体在纯弯下会锁死（常应变单元表示不了弯曲应变），而且是**严重**
    锁死。二次单元把这件事解决掉大半。代价是节点数和装配时间。

    边中点编号规则（与 fem3d.tet10_stiffness 一致）：
        5=(1,2) 6=(2,3) 7=(3,1) 8=(1,4) 9=(2,4) 10=(3,4)
    """
    nodes = list(mesh["nodes"])
    mid = {}
    out = []
    for a, b, c, d in mesh["tets"]:
        ids = [a, b, c, d]
        q = list(ids)
        for (u, v) in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)):
            k = (ids[u], ids[v]) if ids[u] < ids[v] else (ids[v], ids[u])
            if k not in mid:
                p, r = nodes[k[0]], nodes[k[1]]
                nodes.append((0.5 * (p[0] + r[0]), 0.5 * (p[1] + r[1]),
                              0.5 * (p[2] + r[2])))
                mid[k] = len(nodes) - 1
            q.append(mid[k])
        out.append(tuple(q))
    return {"nodes": nodes, "tets": out, "tet4": mesh["tets"],
            "h": mesh["h"], "edges": len(mid)}


def check(shape=None, h=2.0, project=True, bbox=None):
    """把能自检的都自检一遍。

    `ok` 只在**结构性**条件上成立：正体积、面流形、无退化单元。
    体积与解析值的偏差不进 `ok` —— 那个随 h 收敛，是精度不是正确性。
    """
    m = build(shape, h, project=project, bbox=bbox) if shape is not None else None
    if m is None:
        raise ValueError("check 需要给一个实体")
    nodes, tets = m["nodes"], m["tets"]
    neg = 0
    minv = None
    for a, b, c, e in tets:
        v = tet_volume(nodes[a], nodes[b], nodes[c], nodes[e])
        if v <= 0.0:
            neg += 1
        if minv is None or v < minv:
            minv = v
    topo = topology(m)
    vol = volume(m)
    want = shape.volume()
    return {"ok": neg == 0 and topo["faces_nonmanifold"] == 0 and len(tets) > 0,
            "tets": len(tets), "nodes": len(nodes),
            "negative": neg, "min_volume": minv,
            "volume": vol, "volume_exact": want,
            "volume_err": None if want is None else abs(vol - want) / want,
            "lost_volume": m["lost_volume"],
            "projected": m["projected"], "reverted": m["reverted"],
            "euler": topo["euler"], "faces_boundary": topo["faces_boundary"],
            "faces_interior": topo["faces_interior"],
            "faces_nonmanifold": topo["faces_nonmanifold"],
            "V": topo["V"], "E": topo["E"], "F": topo["F"], "T": topo["T"]}
