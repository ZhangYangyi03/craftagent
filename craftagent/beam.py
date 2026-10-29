# -*- coding: utf-8 -*-
"""二维框架单元：轴向拉压 + 欧拉-伯努利弯曲，带坐标变换。

为什么单独一个文件
------------------
`fem3d` 是实体单元（每个节点 3 个平动自由度），适合"一块料"。而机床的夹具、
刀柄、悬伸的刀杆是**细长构件**：用实体单元算它们，要么单元数爆炸，要么锁死。
梁单元一个单元就能给出精确的节点位移，因为欧拉-伯努利梁在**点载荷下的
节点位移是解析精确的**（不是近似的）。

这个性质让它可以当尺子用：`fem3d` 的悬臂梁比值收敛到 1，参照就是这里的
梁理论。两个独立实现给同一个数，才说明那个 1 不是自己编的。

验收对着什么
------------
  端部挠度    PL^3/(3EI)，欧拉-伯努利在节点上**精确**
  端部转角    PL^2/(2EI)
  轴向伸长    PL/(EA)
  简支梁      PL^3/(48EI)
  刚体模态    单个单元零空间恰好 3 维（两平移 + 一转动）
  细化不变    悬臂梁分 1 / 2 / 4 个单元，端部挠度必须**一位不差**

最后一条是这里最有信息量的一条：如果分单元之后结果变了，说明坐标变换或者
装配有错 —— 而错的表现通常是"稍微软了一点"，看起来像网格粗细的正常差别。

没有做的
--------
不做剪切变形（Timoshenko）；不做大变形（几何非线性）；不做三维梁（只有平面内）。
刀杆的剪切变形在 L/h < 10 时不能忽略，这一条写在 `shear_note()` 里，不藏在注释里。
"""
from __future__ import annotations

import math

from . import sparse


def element_stiffness_local(E, A, I, L):
    """6x6 局部刚度矩阵，自由度顺序 (u1, v1, t1, u2, v2, t2)。

    弯曲那 4x4 块是欧拉-伯努利梁的标准形式。写成符号转置的样子（12, 6L, -12, 6L）
    而不是"看起来对称"的排列：6L 与 -6L 的符号错一位，梁会往反方向弯，
    而反方向弯的梁在对称载荷下挠度一样 —— 这种错只有在非对称载荷下才露出来。
    """
    k = [[0.0] * 6 for _ in range(6)]
    ea = E * A / L
    k[0][0] = ea
    k[0][3] = -ea
    k[3][0] = -ea
    k[3][3] = ea

    ei = E * I / L ** 3
    b = [[12.0, 6 * L, -12.0, 6 * L],
         [6 * L, 4 * L * L, -6 * L, 2 * L * L],
         [-12.0, -6 * L, 12.0, -6 * L],
         [6 * L, 2 * L * L, -6 * L, 4 * L * L]]
    # 弯曲自由度是 (v1, t1, v2, t2) = (1, 2, 4, 5)，**不是** 1,2,3,4 ——
    # 3 是 u2。按 1,2,3,4 铺过去，轴向与弯曲会互相污染，而刚度矩阵仍然对称，
    # 解出来的梁"能弯"，只是挠度不对，且 t2 那一行全零（等于铰接）。
    bdof = (1, 2, 4, 5)
    for a in range(4):
        for c in range(4):
            k[bdof[a]][bdof[c]] = ei * b[a][c]
    return k


def transform(angle):
    """全局 -> 局部的 6x6 变换矩阵。angle 是杆轴与 x 轴的夹角。"""
    c, s = math.cos(angle), math.sin(angle)
    T = [[0.0] * 6 for _ in range(6)]
    for base in (0, 3):
        T[base][base] = c
        T[base][base + 1] = s
        T[base + 1][base] = -s
        T[base + 1][base + 1] = c
        T[base + 2][base + 2] = 1.0
    return T


def element_stiffness_global(p1, p2, E, A, I):
    """整体坐标下的 6x6 单元刚度矩阵：Ke = T^T k T。"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L <= 1e-12:
        raise ValueError("单元长度为 0")
    k = element_stiffness_local(E, A, I, L)
    T = transform(math.atan2(dy, dx))
    kt = [[sum(k[i][m] * T[m][j] for m in range(6)) for j in range(6)]
          for i in range(6)]
    return [[sum(T[m][i] * kt[m][j] for m in range(6)) for j in range(6)]
            for i in range(6)], L


def solve_frame(nodes, elements, E, A, I, fixed, loads):
    """装配并求解。elements 是 [(节点a, 节点b)]，loads 是 {自由度: 力}。

    自由度编号 3*node + (0,1,2) = (u, v, theta)。
    """
    n = 3 * len(nodes)
    K = sparse.zeros(n)
    lengths = []
    for a, b in elements:
        Ke, L = element_stiffness_global(nodes[a], nodes[b], E, A, I)
        idx = [3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2]
        sparse.add_block(K, idx, Ke)
        lengths.append(L)
    f = [0.0] * n
    for dof, val in loads.items():
        f[dof] += val
    b = sparse.apply_dirichlet(K, f, dict(fixed))
    res = sparse.pcg(K, b, tol=1e-13, maxit=20000)
    return {"x": res["x"], "converged": res["converged"],
            "rel_res": res["rel_res"], "iters": res["iters"],
            "lengths": lengths, "dof": n}


def cantilever(n=1, L=100.0, E=210000.0, A=100.0, I=1000.0, P=1.0):
    """端部集中力的悬臂梁。理论值 delta = PL^3/(3EI)，theta = PL^2/(2EI)。

    n 是分几个单元。欧拉-伯努利梁在点载荷下节点位移是解析精确的，
    所以 n 变了结果不该变 —— 变了就是变换或装配错了。
    """
    nodes = [(L * k / n, 0.0) for k in range(n + 1)]
    elements = [(k, k + 1) for k in range(n)]
    fixed = {0: 0.0, 1: 0.0, 2: 0.0}
    loads = {3 * n + 1: -P}
    r = solve_frame(nodes, elements, E, A, I, fixed, loads)
    tip_v = r["x"][3 * n + 1]
    tip_t = r["x"][3 * n + 2]
    return {"tip": abs(tip_v), "theory": P * L ** 3 / (3 * E * I),
            "theta": abs(tip_t), "theta_theory": P * L ** 2 / (2 * E * I),
            "converged": r["converged"], "iters": r["iters"], "elements": n}


def axial(n=1, L=100.0, E=210000.0, A=100.0, P=1000.0):
    """纯拉伸：delta = PL/(EA)。弯曲自由度不参与，所以这条单独测一次。"""
    nodes = [(L * k / n, 0.0) for k in range(n + 1)]
    elements = [(k, k + 1) for k in range(n)]
    fixed = {0: 0.0, 1: 0.0, 2: 0.0}
    loads = {3 * n: P}
    r = solve_frame(nodes, elements, E, A, I=1e-6, fixed=fixed, loads=loads)
    return {"tip": r["x"][3 * n], "theory": P * L / (E * A),
            "converged": r["converged"]}


def simply_supported(n=8, L=100.0, E=210000.0, A=100.0, I=1000.0, P=1.0):
    """简支梁中点集中力：delta = PL^3/(48EI)。两端铰支（只约束 v 与 u）。"""
    nodes = [(L * k / n, 0.0) for k in range(n + 1)]
    elements = [(k, k + 1) for k in range(n)]
    fixed = {0: 0.0, 1: 0.0, 3 * n + 1: 0.0, 3 * n: 0.0}
    mid = n // 2
    loads = {3 * mid + 1: -P}
    r = solve_frame(nodes, elements, E, A, I, fixed, loads)
    return {"mid": abs(r["x"][3 * mid + 1]), "theory": P * L ** 3 / (48 * E * I),
            "converged": r["converged"]}


def rigid_modes(E=210000.0, A=100.0, I=1000.0, L=100.0):
    """单个梁单元的零空间维数必须恰好是 3：两平移 + 一转动。

    多一个是机构（能量漏掉、位移场光滑）；少一个是被锁住。
    """
    k = element_stiffness_local(E, A, I, L)
    m = [row[:] for row in k]
    scale = max(abs(v) for row in m for v in row) or 1.0
    rank = 0
    for c in range(6):
        p = max(range(rank, 6), key=lambda i: abs(m[i][c]))
        if abs(m[p][c]) <= 1e-9 * scale:
            continue
        m[rank], m[p] = m[p], m[rank]
        for i in range(rank + 1, 6):
            f = m[i][c] / m[rank][c]
            if f:
                for j in range(c, 6):
                    m[i][j] -= f * m[rank][j]
        rank += 1
    return {"rank": rank, "nullity": 6 - rank}


def inclined(L=100.0, angle=math.pi / 4, E=210000.0, A=100.0, I=1000.0, P=1.0):
    """斜杆端部竖向力：把竖向力分解成轴向与横向，两个理论值分别对。

    这一条是坐标变换的唯一真检验。角度搞反（c 与 s 互换）时，45 度杆的
    结果只差一个因子，看起来像"网格效应"；而 0 度与 90 度两种极端下
    它恰好还是对的。
    """
    dx = L * math.cos(angle)
    dy = L * math.sin(angle)
    nodes = [(0.0, 0.0), (dx, dy)]
    r = solve_frame(nodes, [(0, 1)], E, A, I, {0: 0.0, 1: 0.0, 2: 0.0},
                    {4: -P})
    # 理论：竖向向下的力在杆轴上分量是**压**（-P*sin），垂直杆轴的分量引起弯曲。
    # 符号写错不会让这条测试变红 —— 只要把理论值也写成同一个符号就行，
    # 所以这里把两边的符号都写死成"压为负、向下为负"。
    axial_theory = -P * math.sin(angle) * L / (E * A)
    bend_theory = P * math.cos(angle) * L ** 3 / (3 * E * I)
    u = r["x"][3]
    v = r["x"][4]
    # 把端部位移投影回杆轴与法向
    along = u * math.cos(angle) + v * math.sin(angle)
    across = -u * math.sin(angle) + v * math.cos(angle)
    return {"along": along, "along_theory": axial_theory,
            "across": across, "across_theory": -bend_theory,
            "converged": r["converged"]}


def shear_note(L=100.0, h=10.0, E=210000.0, nu=0.3, G=None):
    """剪切变形什么时候不能忽略：Timoshenko 的修正项是 12EI/(kGA L^2)。

    欧拉-伯努利漏掉的就是这一项。给出比值，让人自己判断该不该换单元，
    而不是在文档里写一句"细长杆适用"。
    """
    G = G if G is not None else E / (2 * (1 + nu))
    k = 5.0 / 6.0
    A = h * 1.0
    I = 1.0 * h ** 3 / 12.0
    bend = L ** 3 / (3 * E * I)
    shear = L / (k * G * A)
    return {"bending": bend, "shear": shear, "ratio": shear / bend,
            "note": "ratio > 0.05 时欧拉-伯努利低估挠度 5% 以上"}
