# -*- coding: utf-8 -*-
"""三维有限元：8 节点六面体（trilinear hex）、2x2x2 高斯积分、结构化箱体网格。

为什么要有这个文件
------------------
README 里写着"在零件上解应力这条路还没通，因为那需要三维四面体网格"。
这句话把两件事混成了一件：**网格生成**和**三维单元**。三维单元是有限元的
标准内容，一晚上能写对；四面体网格生成（Delaunay、边界恢复）是另一个学科。

所以这里只做前者，并且把后者的缺席写死在接口里：`box_mesh` 只生成**长方体
的结构化网格**，不读几何、不做四面体、不做非结构网格。想解一个真实零件，
还差网格生成这一步，这一步这里没有。

顺带把求解器换掉：`fem2d` 用稠密高斯消元，三维一细化就是 O(n^3)，一千自由度
几十秒。这里走 `sparse.pcg`，只做矩阵-向量乘。

验收对着什么
------------
  分片试验    常应变场必须被精确复现（对解析线性场）
  零空间      单个单元的刚度矩阵零空间维数必须恰好是 6
              （三平移 + 三转动；多一个是沙漏，少一个是被锁住）
  悬臂梁      端部挠度必须随网格细化收敛到梁理论 PL^3/(3EI)
  稀疏 vs 稠密 同一方程组两种解法，解必须一致

第三条要连着两句读：hex8 是**会锁死**的单元（三线性插值在纯弯下剪应变不为零），
所以比值从下面逼近 1，而不是一步到位。这不是 bug，是单元本身的性质；
写成"必须等于 1"的测试会逼着人去调参数，那是把单元做错来让测试变绿。

没有做的
--------
不做四面体单元、不做壳与梁的三维壳元（梁在 `beam.py`，是独立的 2D 框架元）、
不做非线性、不做网格生成、不做接触。这些都不是这一个文件里的开关。
"""
from __future__ import annotations

import math

from . import sparse

# 局部节点顺序：底面逆时针四个，顶面逆时针四个（右手系，雅可比为正）
_XI = ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
       (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0))
_GP = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


def D_isotropic(E, nu):
    """各向同性弹性矩阵（6x6，Voigt 顺序 xx,yy,zz,xy,yz,zx）。

    剪切项是 G = E/(2(1+nu))，不是 E/(1+nu)。差一个 2 的表现是弯曲刚、
    拉伸对 —— 而拉伸对的那一半会让人觉得"整体还行"。
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    D = [[0.0] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            D[i][j] = lam
        D[i][i] = lam + 2.0 * G
    for i in (3, 4, 5):
        D[i][i] = G
    return D


def shape(xi, eta, zeta):
    """8 个形函数，与 _XI 同序。"""
    out = []
    for a, b, c in _XI:
        out.append(0.125 * (1 + a * xi) * (1 + b * eta) * (1 + c * zeta))
    return out


def grad_shape(xi, eta, zeta):
    """dN/dxi 等，返回 8x3。"""
    g = []
    for a, b, c in _XI:
        g.append((0.125 * a * (1 + b * eta) * (1 + c * zeta),
                  0.125 * b * (1 + a * xi) * (1 + c * zeta),
                  0.125 * c * (1 + a * xi) * (1 + b * eta)))
    return g


def _jacobian(coords, g):
    """J[i][j] = d x_i / d xi_j，3x3。"""
    J = [[0.0] * 3 for _ in range(3)]
    for n in range(8):
        for i in range(3):
            for j in range(3):
                J[i][j] += coords[n][i] * g[n][j]
    return J


def _inv3(J):
    a, b, c = J[0]
    d, e, f = J[1]
    g, h, i = J[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det <= 0.0:
        # 节点顺序反了 / 单元被压扁。负雅可比会让刚度矩阵负定，
        # 解出来是一个能看的数，但符号是反的 —— 必须当场拒绝。
        raise ValueError("单元雅可比 %.3e 非正：节点顺序反了或单元退化" % det)
    inv = [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
           [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
           [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]
    return inv, det


def element_stiffness(coords, D):
    """24x24 单元刚度矩阵（每个节点 3 个自由度）。"""
    Ke = [[0.0] * 24 for _ in range(24)]
    for xi in _GP:
        for eta in _GP:
            for zeta in _GP:
                g = grad_shape(xi, eta, zeta)
                Jinv, det = _inv3(_jacobian(coords, g))
                # dN/dx = Jinv^T · dN/dxi
                dN = []
                for n in range(8):
                    dN.append(tuple(sum(Jinv[k][i] * g[n][k] for k in range(3))
                                    for i in range(3)))
                B = [[0.0] * 24 for _ in range(6)]
                for n in range(8):
                    dx, dy, dz = dN[n]
                    B[0][3 * n] = dx
                    B[1][3 * n + 1] = dy
                    B[2][3 * n + 2] = dz
                    B[3][3 * n] = dy
                    B[3][3 * n + 1] = dx
                    B[4][3 * n + 1] = dz
                    B[4][3 * n + 2] = dy
                    B[5][3 * n] = dz
                    B[5][3 * n + 2] = dx
                DB = [[sum(D[r][c] * B[c][b] for c in range(6)) for b in range(24)]
                      for r in range(6)]
                w = det
                for a in range(24):
                    Ba = [B[r][a] for r in range(6)]
                    if not any(Ba):
                        continue
                    for b in range(24):
                        s = sum(Ba[r] * DB[r][b] for r in range(6))
                        if s:
                            Ke[a][b] += s * w
    return Ke


def box_mesh(nx, ny, nz, lx=1.0, ly=1.0, lz=1.0):
    """结构化箱体网格。节点编号 i + j*(nx+1) + k*(nx+1)*(ny+1)。

    只生成长方体。这不是"暂时的限制"，是这一层的边界：解真实零件要
    四面体网格生成，那一步没写。
    """
    nodes = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append((lx * i / nx, ly * j / ny, lz * k / nz))
    sx, sy = nx + 1, ny + 1
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = i + j * sx + k * sx * sy
                elems.append((n0, n0 + 1, n0 + 1 + sx, n0 + sx,
                              n0 + sx * sy, n0 + 1 + sx * sy,
                              n0 + 1 + sx + sx * sy, n0 + sx + sx * sy))
    return {"nodes": nodes, "elems": elems, "nx": nx, "ny": ny, "nz": nz,
            "h": (lx / nx, ly / ny, lz / nz)}


def assemble(nodes, elems, D):
    """通用装配：逐个单元算刚度矩阵。"""
    A = sparse.zeros(3 * len(nodes))
    for e in elems:
        coords = [nodes[n] for n in e]
        Ke = element_stiffness(coords, D)
        idx = [3 * n + c for n in e for c in range(3)]
        sparse.add_block(A, idx, Ke)
    return A


def assemble_box(mesh, D):
    """箱体网格的装配：**单元刚度矩阵只算一次**。

    结构化网格里每个单元都是同一个长方体的平移，所以 24x24 完全一样。
    逐个算的话 nx=24 的网格要算 648 次，纯 Python 里是几十秒；算一次是
    0.03 秒。这不是优化技巧，是这个网格的**性质**，用不上它是浪费。
    """
    nodes, elems = mesh["nodes"], mesh["elems"]
    A = sparse.zeros(3 * len(nodes))
    first = elems[0]
    Ke = element_stiffness([nodes[n] for n in first], D)
    for e in elems:
        idx = [3 * n + c for n in e for c in range(3)]
        sparse.add_block(A, idx, Ke)
    return A


# ---------------------------------------------------------------- 验收

def _rank(M, tol=1e-8):
    """稠密矩阵的秩（带主元的行消去）。零空间维数靠它算。"""
    m = [row[:] for row in M]
    rows, cols = len(m), len(m[0])
    scale = max((abs(v) for row in m for v in row), default=1.0) or 1.0
    r = 0
    for c in range(cols):
        p = max(range(r, rows), key=lambda i: abs(m[i][c]))
        if abs(m[p][c]) <= tol * scale:
            continue
        m[r], m[p] = m[p], m[r]
        piv = m[r][c]
        for i in range(r + 1, rows):
            f = m[i][c] / piv
            if f:
                for j in range(c, cols):
                    m[i][j] -= f * m[r][j]
        r += 1
        if r == rows:
            break
    return r


def rigid_body_nullity(h=1.0, E=1000.0, nu=0.3):
    """单个 hex8 的零空间维数。三维刚体运动恰好 6 个：三平移 + 三转动。

    多一个说明有沙漏（2x2x2 积分理论上没有，但插值写错会造出来）；
    少一个说明被锁住。两个方向都是错的，所以用等号断言。
    """
    coords = [(0, 0, 0), (h, 0, 0), (h, h, 0), (0, h, 0),
              (0, 0, h), (h, 0, h), (h, h, h), (0, h, h)]
    Ke = element_stiffness(coords, D_isotropic(E, nu))
    return {"rank": _rank(Ke), "nullity": 24 - _rank(Ke), "size": 24}


def patch_test(nx=3, ny=3, nz=3, E=1000.0, nu=0.3, strain=(1e-3, -4e-4, 2e-4, 3e-4, 0.0, 0.0)):
    """分片试验：给定一个常应变场，位移是坐标的线性函数。

    线性位移场能被三线性单元**精确**表示，所以边界上给够位移之后，
    内部节点的解必须一位不差。通不过说明 B 矩阵或雅可比有错。
    """
    mesh = box_mesh(nx, ny, nz, 1.0, 1.0, 1.0)
    nodes = mesh["nodes"]
    exx, eyy, ezz, gxy, gyz, gzx = strain
    # 由应变反推位移场：u = A x，A 的对称部分等于应变
    A = [[exx, gxy / 2.0, gzx / 2.0],
         [gxy / 2.0, eyy, gyz / 2.0],
         [gzx / 2.0, gyz / 2.0, ezz]]

    def u(p):
        return tuple(sum(A[i][j] * p[j] for j in range(3)) for i in range(3))

    D = D_isotropic(E, nu)
    K = assemble_box(mesh, D)
    ndof = 3 * len(nodes)
    f = [0.0] * ndof
    sx, sy = nx + 1, ny + 1
    fixed = {}
    for n, p in enumerate(nodes):
        i, j, k = n % sx, (n // sx) % sy, n // (sx * sy)
        if i in (0, nx) or j in (0, ny) or k in (0, nz):
            for c in range(3):
                fixed[3 * n + c] = u(p)[c]
    b = sparse.apply_dirichlet(K, f, fixed)
    res = sparse.pcg(K, b, tol=1e-12, maxit=20000)
    if not res["converged"]:
        return {"ok": False, "reason": "pcg 未收敛", "rel_res": res["rel_res"]}
    err = 0.0
    ref = 0.0
    for n, p in enumerate(nodes):
        want = u(p)
        for c in range(3):
            err = max(err, abs(res["x"][3 * n + c] - want[c]))
            ref = max(ref, abs(want[c]))
    return {"ok": err <= 1e-9 * max(ref, 1.0), "max_err": err, "rel_err": err / max(ref, 1e-30),
            "pcg_iters": res["iters"]}


def cantilever(nx=16, ny=4, nz=4, L=100.0, b=12.0, h=12.0,
               E=210000.0, nu=0.3, P=100.0, tol=1e-11):
    """三维悬臂梁：一端全约束，另一端面上均布剪力。

    参照是梁理论 delta = P L^3 / (3 E I)，I = b h^3 / 12。

    hex8 在纯弯下会锁死：三线性插值让弯曲变形里混进了非零的剪应变，
    单元因此偏刚，比值从**下面**逼近 1。实测（L/h=8.3，截面 4x4）：

        nx=8   0.692      nx=16  0.892      nx=24  0.944

    同一套代码换成短梁（L/h=5）：0.779 / 0.934 / 0.971。也就是说锁定程度
    随长径比变化，而不是一个常数 —— 所以测试断言的是"单调上升且逼近 1"，
    不是"等于 1"。写成等于 1 会逼着人去调一个参数让测试变绿，那是把单元做错。
    """
    mesh = box_mesh(nx, ny, nz, L, b, h)
    nodes = mesh["nodes"]
    D = D_isotropic(E, nu)
    K = assemble_box(mesh, D)
    ndof = 3 * len(nodes)
    f = [0.0] * ndof
    sx, sy = nx + 1, ny + 1
    tip = [i for i in range(len(nodes)) if i % sx == nx]
    per = P / len(tip)
    for n in tip:
        f[3 * n + 2] = -per          # z 向（厚度方向）加载
    fixed = {}
    for n in range(len(nodes)):
        if n % sx == 0:
            for c in range(3):
                fixed[3 * n + c] = 0.0
    bvec = sparse.apply_dirichlet(K, f, fixed)
    res = sparse.pcg(K, bvec, tol=tol, maxit=30000)
    I = b * h ** 3 / 12.0
    theory = P * L ** 3 / (3.0 * E * I)
    uz = [-res["x"][3 * n + 2] for n in tip]
    got = sum(uz) / len(uz)
    return {"ratio": got / theory, "tip": got, "theory": theory,
            "converged": res["converged"], "rel_res": res["rel_res"],
            "pcg_iters": res["iters"], "dof": ndof, "elements": len(mesh["elems"])}


def sparse_vs_dense(nx=4, ny=2, nz=2, L=10.0, b=2.0, h=2.0, E=1000.0, nu=0.3, P=1.0):
    """同一方程组两种解法：稀疏 PCG 与稠密高斯消元。

    这一条抓的是装配错误 —— 行列写反的矩阵仍然对称正定，PCG 照样收敛，
    只是收敛到另一个问题的解。
    """
    mesh = box_mesh(nx, ny, nz, L, b, h)
    nodes = mesh["nodes"]
    D = D_isotropic(E, nu)
    K = assemble_box(mesh, D)
    ndof = 3 * len(nodes)
    f = [0.0] * ndof
    sx = nx + 1
    for n in range(len(nodes)):
        if n % sx == nx:
            f[3 * n + 2] = -P / ((ny + 1) * (nz + 1))
    fixed = {}
    for n in range(len(nodes)):
        if n % sx == 0:
            for c in range(3):
                fixed[3 * n + c] = 0.0
    bvec = sparse.apply_dirichlet(K, f, fixed)
    sp = sparse.pcg(K, bvec, tol=1e-13, maxit=20000)
    dn = sparse.solve_dense(K, bvec)
    scale = max((abs(v) for v in dn), default=1.0) or 1.0
    diff = max(abs(a - b2) for a, b2 in zip(sp["x"], dn)) / scale
    return {"rel_diff": diff, "sparse_converged": sp["converged"],
            "dof": ndof, "pcg_iters": sp["iters"]}
