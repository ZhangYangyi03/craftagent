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


# ---------------------------------------------------------------- 四面体单元
#
# 上面那一整块只认六面体，而六面体只能网格化长方体。要解一个真实零件，
# 缺的从来不是单元而是网格（见 tetmesh.py）。这里补上四面体单元，让
# "网格化了的东西"能直接进来求解。
#
# 线性四面体（TET4）是常应变单元：B 是常数矩阵，刚度矩阵一次算完。
# 它在纯弯下**严重**锁死 —— 常应变表示不了弯曲应变，加细之前误差可以
# 到 60% 以上。二次四面体（TET10）每条边加中点，B 是坐标的线性函数，
# 弯曲就活了。代价是节点数翻几倍、装配慢一个量级。两个都留着：
# TET4 用来快速验几何与装配，TET10 用来出数。

def _tet_grad_L(coords):
    """四个体积坐标的梯度 + 体积。

    令 v_j = p_{j+1} - p_1（j=1,2,3），J 以 v_j 为**列**。则
        [L2,L3,L4]^T = J^{-1} (x - p1)
    所以 grad L2/L3/L4 就是 J^{-1} 的三行，grad L1 = -(三者之和)。

    这里踩过一个很隐蔽的坑，值得写下来：第一版把 v 按坐标方向打包
    （`v[i] = (p2[i]-p1[i], p3[i]-p1[i], p4[i]-p1[i])`），拼出来的矩阵
    其实是 J 的**转置**，于是算出的梯度是 J^{-ᵀ} 的行，也就是转置过的
    梯度。它绕过了所有常规检查：

        体积     det(J^T) = det(J)，对的
        零空间   6，对的
        分片试验 通过 —— 因为测试用的是**对称**应变张量，
                 转置后对称部分不变
        单元能量 与解析值一致 —— 因为对照用的 B 是同一个错 B

    只有弯曲问题能暴露它：弯曲的位移梯度不对称。实测同一根梁，六面体
    给 0.883（梁理论），四面体给 0.022 —— 刚度高了 40 倍。所以验收里
    加了一条**纯转动**：u = ω × x 的应变必须为零、内力必须为零。
    转置一出现这一条立刻红，而对称应变的分片试验永远发现不了。
    """
    p1, p2, p3, p4 = coords
    v = [(p2[i] - p1[i], p3[i] - p1[i], p4[i] - p1[i]) for i in range(3)]
    # 现在 v[i] 是第 i 个坐标方向上的三个差分，即 J 的第 i 行
    a, b, c = v[0]
    d, e, f = v[1]
    g, h, i2 = v[2]
    det = a * (e * i2 - f * h) - b * (d * i2 - f * g) + c * (d * h - e * g)
    if det == 0.0:
        raise ValueError("四面体退化（体积为零）")
    inv = [[(e * i2 - f * h) / det, (c * h - b * i2) / det, (b * f - c * e) / det],
           [(f * g - d * i2) / det, (a * i2 - c * g) / det, (c * d - a * f) / det],
           [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]
    gL = [None] * 4
    gL[1] = tuple(inv[0])
    gL[2] = tuple(inv[1])
    gL[3] = tuple(inv[2])
    gL[0] = (-(gL[1][0] + gL[2][0] + gL[3][0]),
             -(gL[1][1] + gL[2][1] + gL[3][1]),
             -(gL[1][2] + gL[2][2] + gL[3][2]))
    return gL, abs(det) / 6.0


def _bt_db(B, DB, n):
    """B^T D B，n 为节点数。每列只有 3 个非零（正应变一行 + 两个剪应变行），
    所以内层求和只跑 3 次而不是 6 次。"""
    Ke = [[0.0] * (3 * n) for _ in range(3 * n)]
    cols = []
    for a in range(3 * n):
        cols.append([(r, B[r][a]) for r in range(6) if B[r][a]])
    for a in range(3 * n):
        ca = cols[a]
        if not ca:
            continue
        row = Ke[a]
        for b in range(3 * n):
            s = 0.0
            for r, v in ca:
                s += v * DB[r][b]
            if s:
                row[b] += s
    return Ke


def _bmat(dN, n):
    """dN: n 个 (dx,dy,dz) -> 6x(3n) 应变-位移矩阵。"""
    B = [[0.0] * (3 * n) for _ in range(6)]
    for k in range(n):
        dx, dy, dz = dN[k]
        B[0][3 * k] = dx
        B[1][3 * k + 1] = dy
        B[2][3 * k + 2] = dz
        B[3][3 * k] = dy
        B[3][3 * k + 1] = dx
        B[4][3 * k + 1] = dz
        B[4][3 * k + 2] = dy
        B[5][3 * k] = dz
        B[5][3 * k + 2] = dx
    return B


def tet4_stiffness(coords, D):
    """12x12 常应变四面体刚度矩阵。"""
    gL, V = _tet_grad_L(coords)
    B = _bmat(gL, 4)
    DB = [[sum(D[r][c] * B[c][b] for c in range(6)) for b in range(12)]
          for r in range(6)]
    Ke = _bt_db(B, DB, 4)
    for a in range(12):
        for b in range(12):
            Ke[a][b] *= V
    return Ke


# 二次四面体的四个积分点（权重各 1/4，对二次多项式精确）
_T10_A = 0.5854101966249685
_T10_B = 0.1381966011250105
_T10_GP = ((_T10_A, _T10_B, _T10_B, _T10_B),
           (_T10_B, _T10_A, _T10_B, _T10_B),
           (_T10_B, _T10_B, _T10_A, _T10_B),
           (_T10_B, _T10_B, _T10_B, _T10_A))


def _tet10_shape(L):
    """10 个形函数与对 L 的偏导。

    节点序：1-4 顶点，5=(1,2) 6=(2,3) 7=(3,1) 8=(1,4) 9=(2,4) 10=(3,4)。
    与 tetmesh.tet10 生成的中点顺序一致 —— 这两处对不上，装配出来的
    矩阵仍然对称正定，解出来是一个位移场，只是它是另一个问题的解。
    """
    L1, L2, L3, L4 = L
    N = (L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1), L4 * (2 * L4 - 1),
         4 * L1 * L2, 4 * L2 * L3, 4 * L3 * L1, 4 * L1 * L4, 4 * L2 * L4, 4 * L3 * L4)
    d = [[0.0] * 4 for _ in range(10)]
    d[0][0] = 4 * L1 - 1
    d[1][1] = 4 * L2 - 1
    d[2][2] = 4 * L3 - 1
    d[3][3] = 4 * L4 - 1
    d[4][0], d[4][1] = 4 * L2, 4 * L1
    d[5][1], d[5][2] = 4 * L3, 4 * L2
    d[6][2], d[6][0] = 4 * L1, 4 * L3
    d[7][0], d[7][3] = 4 * L4, 4 * L1
    d[8][1], d[8][3] = 4 * L4, 4 * L2
    d[9][2], d[9][3] = 4 * L4, 4 * L3
    return N, d


def tet10_stiffness(coords, D):
    """30x30 二次四面体刚度矩阵。coords 为 10 个点，顺序同上。"""
    gL, V = _tet_grad_L(coords[:4])
    Ke = [[0.0] * 30 for _ in range(30)]
    w = V / 4.0
    for gp in _T10_GP:
        _N, dNdl = _tet10_shape(gp)
        dN = []
        for k in range(10):
            dN.append(tuple(sum(dNdl[k][i] * gL[i][c] for i in range(4))
                            for c in range(3)))
        B = _bmat(dN, 10)
        DB = [[sum(D[r][c] * B[c][b] for c in range(6)) for b in range(30)]
              for r in range(6)]
        Kg = _bt_db(B, DB, 10)
        for a in range(30):
            for b in range(30):
                Ke[a][b] += Kg[a][b] * w
    return Ke


def assemble_tet(mesh, D, order=1):
    """四面体网格装配。order=1 用 TET4，order=2 用 TET10。"""
    if order == 1:
        nodes, tets = mesh["nodes"], mesh["tets"]
        A = sparse.zeros(3 * len(nodes))
        for e in tets:
            Ke = tet4_stiffness([nodes[n] for n in e], D)
            idx = [3 * n + c for n in e for c in range(3)]
            sparse.add_block(A, idx, Ke)
        return A, nodes, tets
    from . import tetmesh
    m10 = tetmesh.tet10(mesh)
    nodes, tets = m10["nodes"], m10["tets"]
    A = sparse.zeros(3 * len(nodes))
    for e in tets:
        Ke = tet10_stiffness([nodes[n] for n in e], D)
        idx = [3 * n + c for n in e for c in range(3)]
        sparse.add_block(A, idx, Ke)
    return A, nodes, tets


def _box_beam(nx, ny, nz, L, b, h):
    """结构化箱体 -> 直接用 box_mesh（六面体路径的参照）。"""
    return box_mesh(nx, ny, nz, L, b, h)


def cantilever_tet(h=4.0, L=40.0, b=8.0, hh=8.0, E=210000.0, nu=0.3, P=100.0,
                   order=2, tol=1e-10):
    """同一个悬臂梁，改用**网格化的四面体**来解。

    这是"能不能解真实零件"的分界线：六面体路径只认长方体，四面体路径
    认的是 `voxel.Shape`。这里的形状恰好是个长方体，所以可以和
    `cantilever`（六面体）对着看 —— 两条完全不同的代码路径必须收敛到
    同一个梁理论值。对不上就说明四面体单元或装配错了，而不是"四面体
    天生差一点"。
    """
    from . import voxel, tetmesh
    shape = voxel.Box(0.0, L, 0.0, b, 0.0, hh)
    m = tetmesh.build(shape, h)
    D = D_isotropic(E, nu)
    K, nodes, tets = assemble_tet(m, D, order=order)
    ndof = 3 * len(nodes)
    f = [0.0] * ndof
    tip = [n for n, p in enumerate(nodes) if p[0] >= L - 1e-9]
    root = [n for n, p in enumerate(nodes) if p[0] <= 1e-9]
    if not tip or not root:
        raise ValueError("网格没覆盖到两端：h 太大")
    per = P / len(tip)
    for n in tip:
        f[3 * n + 2] = -per
    fixed = {}
    for n in root:
        for c in range(3):
            fixed[3 * n + c] = 0.0
    bvec = sparse.apply_dirichlet(K, f, fixed)
    res = sparse.pcg(K, bvec, tol=tol, maxit=40000)
    I = b * hh ** 3 / 12.0
    theory = P * L ** 3 / (3.0 * E * I)
    uz = [-res["x"][3 * n + 2] for n in tip]
    got = sum(uz) / len(uz)
    return {"ratio": got / theory, "tip": got, "theory": theory,
            "order": order, "h": h, "tets": len(tets), "nodes": len(nodes),
            "dof": ndof, "converged": res["converged"],
            "rel_res": res["rel_res"], "pcg_iters": res["iters"],
            "reverted": m["reverted"], "volume_err": None}


def patch_test_tet(h=0.5, E=1000.0, nu=0.3, order=1,
                   strain=(1e-3, -4e-4, 2e-4, 3e-4, 0.0, 0.0)):
    """四面体网格上的分片试验。

    线性位移场必须被**精确**复现。TET4 和 TET10 都能表示线性场，所以
    两种单元都必须过 —— 这一条抓的是 B 矩阵、雅可比、节点顺序，与
    单元阶次无关。过不了就不是精度问题。
    """
    from . import voxel, tetmesh
    shape = voxel.Box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    m = tetmesh.build(shape, h, project=False)
    D = D_isotropic(E, nu)
    K, nodes, tets = assemble_tet(m, D, order=order)
    exx, eyy, ezz, gxy, gyz, gzx = strain
    A = [[exx, gxy / 2.0, gzx / 2.0],
         [gxy / 2.0, eyy, gyz / 2.0],
         [gzx / 2.0, gyz / 2.0, ezz]]

    def u(p):
        return tuple(sum(A[i][j] * p[j] for j in range(3)) for i in range(3))

    f = [0.0] * (3 * len(nodes))
    fixed = {}
    for n, p in enumerate(nodes):
        if (p[0] < 1e-9 or p[0] > 1 - 1e-9 or p[1] < 1e-9
                or p[1] > 1 - 1e-9 or p[2] < 1e-9 or p[2] > 1 - 1e-9):
            for c in range(3):
                fixed[3 * n + c] = u(p)[c]
    bvec = sparse.apply_dirichlet(K, f, fixed)
    res = sparse.pcg(K, bvec, tol=1e-12, maxit=40000)
    err = 0.0
    ref = 0.0
    for n, p in enumerate(nodes):
        want = u(p)
        for c in range(3):
            err = max(err, abs(res["x"][3 * n + c] - want[c]))
            ref = max(ref, abs(want[c]))
    return {"ok": res["converged"] and err <= 1e-9 * max(ref, 1.0),
            "order": order, "max_err": err, "rel_err": err / max(ref, 1e-30),
            "converged": res["converged"], "nodes": len(nodes),
            "pcg_iters": res["iters"]}


def rigid_body_nullity_tet(order=1, E=1000.0, nu=0.3):
    """单个四面体单元的零空间维数，必须是 6。

    TET10 是 30 个自由度，TET4 是 12 个。两者都只能有 6 个刚体模态；
    多一个说明单元有沙漏（本构被积分点漏掉），少一个说明被锁住。
    """
    from . import tetmesh
    p = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    D = D_isotropic(E, nu)
    if order == 1:
        Ke = tet4_stiffness(p, D)
        n = 12
    else:
        mids = [(0.5 * (p[a][i] + p[b][i]) for i in range(3))
                for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
        Ke = tet10_stiffness(p + [tuple(m) for m in mids], D)
        n = 30
    r = _rank(Ke)
    return {"rank": r, "nullity": n - r, "size": n, "order": order}


def tet_vs_hex(h=4.0, L=40.0, b=8.0, hh=8.0, E=210000.0, nu=0.3, P=100.0):
    """四面体网格 vs 六面体网格，同一个梁、同一套材料、同一个载荷。

    两条路径唯一的共同点是梁理论。六面体路径自己在 `cantilever` 里
    已经验过（随细化从下面逼近 1），所以这里的判据是：四面体（TET10）
    与六面体（hex8）在**同一格距**下比值相差不超过 15%，且都小于 1.2
    （大于 1.2 说明解出来的位移比理论大，那通常是边界条件给松了）。
    """
    hexa = cantilever(nx=int(round(L / h)), ny=int(round(b / h)),
                      nz=int(round(hh / h)), L=L, b=b, h=hh, E=E, nu=nu, P=P)
    t10 = cantilever_tet(h=h, L=L, b=b, hh=hh, E=E, nu=nu, P=P, order=2)
    t4 = cantilever_tet(h=h, L=L, b=b, hh=hh, E=E, nu=nu, P=P, order=1)
    return {"hex": hexa["ratio"], "tet10": t10["ratio"], "tet4": t4["ratio"],
            "hex_vs_tet10": abs(hexa["ratio"] - t10["ratio"]),
            "theory": hexa["theory"],
            "converged": hexa["converged"] and t10["converged"] and t4["converged"],
            "tets": t10["tets"], "dof": t10["dof"]}


def rotation_test_tet(order=1, E=1000.0, nu=0.3, omega=(1.0, -0.7, 0.4)):
    """纯刚体转动：u = omega × x 的应变必须为零，单元内力必须为零。

    为什么单独立这一条：分片试验用的是**对称**应变张量，把位移梯度转置
    之后对称部分不变，所以它**发现不了转置**。刚体转动的位移梯度是反对称
    的，转置等于取负 —— 一旦把 J 和 J^T 搞混，这里的应变会变成 2ω 而不是
    零。实测这个坑让四面体梁的刚度高了 40 倍，而分片试验一路绿灯。
    """
    p = [(0.0, 0.0, 0.0), (0.9, 0.1, 0.0), (0.2, 1.1, 0.05), (0.05, 0.15, 0.8)]
    D = D_isotropic(E, nu)
    if order == 1:
        co = p
        Ke = tet4_stiffness(co, D)
        n = 4
    else:
        mids = [tuple(0.5 * (p[a][i] + p[b][i]) for i in range(3))
                for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
        co = p + [tuple(m) for m in mids]
        Ke = tet10_stiffness(co, D)
        n = 10
    wx, wy, wz = omega

    def u(q):
        # omega × x
        return (wy * q[2] - wz * q[1], wz * q[0] - wx * q[2], wx * q[1] - wy * q[0])

    uu = []
    for q in co:
        uu.extend(u(q))
    f = [sum(Ke[a][b] * uu[b] for b in range(3 * n)) for a in range(3 * n)]
    scale = max(abs(v) for v in f) if any(f) else 1.0
    # 用内力相对量级判断：正确时严格为 0（浮点噪声 ~1e-16）
    ref = E * sum(abs(v) for v in uu) / n
    return {"order": order, "max_force": max(abs(v) for v in f),
            "rel": max(abs(v) for v in f) / (ref or 1.0),
            "ok": max(abs(v) for v in f) <= 1e-10 * (ref or 1.0)}
