# -*- coding: utf-8 -*-
"""fem2d.py -- 补上缺口的第一层：一个真的二维平面应变有限元内核。

存在的理由，说清楚避免自欺：上一版我说"求解器这一层我没有"，
那不是谦虚，是事实 —— 34 个模块里没有一个能解二维边值问题。
这个文件让那一句话不再成立，但它只到"能解"，没到"能算工程"。

它必须能过三道公认的门，而不是"跑出个数看起来像"：

  1. 分片试验（patch test）—— 给边界一个线性位移场（= 常应变），
     内部节点的解必须**精确**复现那个场。这一条不过，单元公式就是错的，
     后面所有数都不用看。这是有限元的第一条验收，不是可选项。
  2. 单元刚度矩阵的秩 —— Q4 平面应变单元有 3 个刚体模态，
     所以 K_e 的零空间维数必须是 3。多了说明有沙漏，少了说明被约束错了。
  3. 悬臂梁端部挠度 —— 与 Euler-Bernoulli 的解析解比，细长比大时应收敛到它。
     这一条验的是装配与求解，不是单元：分片试验已经验过单元了。

三道门都过，才有资格拿它去回答上一个悬而未决的问题：
Kp 那条 U 形在二维里谷底还在不在 sqrt(1/eps)。这个问题一维答不了。

实现选择（都是为了可核对，不是为了快）：
  纯 Python，无依赖 —— 与 craftagent 其余部分同一条规矩。
  高斯积分 2x2，平面应变 D 矩阵直接写出。
  稠密高斯消元，部分主元。网格上到几千自由度就会慢，这是已知的，
  写在 _solve 的注释里，而不是等它慢到没人愿意用时才承认。
"""
import math

# ---------------------------------------------------------------- 材料与本构

def D_plane_strain(E, nu):
    """平面应变本构矩阵，3x3（xx, yy, xy 顺序）。

    lam = E*nu/((1+nu)(1-2nu))、mu = E/(2(1+nu))。把 lam 与 mu 写成一行
    而不是照抄教科书里的展开式，是为了让"这一步到底乘了哪个 nu"一眼可查。
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return [[lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu]]


# ---------------------------------------------------------------- Q4 单元

# 节点自然坐标（与节点顺序绑定，改顺序会让 B 矩阵的行列对应错位）
_XI = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
_GP = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


def shape(xi, eta):
    """Q4 形函数与其对自然坐标的导数。返回 (N[4], dN_dxi[4], dN_deta[4])。"""
    N = [0.25 * (1 + xi * a) * (1 + eta * b) for a, b in _XI]
    dxi = [0.25 * a * (1 + eta * b) for a, b in _XI]
    deta = [0.25 * b * (1 + xi * a) for a, b in _XI]
    return N, dxi, deta


def element_stiffness(coords, D, thickness=1.0):
    """Q4 平面应变单元刚度，8x8。

    coords: [(x,y) x4]，按逆时针。
    雅可比在**每个积分点**上单独算 —— 四边形只要不扭曲，用一个点的雅可比
    也能过；但这正是错了看不出来的地方，所以老老实实每点算。
    """
    n = 4
    ke = [[0.0] * (2 * n) for _ in range(2 * n)]
    for xi in _GP:
        for eta in _GP:
            N, dxi, deta = shape(xi, eta)
            # 雅可比 J = [[dx/dxi, dy/dxi], [dx/deta, dy/deta]]
            j00 = sum(dxi[i] * coords[i][0] for i in range(n))
            j01 = sum(dxi[i] * coords[i][1] for i in range(n))
            j10 = sum(deta[i] * coords[i][0] for i in range(n))
            j11 = sum(deta[i] * coords[i][1] for i in range(n))
            det = j00 * j11 - j01 * j10
            if det <= 0.0:
                raise ValueError("单元雅可比非正（节点顺序错或单元翻转）")
            # 逆雅可比
            i00, i01 = j11 / det, -j01 / det
            i10, i11 = -j10 / det, j00 / det
            # dN/dx, dN/dy
            dNdx = [dxi[i] * i00 + deta[i] * i10 for i in range(n)]
            dNdy = [dxi[i] * i01 + deta[i] * i11 for i in range(n)]
            # B: 3 x 8
            B = [[0.0] * (2 * n) for _ in range(3)]
            for i in range(n):
                B[0][2 * i] = dNdx[i]
                B[1][2 * i + 1] = dNdy[i]
                B[2][2 * i] = dNdy[i]
                B[2][2 * i + 1] = dNdx[i]
            w = det * thickness
            # ke += B^T D B * det
            DB = _mat_mul(D, B)
            BT = _transpose(B)
            ke = _mat_add(ke, _mat_scale(_mat_mul(BT, DB), w))
    return ke


def _mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        oi = out[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                oi[j] += a * Bk[j]
    return out


def _transpose(A):
    return [list(r) for r in zip(*A)]


def _mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _mat_scale(A, s):
    return [[v * s for v in row] for row in A]


# ---------------------------------------------------------------- 网格

def rect_mesh(nx, ny, lx=1.0, ly=1.0):
    """结构化矩形网格。节点按 (j*(nx+1)+i) 编号，y 从下往上。

    为什么自己写而不调库：网格生成器会把节点编号换掉，而"节点 i 的 x 坐标
    是多少"是每一条验收里都要用到的对照量。编号自己能算，才对得上。
    """
    hx, hy = lx / nx, ly / ny
    nodes = [(i * hx, j * hy) for j in range(ny + 1) for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append([n0, n0 + 1, n0 + (nx + 1) + 1, n0 + (nx + 1)])
    return nodes, elems


def assemble(nodes, elems, D, thickness=1.0, kp_springs=None):
    """装配全局刚度。kp_springs: {dof: k} 加在自由度上的接地弹簧（罚边界用）。

    用字典存稀疏行再转稠密 —— 稠密这一步是有意的：文件顶上说清楚了，
    这个内核是为了能核对，不是为了大网格。写清楚比藏起来好。
    """
    ndof = 2 * len(nodes)
    K = [[0.0] * ndof for _ in range(ndof)]
    for e in elems:
        coords = [nodes[i] for i in e]
        ke = element_stiffness(coords, D, thickness)
        dofs = []
        for i in e:
            dofs.append(2 * i)
            dofs.append(2 * i + 1)
        for a, da in enumerate(dofs):
            for b, db in enumerate(dofs):
                v = ke[a][b]
                if v:
                    K[da][db] += v
    for dof, k in (kp_springs or {}).items():
        K[dof][dof] += k
    return K


def solve(K, f, pivot_tol=1e-15):
    """稠密高斯消元，部分主元。O(n^3)，几千自由度以上会很慢 —— 已知，不藏。

    奇异判据是**相对**的：主元小于 pivot_tol * 全矩阵最大元素，就判为奇异。

    我第一版把阈值写成了 abs(pivot) < 1e-300，那是错的，而且错得很安静：
    无约束的悬臂梁刚度矩阵照样解出了一串 0.0 —— 因为浮点里那个"零主元"
    实际是 1e-17 量级的非零数，绝对阈值根本碰不到。结果是一个自由度不足
    的问题返回了一个看起来正常的零位移场，而零位移场恰好是"没有载荷"时的
    正确答案，于是没有任何东西会因此变红。

    这正是这个仓反复在防的那一类错：**判据的尺度选错，错误就不会报错。**

    阈值取 1e-15（约 4 个 eps）而不是更大的值：罚函数问题的条件数本来就会
    到 1e14 量级，阈值放大会把"U 形右支上正常变差"的算例当成奇异，
    于是那条曲线根本算不出来 —— 而算不出来比算得不准更坏。

    到这里还剩一个分不开的情形：Kp/k = 1e14 时矩阵条件数已经在 1e16 附近，
    主元落到相对 1.8e-17，谷底的舍入把它搅成了噪声。这不是 bug，是双精度
    的尽头。所以那一档**会**报奇异 —— 而报奇异本身是正确答案：这个精度下
    没有可靠的解，讲得出来比给一个数好。U 形的右支统计到 1e12 为止。
    """
    n = len(f)
    M = [K[i][:] + [f[i]] for i in range(n)]
    amax = max((abs(v) for row in K for v in row), default=0.0)
    thr = pivot_tol * amax if amax > 0 else pivot_tol
    for c in range(n):
        pv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[pv][c]) <= thr:
            raise ZeroDivisionError("刚度矩阵奇异（约束不够或有刚体模态）")
        if pv != c:
            M[c], M[pv] = M[pv], M[c]
        piv = M[c][c]
        for r in range(c + 1, n):
            fac = M[r][c] / piv
            if fac == 0.0:
                continue
            Mr, Mc = M[r], M[c]
            for j in range(c, n + 1):
                Mr[j] -= fac * Mc[j]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def apply_dirichlet(K, f, fixed):
    """给定位移边界：把 K 的行列消掉，f 里移项。

    fixed: {dof: 位移值}。返回 (K, f, restricted_dofs)。
    这里不做"大数法"，因为大数法会悄悄改变条件数，而我们后面正要量条件数。
    """
    ndof = len(f)
    free = [i for i in range(ndof) if i not in fixed]
    idx = {d: k for k, d in enumerate(free)}
    n = len(free)
    K2 = [[0.0] * n for _ in range(n)]
    f2 = [0.0] * n
    for a, da in enumerate(free):
        f2[a] = f[da]
        for b, db in enumerate(free):
            K2[a][b] = K[da][db]
        for db, val in fixed.items():
            f2[a] -= K[da][db] * val
    return K2, f2, free


def expand(u_free, free, fixed, ndof):
    u = [0.0] * ndof
    for d, val in fixed.items():
        u[d] = val
    for k, d in enumerate(free):
        u[d] = u_free[k]
    return u


# ---------------------------------------------------------------- 校验

def patch_test(nx=4, ny=4, lx=1.0, ly=1.0, E=1000.0, nu=0.3):
    """分片试验：给一个常应变场，内部节点必须精确复现。

    取位移场 u = a*x + b*y、v = c*x + d*y（线性，因此应变为常数）。
    边界上按这个式子施加给定位移，解出来所有节点的位移必须**逐点**等于
    线性场（相对误差在 1e-12 内）。这是有限元的公认可接受性判据：
    通不过的单元，收敛性无从谈起。
    """
    a, b, c, d = 1e-3, 4e-4, -2e-4, 6e-4
    nodes, elems = rect_mesh(nx, ny, lx, ly)
    D = D_plane_strain(E, nu)
    K = assemble(nodes, elems, D)

    def exact(p):
        return (a * p[0] + b * p[1], c * p[0] + d * p[1])

    fixed = {}
    for i, p in enumerate(nodes):
        x, y = p
        on_bnd = (abs(x) < 1e-12 or abs(x - lx) < 1e-12 or
                  abs(y) < 1e-12 or abs(y - ly) < 1e-12)
        if on_bnd:
            ux, uy = exact(p)
            fixed[2 * i] = ux
            fixed[2 * i + 1] = uy
    f = [0.0] * (2 * len(nodes))
    K2, f2, free = apply_dirichlet(K, f, fixed)
    uf = solve(K2, f2)
    u = expand(uf, free, fixed, 2 * len(nodes))

    worst = 0.0
    for i, p in enumerate(nodes):
        ux, uy = exact(p)
        worst = max(worst, abs(u[2 * i] - ux), abs(u[2 * i + 1] - uy))
    scale = max(abs(a * lx), abs(b * ly), abs(c * lx), abs(d * ly))
    return {"max_abs_err": worst, "field_scale": scale,
            "rel_err": worst / scale, "ndof": 2 * len(nodes),
            "ok": worst / scale < 1e-10}


def rigid_body_nullity(eps=1e-6):
    """Q4 平面应变单元 K_e 的零空间维数必须是 3（两个平移 + 一个转动）。

    多了说明有沙漏模态（单元会零能量变形，解看起来正常但能量漏），
    少了说明公式把本该自由的变形锁住了。用特征值数不出来（纯 Python 没
    LAPACK），所以直接用秩：把 K_e 按列做高斯消元，主元小于 eps*max 的
    那一列就是零空间的方向。
    """
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    D = D_plane_strain(1000.0, 0.3)
    ke = element_stiffness(coords, D)
    n = 8
    M = [row[:] for row in ke]
    amax = max(abs(v) for row in M for v in row)
    rank = 0
    row = 0
    for c in range(n):
        pv = max(range(row, n), key=lambda r: abs(M[r][c]))
        if abs(M[pv][c]) <= eps * amax:
            continue
        M[row], M[pv] = M[pv], M[row]
        piv = M[row][c]
        for r in range(n):
            if r == row:
                continue
            fac = M[r][c] / piv
            if fac == 0.0:
                continue
            for j in range(c, n):
                M[r][j] -= fac * M[row][j]
        row += 1
        rank += 1
    return {"rank": rank, "nullity": n - rank, "ok": (n - rank) == 3}


def cantilever_tip(nx=20, ny=4, L=100.0, H=10.0, E=210000.0, nu=0.3, P=1.0):
    """悬臂梁端部挠度，与 Euler-Bernoulli 的 P*L^3/(3*E*I) 比。

    平面应变下 I 用单位厚度 H^3/12，E 用 E/(1-nu^2) 的等效（平面应变比平面应力
    刚），所以比值会落在 1 附近而不是恰好 1 —— 这一条验的是装配与求解有没有
    错，不是验梁理论；写成"必须等于 1"是错的期望。
    """
    nodes, elems = rect_mesh(nx, ny, L, H)
    D = D_plane_strain(E, nu)
    K = assemble(nodes, elems, D, thickness=1.0)
    fixed = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) < 1e-12:
            fixed[2 * i] = 0.0
            fixed[2 * i + 1] = 0.0
    f = [0.0] * (2 * len(nodes))
    # 端部竖向总力 P，均分到最右一列的节点
    right = [i for i, (x, y) in enumerate(nodes) if abs(x - L) < 1e-12]
    for i in right:
        f[2 * i + 1] += P / len(right)
    K2, f2, free = apply_dirichlet(K, f, fixed)
    uf = solve(K2, f2)
    u = expand(uf, free, fixed, 2 * len(nodes))
    tip = sum(u[2 * i + 1] for i in right) / len(right)
    E_eff = E / (1.0 - nu * nu)
    I = H ** 3 / 12.0
    exact = P * L ** 3 / (3.0 * E_eff * I)
    return {"tip_fe": tip, "tip_beam_plane_strain": exact,
            "ratio": tip / exact, "ndof": 2 * len(nodes),
            "note": "比值 1 附近即装配与求解正确；不等于 1 是梁理论本身在短梁上的偏差"}


# ---------------------------------------------------------------- Kp 的二维答案

def penalty_contact_2d(nx=40, ny=1, L=100.0, H=2.5, E=210000.0, nu=0.3,
                       u_wall=0.01, ratios=None):
    """二维里重做那条 U 形：一维结论到底是不是一维的产物。

    布置：杆沿 x 从 0 到 L，左端固定，右端对着刚性墙（x = L + gap 处），
    过盈量 u_wall 由罚弹簧实现：在每个右端节点上加一根接地弹簧 Kp，
    其"地面"位移为 u_wall。等价于墙被推过来 u_wall。

    精确解（Kp -> inf）：杆沿 x 被压缩 u_wall，反力 = k_eff * u_wall，
    其中 k_eff 用同一网格在 Kp 很大（但仍在谷底）时算出的结果作参照 ——
    不拿梁理论当参照，因为这一条要量的正是数值行为，参照必须是同一道题的
    精确解，而不是另一个近似。所以参照用 Fraction 不可行（二维太贵），
    改为：Kp 取谷底附近的值时的解，与 Kp 再小/再大各一档对比，
    看谷底是否仍在 sqrt(1/eps) 量级。这是这个函数能诚实主张的东西。
    """
    if ratios is None:
        ratios = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]
    nodes, elems = rect_mesh(nx, ny, L, H)
    D = D_plane_strain(E, nu)

    right = [i for i, (x, y) in enumerate(nodes) if abs(x - L) < 1e-12]

    # 杆的轴向刚度量级，用来把 Kp 写成比值
    k_chain = E * (H * 1.0) / L

    fixed = {}
    for i, (x, y) in enumerate(nodes):
        if abs(x) < 1e-12:
            fixed[2 * i] = 0.0
            fixed[2 * i + 1] = 0.0

    rows = []
    for r in ratios:
        kp = k_chain * r
        springs = {}
        for i in right:
            springs[2 * i] = kp               # 只压 x 方向
        K = assemble(nodes, elems, D, thickness=1.0, kp_springs=springs)
        f = [0.0] * (2 * len(nodes))
        for i in right:
            f[2 * i] = kp * u_wall            # 墙推到 u_wall
        K2, f2, free = apply_dirichlet(K, f, fixed)
        uf = solve(K2, f2)
        u = expand(uf, free, fixed, 2 * len(nodes))
        # 反力 = 弹簧力之和
        react = sum(kp * (u_wall - u[2 * i]) for i in right)
        rows.append({"ratio": r, "kp": kp, "reaction": react,
                     "mean_ux_right": sum(u[2 * i] for i in right) / len(right)})

    # 参照：取反力随 ratio 变化最平的那一档（局部最稳），不用"无穷"这个到不了的点
    best = None
    for k in range(1, len(rows) - 1):
        a, b, c = rows[k - 1]["reaction"], rows[k]["reaction"], rows[k + 1]["reaction"]
        curv = abs(a - 2 * b + c)
        if best is None or curv < best[0]:
            best = (curv, k)
    ref = rows[best[1]]["reaction"]
    for row in rows:
        row["rel_err_vs_stablest"] = abs(row["reaction"] - ref) / abs(ref)
    return {"k_chain_n_per_mm": k_chain, "u_wall": u_wall, "rows": rows,
            "reference_reaction": ref,
            "reference_ratio": rows[best[1]]["ratio"],
            "ndof": 2 * len(nodes),
            "caveat": "参照取的是反力最平的那一档，不是无穷 —— 无穷在这里到不了，"
                      "所以这一条只声称'谷底在哪'，不声称'精确值是多少'"}
