# -*- coding: utf-8 -*-
"""shell.py -- 四节点平板壳（MITC4）：膜 + 板弯 + 假定剪应变。

为什么不能"用三维实体单元去近似壳"
----------------------------------
薄板弯曲的能量几乎全部来自弯曲，剪应变贡献 O((t/L)^2)。用三维实体单元
（hex8 或 tet）表示弯曲时，位移场必须在厚度方向线性变化，而线性插值在
纯弯下会混进非零剪应变 —— 单元于是偏刚。这就是 fem3d 里实测到的
0.692 / 0.892 / 0.944：细化能收敛，但要细化到"厚度方向放很多层"才有
准确结果，代价是自由度按立方增长。

壳单元换一条路：把厚度方向的运动**解析地**积掉，只留中面的 w 和两个转动。
薄板的问题于是从"三维"降成"二维"，代价是必须额外处理一件事 ——
Reissner-Mindlin 板的剪应变在薄极限下必须趋于零，而位移插值做不到这一点，
单元会**剪切锁死**（t/L → 0 时挠度也趋于零，而不是趋于薄板解）。

MITC4 的做法
------------
剪应变不在积分点上直接用位移算，而是先算四个边中点的剪应变，再线性插值
到积分点。这一改让剪应变场的阶次和弯曲应变匹配，薄极限下剪应变可以自然
趋于零，锁死消失。这不是数值技巧，是**假定应变场**：它改变的是离散空间，
不是积分规则。

验收对着什么
------------
  简支方板   均布载荷中心挠度 w = 0.00406 q a^4 / D（Navier 解）
  固支方板   w = 0.00126 q a^4 / D
  膜拉伸     单轴拉伸的位移必须精确等于 sigma L / E
  刚体零空间 24 个自由度里必须恰好 6 个刚体模态（三平移 + 三转动）
  薄极限      t/a 从 0.1 降到 0.001，归一化挠度不能塌下去 —— 这一条专门
              抓剪切锁死；锁死的单元在这里会掉到接近零

前两条是**外部**参照：它们来自板理论，不是来自这套代码。
"""
from __future__ import annotations

import math

from . import sparse

__all__ = ["element_stiffness", "plate_mesh", "solve_plate",
           "simply_supported", "clamped", "membrane_tension",
           "rigid_modes", "thin_limit"]

_G2 = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
_NODE = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def _shape(xi, eta):
    """4 个形函数与对 (xi, eta) 的偏导。"""
    N = []
    d = []
    for a, b in _NODE:
        N.append(0.25 * (1 + a * xi) * (1 + b * eta))
        d.append((0.25 * a * (1 + b * eta), 0.25 * b * (1 + a * xi)))
    return N, d


def _D_mem(t, E, nu):
    c = E * t / (1.0 - nu * nu)
    return [[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]]


def _D_bend(t, E, nu):
    c = E * t ** 3 / (12.0 * (1.0 - nu * nu))
    return [[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]]


def _D_shear(t, E, nu, k=5.0 / 6.0):
    G = E / (2.0 * (1.0 + nu))
    c = k * G * t
    return [[c, 0.0], [0.0, c]]


def _geom(coords, d):
    """x_xi, x_eta, y_xi, y_eta, 雅可比行列式。"""
    xxi = sum(d[a][0] * coords[a][0] for a in range(4))
    xet = sum(d[a][1] * coords[a][0] for a in range(4))
    yxi = sum(d[a][0] * coords[a][1] for a in range(4))
    yet = sum(d[a][1] * coords[a][1] for a in range(4))
    det = xxi * yet - xet * yxi
    if det <= 0.0:
        raise ValueError("单元雅可比 %.3e 非正：节点顺序反了" % det)
    return xxi, xet, yxi, yet, det


def _cart_dN(d, g):
    """dN/dx, dN/dy（4 个节点）。"""
    xxi, xet, yxi, yet, det = g
    out = []
    for a in range(4):
        dx = (yet * d[a][0] - yxi * d[a][1]) / det
        dy = (-xet * d[a][0] + xxi * d[a][1]) / det
        out.append((dx, dy))
    return out


def _B_mem(dN):
    """3x24 膜应变-位移矩阵。DOF 顺序 6*a + (u,v,w,tx,ty,tz)。"""
    B = [[0.0] * 24 for _ in range(3)]
    for a in range(4):
        dx, dy = dN[a]
        B[0][6 * a] = dx
        B[1][6 * a + 1] = dy
        B[2][6 * a] = dy
        B[2][6 * a + 1] = dx
    return B


def _B_bend(dN):
    """3x24 弯曲应变-位移矩阵。

    中面位移 u = z*beta_x, v = z*beta_y，其中 beta_x = theta_y、beta_y = -theta_x
    （绕 y 转使 u = z*theta_y；绕 x 转使 v = -z*theta_x）。

        kappa_x  = d beta_x / dx = d theta_y / dx
        kappa_y  = d beta_y / dy = -d theta_x / dy
        kappa_xy = d beta_x/dy + d beta_y/dx = d theta_y/dy - d theta_x/dx
    """
    B = [[0.0] * 24 for _ in range(3)]
    for a in range(4):
        dx, dy = dN[a]
        B[0][6 * a + 4] = dx          # d theta_y / dx
        B[1][6 * a + 3] = -dy         # -d theta_x / dy
        B[2][6 * a + 4] = dy          # d theta_y / dy
        B[2][6 * a + 3] = -dx         # -d theta_x / dx
    return B


def _shear_row(coords, xi, eta):
    """在 (xi,eta) 处按位移插值算共变剪应变，返回两条 24 维系数行。

    共变剪应变（把 theta 换成 beta：beta_x = theta_y，beta_y = -theta_x）

        gamma_xi  = dw/dxi  + x_xi *beta_x + y_xi *beta_y
        gamma_eta = dw/deta + x_eta*beta_x + y_eta*beta_y

    返回 (gamma_xi 的行, gamma_eta 的行)。
    """
    N, d = _shape(xi, eta)
    xxi, xet, yxi, yet, _det = _geom(coords, d)
    gxi = [0.0] * 24
    geta = [0.0] * 24
    for a in range(4):
        gxi[6 * a + 2] = d[a][0]          # dw/dxi
        geta[6 * a + 2] = d[a][1]         # dw/deta
        gxi[6 * a + 4] += N[a] * xxi      # x_xi * beta_x,  beta_x = theta_y
        gxi[6 * a + 3] -= N[a] * yxi      # y_xi * beta_y,  beta_y = -theta_x
        geta[6 * a + 4] += N[a] * xet
        geta[6 * a + 3] -= N[a] * yet
    return gxi, geta


def _B_shear(coords, xi, eta):
    """2x24 假定剪应变矩阵（MITC4）。

    只在四个边中点采样：
        gamma_xi  取自 (0,-1) 与 (0,1)
        gamma_eta 取自 (-1,0) 与 (1,0)
    然后线性插值。剪应变场的阶次因此与弯曲应变匹配，薄极限下可以自然
    趋于零 —— 直接在每个积分点用位移算剪应变会锁死。
    """
    gxi_a, _ = _shear_row(coords, 0.0, -1.0)
    gxi_c, _ = _shear_row(coords, 0.0, 1.0)
    _, geta_b = _shear_row(coords, -1.0, 0.0)
    _, geta_d = _shear_row(coords, 1.0, 0.0)
    wA, wC = 0.5 * (1.0 - eta), 0.5 * (1.0 + eta)
    wB, wD = 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)
    gxi = [wA * gxi_a[i] + wC * gxi_c[i] for i in range(24)]
    geta = [wB * geta_b[i] + wD * geta_d[i] for i in range(24)]
    # 共变 -> 笛卡尔： [g_xi; g_eta] = J2 [g_xz; g_yz]
    _N, d = _shape(xi, eta)
    xxi, xet, yxi, yet, det = _geom(coords, d)
    gxz = [0.0] * 24
    gyz = [0.0] * 24
    for i in range(24):
        gxz[i] = (yet * gxi[i] - yxi * geta[i]) / det
        gyz[i] = (-xet * gxi[i] + xxi * geta[i]) / det
    return [gxz, gyz]


def _drill_row(dN, N):
    """钻头自由度：theta_z 与面内刚体转动 omega_z = (dv/dx - du/dy)/2 之差。

    为什么不是简单加一个 k*theta_z^2：那样绕 z 的刚体转动会被罚出能量，
    零空间从 6 掉到 5，单元在整体转动下会产生假应力。罚"差值"则刚体转动
    时 theta_z = omega_z，能量为零。

    这里也踩过一个坑：theta_z 的系数必须写成形函数 N_a(xi,eta)，不是 1.0。
    写成 1.0 相当于只罚了"平均 theta_z"，四个积分点给出同一行，罚项秩只有
    1，剩下的 3 个 theta_z 模态仍然是零能量的 —— 实测零空间 9 而不是 6。
    """
    r = [0.0] * 24
    for a in range(4):
        dx, dy = dN[a]
        r[6 * a + 5] = N[a]        # theta_z 本身要按形函数分布
        r[6 * a + 1] -= 0.5 * dx   # -1/2 dv/dx
        r[6 * a] += 0.5 * dy       # +1/2 du/dy
    return r


def element_stiffness(coords, t, E=210000.0, nu=0.3, k_shear=5.0 / 6.0,
                      drill_scale=1e-4):
    """24x24 平板壳单元刚度矩阵。coords 为 4 个 (x,y)，在单元自身平面内。

    DOF 顺序：每节点 (u, v, w, theta_x, theta_y, theta_z)。
    """
    Dm = _D_mem(t, E, nu)
    Db = _D_bend(t, E, nu)
    Ds = _D_shear(t, E, nu, k_shear)
    K = [[0.0] * 24 for _ in range(24)]
    for xi in _G2:
        for eta in _G2:
            _N, d = _shape(xi, eta)
            g = _geom(coords, d)
            det = g[4]
            dN = _cart_dN(d, g)
            Bm = _B_mem(dN)
            Bb = _B_bend(dN)
            Bs = _B_shear(coords, xi, eta)
            for Bi, Di in ((Bm, Dm), (Bb, Db)):
                DB = [[sum(Di[r][c] * Bi[c][b] for c in range(3)) for b in range(24)]
                      for r in range(3)]
                for a in range(24):
                    ca = [(r, Bi[r][a]) for r in range(3) if Bi[r][a]]
                    if not ca:
                        continue
                    for b in range(24):
                        s = 0.0
                        for r, v in ca:
                            s += v * DB[r][b]
                        if s:
                            K[a][b] += s * det
            DBs = [[Ds[r][0] * Bs[0][b] + Ds[r][1] * Bs[1][b] for b in range(24)]
                   for r in range(2)]
            for a in range(24):
                ca = [(r, Bs[r][a]) for r in range(2) if Bs[r][a]]
                if not ca:
                    continue
                for b in range(24):
                    s = 0.0
                    for r, v in ca:
                        s += v * DBs[r][b]
                    if s:
                        K[a][b] += s * det
    # 钻头罚项：量级取膜刚度的极小份额，只为消掉奇异性，不参与承载
    kd = drill_scale * E * t * 1.0
    for xi in _G2:
        for eta in _G2:
            _N, d = _shape(xi, eta)
            g = _geom(coords, d)
            dN = _cart_dN(d, g)
            r = _drill_row(dN, _N)
            nz = [(i, v) for i, v in enumerate(r) if v]
            if len(nz) < 2:
                continue
            for i, vi in nz:
                for j, vj in nz:
                    K[i][j] += kd * vi * vj * g[4]
    return K


# ---------------------------------------------------------------- 网格与求解

def plate_mesh(n, a, b=None):
    """a x b 的方板/矩板，n x n 个 Q4 单元。节点编号 i + j*(n+1)。"""
    b = a if b is None else b
    nodes = []
    for j in range(n + 1):
        for i in range(n + 1):
            nodes.append((a * i / n, b * j / n))
    sx = n + 1
    elems = []
    for j in range(n):
        for i in range(n):
            n0 = i + j * sx
            elems.append((n0, n0 + 1, n0 + 1 + sx, n0 + sx))
    return {"nodes": nodes, "elems": elems, "n": n, "a": a, "b": b,
            "h": (a / n, b / n)}


def assemble(mesh, t, E=210000.0, nu=0.3):
    nodes, elems = mesh["nodes"], mesh["elems"]
    A = sparse.zeros(6 * len(nodes))
    for e in elems:
        Ke = element_stiffness([nodes[n] for n in e], t, E, nu)
        idx = [6 * n + c for n in e for c in range(6)]
        sparse.add_block(A, idx, Ke)
    return A


def solve_plate(mesh, t, f_load, fixed, E=210000.0, nu=0.3, tol=1e-11):
    """f_load: 每节点面内 6 个分量的载荷；fixed: {dof: value}。"""
    K = assemble(mesh, t, E, nu)
    ndof = 6 * len(mesh["nodes"])
    f = list(f_load) + [0.0] * (ndof - len(f_load))
    b = sparse.apply_dirichlet(K, f, fixed)
    res = sparse.pcg(K, b, tol=tol, maxit=60000)
    return {"K": K, "x": res["x"], "converged": res["converged"],
            "rel_res": res["rel_res"], "iters": res["iters"], "dof": ndof}


def _D_flex(t, E, nu):
    return E * t ** 3 / (12.0 * (1.0 - nu * nu))


def _uniform(mesh, q):
    """均布横向载荷 -> 每节点 w 分量（按单元面积分配到节点）。"""
    nodes, elems, n = mesh["nodes"], mesh["elems"], mesh["n"]
    f = [0.0] * (6 * len(nodes))
    a, b = mesh["a"], mesh["b"]
    area = (a / n) * (b / n)
    for e in elems:
        for nd in e:
            f[6 * nd + 2] += -q * area / 4.0
    return f


def simply_supported(n=8, a=100.0, t=1.0, q=1.0, E=210000.0, nu=0.3):
    """四边简支方板，均布载荷 q。参照 w = 0.00406 q a^4 / D（Navier 解）。

    边界条件：周边 w = 0，且沿边的**切向转动**为 0（硬简支）。板理论里
    这个系数是 0.00406；硬简支与软简支的差别在薄板下小于 0.1%。
    """
    mesh = plate_mesh(n, a)
    sx = n + 1
    fixed = {}
    for j in range(sx):
        for i in range(sx):
            nd = i + j * sx
            if i in (0, n) or j in (0, n):
                fixed[6 * nd + 2] = 0.0            # w = 0
            if i == 0:
                fixed[6 * nd] = 0.0                # 只固定面内刚体平移
            if j == 0:
                fixed[6 * nd + 1] = 0.0
    f = _uniform(mesh, q)
    r = solve_plate(mesh, t, f, fixed, E, nu)
    D = _D_flex(t, E, nu)
    theory = 0.00406 * q * a ** 4 / D
    c = (n // 2) + (n // 2) * sx
    got = -r["x"][6 * c + 2]
    return {"ratio": got / theory, "w": got, "theory": theory, "D": D,
            "converged": r["converged"], "iters": r["iters"], "dof": r["dof"],
            "t_over_a": t / a}


def clamped(n=8, a=100.0, t=1.0, q=1.0, E=210000.0, nu=0.3):
    """四边固支方板。参照 w = 0.00126 q a^4 / D。"""
    mesh = plate_mesh(n, a)
    sx = n + 1
    fixed = {}
    for j in range(sx):
        for i in range(sx):
            nd = i + j * sx
            if i in (0, n) or j in (0, n):
                for c in (2, 3, 4):
                    fixed[6 * nd + c] = 0.0
    f = _uniform(mesh, q)
    r = solve_plate(mesh, t, f, fixed, E, nu)
    D = _D_flex(t, E, nu)
    theory = 0.00126 * q * a ** 4 / D
    c = (n // 2) + (n // 2) * sx
    got = -r["x"][6 * c + 2]
    return {"ratio": got / theory, "w": got, "theory": theory,
            "converged": r["converged"], "iters": r["iters"], "dof": r["dof"],
            "t_over_a": t / a}


def membrane_tension(n=4, L=100.0, t=2.0, sigma=10.0, E=210000.0, nu=0.3):
    """单轴拉伸：位移必须精确等于 sigma L / E（均匀应变，与网格无关）。

    膜部分用的是双线性 Q4，常应变场能精确表示，所以这不是"接近"，
    是"必须一位不差"。横向收缩 -nu sigma L / E 一起查。
    """
    mesh = plate_mesh(n, L)
    sx = n + 1
    fixed = {}
    for j in range(sx):
        for i in range(sx):
            nd = i + j * sx
            if i == 0:
                fixed[6 * nd] = 0.0
    # 横向收缩必须自由：v 只在一个节点上定住以消掉 y 向刚体平移。
    # 整条边都定住 v 会让泊松收缩变成零，实测横向位移少一半。
    fixed[1] = 0.0
    f = [0.0] * (6 * len(mesh["nodes"]))
    # 在 x = L 边上加均布拉应力
    seg = sigma * t * (L / n)
    for j in range(sx):
        nd = n + j * sx
        f[6 * nd] += seg * (0.5 if j in (0, n) else 1.0)
    r = solve_plate(mesh, t, f, fixed, E, nu)
    # 单轴应力：u = sigma*x/E（与 y 无关），v = -nu*sigma*y/E（与 x 无关）。
    # 取样点在 (L, L/2)，所以 v 的参照是 -nu*sigma*(L/2)/E —— 不是 -nu*sigma*L/E。
    tip = n + (n // 2) * sx
    px, py = mesh["nodes"][tip]
    want_u = sigma * px / E
    want_v = -nu * sigma * py / E
    u = r["x"][6 * tip]
    v = r["x"][6 * tip + 1]
    return {"u": u, "u_exact": want_u, "u_rel": abs(u - want_u) / abs(want_u),
            "v": v, "v_exact": want_v, "v_rel": abs(v - want_v) / abs(want_v),
            "converged": r["converged"]}


def rigid_modes(t=1.0, E=1000.0, nu=0.3):
    """单个壳单元的零空间维数，必须是 6。

    平面壳的 24 个自由度里，刚体运动恰好 6 个：三平移 + 面内转动 + 两个
    面外转动。多一个说明钻头自由度没约束住（矩阵奇异），少一个说明
    某个刚体模态被罚出了能量 —— 后者在整体转动时会凭空产生应力。
    """
    coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    K = element_stiffness(coords, t, E, nu)
    from .fem3d import _rank
    r = _rank(K, tol=1e-7)
    return {"rank": r, "nullity": 24 - r, "size": 24}


def thin_limit(n=6, a=100.0, q=1.0, E=210000.0, nu=0.3):
    """薄极限：t/a 从 0.1 降到 0.001，归一化挠度不能塌。

    锁死的单元在 t/a 变小时挠度会跟着掉（因为剪应变项压倒了弯曲项）。
    这里返回每一档的比值，调用方看的是"稳不稳定"，不是"等不等于 1"。
    """
    out = []
    for t in (a * 0.1, a * 0.01, a * 0.001):
        r = simply_supported(n=n, a=a, t=t, q=q, E=E, nu=nu)
        out.append({"t_over_a": t / a, "ratio": r["ratio"],
                    "converged": r["converged"]})
    return out
