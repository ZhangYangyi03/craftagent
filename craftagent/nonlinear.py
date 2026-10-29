# -*- coding: utf-8 -*-
"""nonlinear.py -- 几何非线性（共旋梁 + Newton-Raphson）与材料非线性（返回映射）。

为什么线性求解器不能加个循环了事
--------------------------------
线性有限元解的是 K u = f，K 与 u 无关。大变形下这个前提不成立：
结构的刚度取决于它当前的形状（几何非线性），材料的刚度取决于它已经
屈服了多少（材料非线性）。两者都让 K 变成 u 的函数，于是必须迭代。

迭代本身不难，难的是**切线**。切线给错了，Newton 只是收敛慢，不会报错 ——
你会得到一个"收敛"的解，它满足的是你自己写错的那个方程。所以这里
两件事一起做：

一、切线用**中心差分**，不用手推公式。代价是每步多 2n 次内力计算
    （n 是自由度数），好处是不会把切线推错。作为补偿，验收里有一条
    "残差必须二次下降"：切线对，每步残差降到上一步的平方量级；
    切线错，就退化成一阶，这一条立刻红。
二、几何非线性对着**弹性线**（elastica）验：悬臂梁端部加纯力偶，
    精确解是半径 R = EI/M 的圆弧，任意大挠度都成立。线性理论在这里
    差 23%，所以它能区分"真的非线性"和"加了几步循环的线性"。

材料非线性对着**双线性硬化杆**验：单轴拉伸下 σ-ε 曲线必须是两段直线，
拐点在 σ_y，第二段斜率是 E·H/(E+H)。这个有闭式解，不是自己跟自己比。

没有做的
--------
不做弧长法（limit point）、不做接触、不做大应变塑性（对数应变）、
不做三维。这些都不是开关。
"""
from __future__ import annotations

import math

__all__ = ["corotational_beam", "newton", "elastica_end_moment",
           "objectivity", "plastic_bar", "plastic_bar_exact", "quadratic"]


# ---------------------------------------------------------------- 共旋梁单元

def _elem(coords, u, EA, EI, L0):
    """2 节点平面梁（每节点 u, v, theta）在全局坐标下的内力向量。

    共旋：把单元当前的弦方向当作"刚体转角"，扣掉它之后剩下的才是
    局部变形（轴向伸长 + 两端相对转角）。这样大转动不会混进应变里 ——
    这正是线性梁单元在大转角下失效的原因：它把刚体转动当成了弯曲。

    内力向量由虚功推出来，不是猜的：

        dW = N d(eps) + M1 d(theta1_bar) + M2 d(theta2_bar)

    eps 和 theta_bar 对节点位移的偏导就是下面三个 g 向量，
    q = N*g_eps + M1*g1 + M2*g2。推导过程写在注释里，因为符号一旦反了，
    解出来仍然是一条光滑曲线，只是它是另一个问题的解。
    """
    (x1, y1), (x2, y2) = coords
    u1x, u1y, t1 = u[0], u[1], u[2]
    u2x, u2y, t2 = u[3], u[4], u[5]
    dx = (x2 + u2x) - (x1 + u1x)
    dy = (y2 + u2y) - (y1 + u1y)
    Ln = math.hypot(dx, dy)
    if Ln <= 1e-14:
        raise ValueError("单元长度退化")
    c, s = dx / Ln, dy / Ln
    alpha = math.atan2(dy, dx)
    eps = (Ln - L0) / L0
    tb1 = t1 - alpha
    tb2 = t2 - alpha
    N = EA * eps
    M1 = EI * (4.0 * tb1 + 2.0 * tb2) / L0
    M2 = EI * (2.0 * tb1 + 4.0 * tb2) / L0

    # d(eps)/du
    g_eps = (-c / L0, -s / L0, 0.0, c / L0, s / L0, 0.0)
    # d(theta_bar)/du：theta_bar = theta - alpha
    #   d(alpha)/du1 = ( s, -c)/Ln,  d(alpha)/du2 = (-s, c)/Ln
    g1 = (-s / Ln, c / Ln, 1.0, s / Ln, -c / Ln, 0.0)
    g2 = (-s / Ln, c / Ln, 0.0, s / Ln, -c / Ln, 1.0)
    q = [N * g_eps[i] + M1 * g1[i] + M2 * g2[i] for i in range(6)]
    return q, {"N": N, "M1": M1, "M2": M2, "Ln": Ln, "alpha": alpha,
               "eps": eps, "tb1": tb1, "tb2": tb2}


def corotational_beam(n=20, L=100.0, b=10.0, h=10.0, E=210000.0, nu=0.3):
    """生成 n 个共旋梁单元的悬臂梁模型（左端固支）。"""
    A = b * h
    I = b * h ** 3 / 12.0
    nodes = [(L * i / n, 0.0) for i in range(n + 1)]
    elems = [(i, i + 1) for i in range(n)]
    return {"nodes": nodes, "elems": elems, "n": n,
            "EA": E * A, "EI": E * I, "L": L, "A": A, "I": I, "E": E}


def _internal(m, u):
    ndof = 3 * len(m["nodes"])
    f = [0.0] * ndof
    for (a, b) in m["elems"]:
        L0 = m["nodes"][b][0] - m["nodes"][a][0]
        q, _st = _elem((m["nodes"][a], m["nodes"][b]),
                       [u[3 * a + i] for i in range(3)] + [u[3 * b + i] for i in range(3)],
                       m["EA"], m["EI"], L0)
        for i in range(3):
            f[3 * a + i] += q[i]
            f[3 * b + i] += q[3 + i]
    return f


def _tangent_fd(m, u, h=1e-8):
    """中心差分切线。O(n^2) 次内力计算，n 小时完全够用。

    为什么不用解析切线：解析切线要推 dq/du 的 18 个分量，符号错一个，
    Newton 仍然收敛（只是慢），结果仍然是"收敛的解"。差分切线慢一点，
    但它对着的就是内力本身，不会和内力不一致。
    """
    ndof = len(u)
    K = [[0.0] * ndof for _ in range(ndof)]
    for j in range(ndof):
        step = h * max(1.0, abs(u[j]))
        up = list(u)
        um = list(u)
        up[j] += step
        um[j] -= step
        fp = _internal(m, up)
        fm = _internal(m, um)
        for i in range(ndof):
            K[i][j] = (fp[i] - fm[i]) / (2.0 * step)
    return K


def _solve_lin(K, r):
    """稠密高斯消元（带部分主元）。切线是满阵，规模又小，不值得上稀疏。"""
    n = len(r)
    a = [row[:] + [r[i]] for i, row in enumerate(K)]
    for col in range(n):
        p = max(range(col, n), key=lambda i: abs(a[i][col]))
        if abs(a[p][col]) < 1e-300:
            raise ValueError("切线奇异：约束不足或机构")
        a[col], a[p] = a[p], a[col]
        piv = a[col][col]
        for i in range(col + 1, n):
            f = a[i][col] / piv
            if f:
                for j in range(col, n + 1):
                    a[i][j] -= f * a[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / a[i][i]
    return x


def newton(m, f_ext, fixed, u0=None, steps=1, maxit=40, tol=1e-10):
    # tol 是**相对**残差。差分切线的噪声下限大约在 1e-11 相对量级，
    # 要求比它更紧只会让迭代在噪声里空转然后报"未收敛"（实测 n=40 时
    # 残差停在 4.4e-12 相对，而 tol=1e-12 时判为失败）。
    """增量 + Newton-Raphson。fixed 是 {dof: 值}。

    `steps` 是载荷步数。一步到位在大变形下经常不收敛（初始切线离解太远），
    分几步走是标准做法，不是技巧。

    返回的 `history` 是每步每迭代的残差范数 —— 验收靠它判断切线对不对。
    """
    ndof = 3 * len(m["nodes"])
    u = list(u0) if u0 else [0.0] * ndof
    for dof, val in fixed.items():
        u[dof] = val
    free = [i for i in range(ndof) if i not in fixed]
    hist = []
    for step in range(1, steps + 1):
        lam = step / float(steps)
        f = [lam * v for v in f_ext]
        res_hist = []
        fscale = max(1.0, math.sqrt(sum(v * v for v in f)))
        stall = 0
        for it in range(maxit):
            fi = _internal(m, u)
            r = [f[i] - fi[i] for i in free]
            rn = math.sqrt(sum(v * v for v in r))
            res_hist.append(rn)
            if rn <= tol * fscale:
                break
            # 差分切线有噪声下限（约 1e-11 相对）。残差落到那附近就不再
            # 改善，继续迭代只会在噪声里空转。这里如实记下"停滞"并按
            # 已达到的精度接受 —— 但 stagnated 标记必须传出去，不能
            # 假装是正常收敛。
            if it and rn > 0.5 * res_hist[-2]:
                stall += 1
                if stall >= 3 and rn <= 1e-8 * fscale:
                    return {"ok": True, "stagnated": True, "u": u,
                            "history": hist + [res_hist],
                            "res": rn, "rel_res": rn / fscale,
                            "iters": sum(len(h) for h in hist) + len(res_hist)}
            else:
                stall = 0
            K = _tangent_fd(m, u)
            Kff = [[K[i][j] for j in free] for i in free]
            du = _solve_lin(Kff, r)
            for k, i in enumerate(free):
                u[i] += du[k]
            for dof, val in fixed.items():
                u[dof] = val
        else:
            return {"ok": False, "stagnated": False, "reason": "Newton 未收敛",
                    "u": u, "history": hist + [res_hist],
                    "res": res_hist[-1], "rel_res": res_hist[-1] / fscale,
                    "iters": sum(len(h) for h in hist) + len(res_hist)}
        hist.append(res_hist)
    fi = _internal(m, u)
    r = [f[i] - fi[i] for i in free]
    rn = math.sqrt(sum(v * v for v in r))
    return {"ok": True, "stagnated": False, "u": u, "history": hist,
            "res": rn, "rel_res": rn / max(1.0, math.sqrt(sum(v * v for v in f))),
            "iters": sum(len(h) for h in hist)}


# ---------------------------------------------------------------- 弹性线

def elastica_end_moment(n=20, L=100.0, b=10.0, h=10.0, E=210000.0,
                        turn=0.5, steps=6):
    """悬臂梁端部加纯力偶：精确解是圆弧，半径 R = EI/M。

    `turn` = L/R，即梁总共转过的角度除以 2*pi... 更准确地说，L/R 是
    弧对应的圆心角（弧度）。turn=0.5 时是 28.6 度，turn=pi/2 是四分之一圆。

    精确解（任意大变形都成立）：
        R = EI / M
        v_tip = R (1 - cos(L/R))
        u_tip = R sin(L/R)
        phi_tip = L/R

    线性梁理论给 v = M L^2 / (2EI) = L^2/(2R)。取 L/R = pi/2：
    精确 0.6366 L，线性 0.7854 L —— 差 23%。所以这一条能区分
    "真的解了非线性" 和 "把线性解多迭代了几遍"。
    """
    m = corotational_beam(n=n, L=L, b=b, h=h, E=E)
    R = L / turn
    M = E * m["I"] / R
    f = [0.0] * (3 * len(m["nodes"]))
    tip = len(m["nodes"]) - 1
    f[3 * tip + 2] = M
    fixed = {0: 0.0, 1: 0.0, 2: 0.0}
    r = newton(m, f, fixed, steps=steps, tol=1e-10)
    if not r["ok"]:
        return {"ok": False, "reason": r["reason"], "history": r["history"]}
    u = r["u"]
    ut = u[3 * tip]
    vt = u[3 * tip + 1]
    phi = u[3 * tip + 2]
    # 注意 u_tip 是**位移**不是位置：圆弧端点在 (R sin, R(1-cos))，初始在 (L, 0)，
    # 所以水平位移是 R sin(L/R) - L（负值：梁被"拉弯"之后在 x 上缩短了）。
    # 第一版把位置当位移，误差算出来是 157%，而 v 和 phi 都对 —— 只看挠度
    # 的话这个错永远不会暴露。
    ex_u = R * math.sin(L / R) - L
    ex_v = R * (1.0 - math.cos(L / R))
    # 变形后折线长度：纯力偶下梁不可伸长，长度必须还是 L
    length = 0.0
    for (a, b2) in m["elems"]:
        dx = (m["nodes"][b2][0] + u[3 * b2]) - (m["nodes"][a][0] + u[3 * a])
        dy = (m["nodes"][b2][1] + u[3 * b2 + 1]) - (m["nodes"][a][1] + u[3 * a + 1])
        length += math.hypot(dx, dy)
    linear_v = M * L ** 2 / (2.0 * E * m["I"])
    return {"ok": True, "R": R, "M": M, "turn": turn, "n": n,
            "u_tip": ut, "u_exact": ex_u,
            "v_tip": vt, "v_exact": ex_v,
            "v_rel_err": abs(vt - ex_v) / ex_v,
            "u_rel_err": abs(ut - ex_u) / max(abs(ex_u), 1e-30),
            "phi_tip": phi, "phi_exact": L / R,
            "phi_rel_err": abs(phi - L / R) / (L / R),
            "linear_v": linear_v,
            "linear_rel_err": abs(linear_v - ex_v) / ex_v,
            "length": length, "length_err": abs(length - L) / L,
            "rel_res": r["rel_res"], "stagnated": r.get("stagnated", False),
            "history": r["history"]}


def objectivity(n=6, angle=0.7, L=60.0, b=10.0, h=10.0, E=210000.0):
    """客观性：整体刚体转动下内力必须为零。

    这是几何非线性最容易被写错的地方。把刚体转动当成变形，单元就会
    在纯转动下产生巨大的假内力 —— 而解出来的形状看起来"转了，很正常"。
    做法：把每个节点按刚体转动后的位置作为**指定位移**，算内力。
    """
    m = corotational_beam(n=n, L=L, b=b, h=h, E=E)
    ca, sa = math.cos(angle), math.sin(angle)
    u = []
    for (x, y) in m["nodes"]:
        # 绕原点转 angle，位移 = 转后位置 - 原位
        u.extend([ca * x - sa * y - x, sa * x + ca * y - y, angle])
    f = _internal(m, u)
    ref = E * m["A"] * 1e-3
    return {"max_force": max(abs(v) for v in f),
            "rel": max(abs(v) for v in f) / ref,
            "ok": max(abs(v) for v in f) <= 1e-9 * ref}


def quadratic(history, tail=3):
    """从残差历史看收敛阶：p = log(r_{k+1}/r_k) / log(r_k/r_{k-1})。

    切线正确时 p 接近 2；切线错（或差分步长太粗）时 p 掉到 1 附近。
    这一条是"切线写对了"的**唯一**可靠证据 —— 收敛本身不构成证据，
    因为错的切线也会收敛，只是收敛到另一个方程的解。
    """
    out = []
    for h in history:
        hs = [v for v in h if v > 0]
        for k in range(2, len(hs)):
            r0, r1, r2 = hs[k - 2], hs[k - 1], hs[k]
            if r1 <= 0 or r2 <= 0 or r0 <= r1:
                continue
            p = math.log(r2 / r1) / math.log(r1 / r0)
            out.append(p)
    return out


# ---------------------------------------------------------------- 材料非线性

def plastic_bar(E=210000.0, sy=250.0, H=2000.0, n=40, eps_max=0.006):
    """单轴双线性硬化杆：一维返回映射（radial return）。

    这是材料非线性最小的完整例子：有屈服判据、有加载/卸载判断、有
    一致性条件。三维塑性只是把它写成张量，结构是一样的。

    返回映射的意思：先假设这一步全弹性（trial），算出试探应力；
    试探应力超出屈服面就按一致性条件把塑性应变补回去，使应力恰好
    落在**更新后**的屈服面上。不能简单地"按比例缩小" —— 那样卸载
    路径会错，而单调加载看不出来。
    """
    d_eps = eps_max / n
    eps_p = 0.0
    sig_hist = []
    eps_hist = []
    for k in range(1, n + 1):
        eps = k * d_eps
        sig_tr = E * (eps - eps_p)
        fy = sy + H * eps_p
        if abs(sig_tr) <= fy:
            sig = sig_tr
        else:
            # 一致性：sig = sy + H*eps_p_new, sig = E*(eps - eps_p_new)
            # => eps_p_new = (E*eps - sy) / (E + H)
            sgn = 1.0 if sig_tr > 0 else -1.0
            eps_p = (E * eps - sgn * sy) / (E + H)
            sig = E * (eps - eps_p)
        sig_hist.append(sig)
        eps_hist.append(eps)
    return {"eps": eps_hist, "sig": sig_hist, "eps_p": eps_p,
            "E": E, "sy": sy, "H": H}


def plastic_bar_exact(E=210000.0, sy=250.0, H=2000.0, eps_max=0.006):
    """双线性硬化的闭式解：两段直线。

        eps <= sy/E           sigma = E eps
        eps >  sy/E           sigma = sy + Ht (eps - sy/E),  Ht = E H/(E+H)

    第二段的斜率是 Ht 而不是 H —— 因为总应变里还含弹性部分。
    写成 H 是很常见的错，而且它只在屈服后才有偏差，单调加载下看起来
    "也是两段直线"。
    """
    ey = sy / E
    Ht = E * H / (E + H)
    sig = sy + Ht * (eps_max - ey)
    return {"eps_yield": ey, "H_tangent": Ht, "sig_end": sig,
            "sig_end_elastic": E * eps_max}


def plastic_bar_check(E=210000.0, sy=250.0, H=2000.0, n=40, eps_max=0.006):
    """逐点对比返回映射与闭式解。"""
    r = plastic_bar(E, sy, H, n, eps_max)
    ex = plastic_bar_exact(E, sy, H, eps_max)
    ey = ex["eps_yield"]
    Ht = ex["H_tangent"]
    err = 0.0
    for eps, sig in zip(r["eps"], r["sig"]):
        want = E * eps if eps <= ey else sy + Ht * (eps - ey)
        err = max(err, abs(sig - want) / max(abs(want), 1e-30))
    # 屈服前斜率必须是 E，屈服后必须是 Ht
    pre = [i for i, e in enumerate(r["eps"]) if e < ey * 0.9]
    post = [i for i, e in enumerate(r["eps"]) if e > ey * 1.5]
    slope_pre = slope_post = None
    if len(pre) >= 2:
        i, j = pre[0], pre[-1]
        slope_pre = (r["sig"][j] - r["sig"][i]) / (r["eps"][j] - r["eps"][i])
    if len(post) >= 2:
        i, j = post[0], post[-1]
        slope_post = (r["sig"][j] - r["sig"][i]) / (r["eps"][j] - r["eps"][i])
    return {"ok": err < 1e-12, "max_rel_err": err,
            "slope_pre": slope_pre, "slope_post": slope_post,
            "E": E, "H_tangent": Ht, "eps_yield": ey,
            "sig_end": r["sig"][-1], "sig_end_exact": ex["sig_end"],
            "eps_p": r["eps_p"]}
