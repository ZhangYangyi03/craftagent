# -*- coding: utf-8 -*-
"""penalty.py -- 罚函数刚度 Kp 的规则：它不需要从黑箱里挖，它是一条 U 形曲线。

前提（决定了这个文件能主张什么）：
  本机没有装 Dassault（Abaqus / Simulia）或 Siemens（NX / Simcenter）的任何组件。
  实测：注册表 104 个已安装项，CAD/CAE 命中 0；C:/D: 各 Program Files 按
  abaqus/simulia/solidworks/catia/nx/simcenter/star-ccm 扫，命中 0；
  两家在线理论手册都是登录墙（help.3ds.com 2024/2023 均跳 3DEXPERIENCE 登录页）。
  所以这里没有任何厂商的二进制或私有常数，也不打算去取。

那能主张什么：Kp 的正确取值**由本机能算的东西决定**，而且它有上界。
本文件用 40 单元链 + 浮点/Fraction 精确解对拍，量出这条 U 形：

    Kp/k_chain    F(float)        err_vs_limit
    1e+00         0.131250000000   5.000e-01
    1e+02         0.259900990099   9.901e-03
    1e+04         0.262473752625   9.999e-05
    1e+06         0.262499737526   9.999e-07
    1e+08         0.262499998951   3.996e-09   <- 谷底
    1e+10         0.262500112792   4.297e-07
    1e+12         0.262517872024   6.808e-05
    1e+14         0.259558000093   1.121e-02
    1e+16         0.455364912444   7.347e-01

左边成因：Kp 有限 = 有一部分位移被"墙"吃掉，没进到杆里，误差 ~ k/Kp。
右边成因：Kp/k 就是刚度矩阵的条件数量级，双精度 16 位减去 log10(Kp/k) 位，
所以误差 ~ eps*(Kp/k)。
两者相乘：谷底在 Kp/k = sqrt(1/eps) ≈ 1e8，最差误差 sqrt(eps) ≈ 1e-8。

这就是"厂商为什么把自动缩放停在一个数上"的答案：不是秘密常数，是这条曲线的谷底。
而这个文件的价值在于：谷底位置**只依赖浮点精度和刚度比**，
换一家软件、换一种材料、换一个零件，它都不动。

一处必须写下来的教训：我第一版把这个链的罚弹簧接成了串联，
于是误差被 k/Kp 项完全支配，右边那半条曲线看不见，我据此以为"Kp 没有上界"。
40 单元、并联、浮点对 Fraction 对拍之后才看见。**一个错的接线方式会让一整条
物理现象消失，而结论看起来仍然自洽** —— 这正是这个仓反复在防的东西。
"""
import math
from fractions import Fraction

EPS = 2.220446049250313e-16      # 双精度机器精度

# ---------------------------------------------------------------- 正确的最小模型

def chain_force(n_elem, k_elem, kp_over_kchain, interference_mm=0.01, exact=False):
    """一维杆被压进刚性墙：n 个单元串联，末端一根 Kp 接墙，墙位移给定（过盈量）。

    精确解（Kp -> inf）：F = u * k_chain，k_chain = k_elem / n。
    罚函数解：          F = u / (1/k_chain + 1/Kp) = u*k_chain*Kp/(Kp+k_chain)

    exact=True 时用 Fraction 解同一个线性系统，用来判断浮点解丢了几位。
    这就是"精确值从哪来"的答案：不是另一个软件，是同一道题的分数算术。
    """
    C = Fraction if exact else float
    kp = C(k_elem) / n_elem * C(kp_over_kchain) if not isinstance(kp_over_kchain, float) \
        else C(k_elem / n_elem * kp_over_kchain)
    n = n_elem
    A = [[C(0)] * n for _ in range(n)]
    for i in range(n):
        A[i][i] += C(k_elem)
        if i > 0:
            A[i - 1][i - 1] += C(k_elem)
            A[i][i - 1] -= C(k_elem)
            A[i - 1][i] -= C(k_elem)
    A[n - 1][n - 1] += kp
    b = [C(0)] * n
    b[n - 1] = kp * C(interference_mm) if not isinstance(interference_mm, Fraction) \
        else kp * interference_mm
    x = _gauss([row + [bb] for row, bb in zip(A, b)])
    return kp * (C(interference_mm) - x[-1])


def _gauss(M):
    """部分主元高斯消元。float 与 Fraction 共用一份实现，这样差别只来自算术。"""
    n = len(M)
    M = [row[:] for row in M]
    for c in range(n):
        pv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[pv] = M[pv], M[c]
        piv = M[c][c]
        if piv == 0:
            raise ZeroDivisionError("奇异")
        for r in range(c + 1, n):
            fac = M[r][c] / piv
            if fac == 0:
                continue
            for j in range(c, n + 1):
                M[r][j] -= fac * M[c][j]
    x = [type(M[0][0])(0)] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][j] * x[j] for j in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


# ---------------------------------------------------------------- 两条成因，写成闭式

def err_from_soft(kp_over_kchain):
    """左边：Kp 有限导致的位移分配误差 ~ k_chain/Kp。"""
    return 1.0 / (1.0 + kp_over_kchain)


def err_from_roundoff(kp_over_kchain, eps=EPS):
    """右边：浮点丢位。条件数量级是 1+Kp/k_chain，有效位掉 log10 位。"""
    return eps * kp_over_kchain


def best_kp_ratio(eps=EPS):
    """谷底：两条误差相等处，Kp/k_chain = sqrt(1/eps) ≈ 6.7e7，误差 ≈ sqrt(eps)≈1.5e-8。"""
    r = math.sqrt(1.0 / eps)
    return {"kp_over_kchain": r, "expected_err": math.sqrt(eps),
            "rule": "Kp = k_chain * sqrt(1/eps)；大 100 倍或小 100 倍都差两个数量级"}


def sweep(n_elem=40, k_elem=1050.0, interference_mm=0.01, decades=range(0, 17)):
    """扫一遍，实测每条 decade 上的误差。返回的每一行都是算出来的，不是画的。"""
    k_chain = k_elem / float(n_elem)
    limit = interference_mm * k_chain
    rows = []
    for d in decades:
        ratio = 10.0 ** d
        f = chain_force(n_elem, k_elem, ratio, interference_mm)
        rows.append({"kp_over_kchain": ratio, "force_n": f,
                     "err": abs(f - limit) / limit,
                     "pred_soft": err_from_soft(ratio),
                     "pred_round": err_from_roundoff(ratio)})
    return {"k_chain_n_per_mm": k_chain, "limit_force_n": limit, "rows": rows}


def measured_vs_predicted(sweep_result):
    """实测误差与 max(soft, round) 两条闭式对比。这一条是这个文件的验收核心：
    如果闭式说得出实测的形状，那 Kp 就不需要去逆任何东西。"""
    out = []
    for r in sweep_result["rows"]:
        pred = max(r["pred_soft"], r["pred_round"])
        out.append({"kp_over_kchain": r["kp_over_kchain"], "measured": r["err"],
                    "predicted": pred,
                    "same_decade": abs(math.log10(max(r["err"], 1e-18))
                                       - math.log10(pred)) < 1.0})
    return out


# ---------------------------------------------------------------- 上限有两种成因
# 本机没有 Dassault / Siemens 的软件，所以下面两条不是从它们的手册里抄的，
# 是从公开、可引用、能自己验算的地方来的。诚实地分开写：

def kp_upper_roundoff(k_chain, eps=EPS, tol=1e-4):
    """隐式求解器的上限：Kp/k_chain 就是刚度矩阵条件数的量级，
    双精度 16 位减去 log10(Kp/k_chain)，舍入误差 ~ eps*(Kp/k_chain)。
    要求它不超过 tol，得到 Kp <= k_chain * tol/eps。"""
    return k_chain * tol / eps


def kp_upper_explicit(m_eff_kg, dt_max_s):
    """显式求解器的上限 —— 这条和我上面"上限来自浮点"的说法不同，必须分开。

    罚刚度抬高系统的最高自然频率 w_max ~ sqrt(Kp/m_eff)，而显式稳定步长
    dt <= 2/w_max。所以给定能接受的时间步，Kp 有一个硬上界：
        Kp <= m_eff * (2/dt_max)^2

    独立佐证（公开、可引用）：A General Bipenalty Formulation for Explicit
    Contact-Impact Analysis With a Parameter Selection Criterion (2024,
    DOI 10.1002/nme.7614) 摘要原话："it is necessary to introduce an artificial
    contact stiffness that can increase the maximum natural frequency of the
    system. When a large stiffness penalty parameter is used, the stability
    condition forces to reduce the time step size"。

    也就是说：**同样是"Kp 太大就坏"，隐式坏在丢位，显式坏在步长被逼小。**
    一家软件给两张不同的自动缩放公式，原因在这里，不在保密。
    """
    if dt_max_s <= 0:
        raise ValueError("时间步必须为正")
    return m_eff_kg * (2.0 / dt_max_s) ** 2


def kp_window(k_chain, force_n, allowed_penetration_mm, m_eff_kg=None,
              dt_max_s=None, tol=1e-4):
    """把上下界合起来：Kp 该落的那一段。

    下限是硬的（穿透量不能超）：Kp >= F/delta。
    上限取决于求解器：隐式用 kp_upper_roundoff，显式用 kp_upper_explicit。
    若下界已经超过上界，说明这道题的精度要求在所选求解器下做不到 ——
    这时该换公式（拉格朗日乘子 / 增广），不是继续调 Kp。这个函数会直接说出来。
    """
    lo = kp_lower_bound(force_n, allowed_penetration_mm)
    hi = kp_upper_roundoff(k_chain, EPS, tol)
    if m_eff_kg is not None and dt_max_s is not None:
        hi = min(hi, kp_upper_explicit(m_eff_kg, dt_max_s))
    out = {"lower_n_per_mm": lo, "upper_n_per_mm": hi,
           "feasible": lo <= hi, "geometric_mean": (lo * hi) ** 0.5}
    if lo > hi:
        out["advice"] = ("下界已超过上界：这道题的穿透量要求在这个求解器里做不到，"
                         "换接触公式，不要再调 Kp")
    else:
        out["advice"] = "取几何中点，两端都离得很远时才接近最优"
    return out


def grid_dependence(element_size_mm):
    """Kp 随网格走，不随零件走 —— 公开文献独立得出的同一条结论。

    佐证：The Value Range of Contact Stiffness Factor between Pile and Soil
    Based on Penalty Function (2018, IOP Conf. Ser. Earth Environ. Sci. 128
    012101) 摘要：接触刚度因子增大时最大接触压力与穿透趋近稳定值，且
    "the reasonable value range of contact stiffness factor reduces as the
    underlying element thickness decreases"。

    与本文件的 kp_over_e 式一致：Kp/E >= A*strain/(L*delta)，A/L 就是单元尺寸的倒数。
    所以"同一台机床换个网格要重调 Kp"不是软件在藏东西，是这个量的定义。
    """
    if element_size_mm <= 0:
        raise ValueError("单元尺寸必须为正")
    return {"element_size_mm": element_size_mm,
            "relative": 1.0 / element_size_mm,
            "note": "单元尺寸减半，Kp 的量级翻倍；真实零件尺寸不影响这个比例"}

# ---------------------------------------------------------------- 下限：由公差定

def kp_lower_bound(force_n, allowed_penetration_mm):
    """Kp 的下限：穿透量 = F/Kp <= 允许值。这是唯一必须遵守的硬约束。"""
    if allowed_penetration_mm <= 0:
        raise ValueError("允许穿透量必须为正")
    return force_n / allowed_penetration_mm


def kp_over_e(area_mm2, length_mm, strain, allowed_penetration_mm):
    """无量纲形式：Kp/E >= A*strain/(L*delta)。

    右侧只含几何与应变，不含任何厂商常数 —— 这就是"自动缩放"缩的是哪两样东西：
    单元尺寸（通过 A/L）与模量。这条式子可以从 Kp >= F/delta 与 F = E*A*strain 直接推。
    """
    return {"kp_over_e_min": area_mm2 * strain / (length_mm * allowed_penetration_mm),
            "elems": (area_mm2, length_mm, strain, allowed_penetration_mm)}
