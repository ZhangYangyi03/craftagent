# -*- coding: utf-8 -*-
"""closedloop.py -- 实时闭环：量测 -> 估计 -> 修正，按固定周期跑，按墙钟记截止期。

为什么闭环要单独一层
--------------------
前面几层都是"离线算一次"：给一个模型，算出一个数。闭环不一样，它是
**每拍都要在下一个采样到来之前算完**。这一层要回答三个不同的问题，
而且它们的验收标准互不相干：

一、估计器值不值。量测有噪声，估计器把噪声压下去多少？这个有闭式的
    答案：后验方差 = 先验方差 / (1 + snr)，snr = P⁻/R。不是拟合出来的。
二、闭环整体值不值。估计器压噪，补偿器把压出来的偏差补掉，但补偿量
    自己带着估计误差和噪声，又被机床的跟随滞后滤一遍。净效果是变好
    还是变坏，取决于漂移有多慢、传感器有多脏、环路增益多大。这一条
    **没有** 1/(1+g) 那种漂亮公式，它的精确值是离散 Lyapunov 方程的解。
    所以这里老老实实解 Lyapunov，不用仿真当标准。
三、来不来得及。一拍的计算超过采样周期就是 miss。miss 必须被数出来，
    不能悄悄顺延、也不能把实际周期写成标称周期。

关于 1/(1+snr)
--------------
把 g 理解成"环路增益"很容易，但那个 1/(1+g) 只在**估计器**这一步成立，
而且 g 是 snr = P⁻/R，不是 Kalman 增益 g_K。两者差一个变换：
    P⁺ = P⁻·R/(P⁻+R) = P⁻/(1+snr)，  snr = P⁻/R
    P⁺/P⁻ = 1/(1+snr) = 1 − g_K
闭环的**输出**方差比没有这个形式。拿它去套输出方差，会在传感器很脏、
增益很小的时候给出"闭环一定更好"的错误结论 —— 实测那一片区域闭环是
更差的，这里把分界算出来。

时间的两种读法
--------------
`VirtualClock` 只用于验收：sleep 推进虚拟时间，于是"哪几拍超时"是确定
的，测试不会因为机器忙而红。`WallClock` 是真实读数，用来报这个循环在
这台机器上一拍到底花多少微秒。虚拟时钟算出来的数一个都不往 README 里
写，写进去的必须是墙钟。

没有做的
--------
不做实时调度（不设优先级、不锁内存、不做 CPU 亲和），所以这一层报的
"missed" 是**计算耗时**超期，不是操作系统的调度抖动。真机上的抖动
要靠 RT 内核和 EtherCAT 之类的总线，不是 Python 能保证的。
"""
from __future__ import annotations

import math
import random
import time

__all__ = ["VirtualClock", "WallClock", "riccati", "filter_gate",
           "reference", "run", "loop_gate", "crossover", "deadline_gate",
           "step_gate", "check"]


# ---------------------------------------------------------------- 时钟

class VirtualClock:
    """虚拟时钟：只有 sleep 推进时间。用于确定性的截止期验收。"""

    __slots__ = ("t",)

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, d):
        if d > 0.0:
            self.t += d


class WallClock:
    """墙钟：真实经过的时间。用于报这台机器上的实测开销。"""

    __slots__ = ("t0",)

    def __init__(self):
        self.t0 = time.perf_counter()

    def now(self):
        return time.perf_counter() - self.t0

    def sleep(self, d):
        if d > 0.0:
            time.sleep(d)


# ---------------------------------------------------------------- 小工具

def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _var(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = _mean(v)
    return sum((t - m) ** 2 for t in v) / (n - 1)


def _pct(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _solve(M, b):
    """高斯消元（列主元）。n 最大 16，够 Lyapunov 的 n^2 用。"""
    n = len(M)
    A = [list(M[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-14:
            raise ZeroDivisionError("Lyapunov 方程奇异（闭环不稳定或不可稳）")
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        for r in range(col + 1, n):
            f = A[r][col] / pv
            if f != 0.0:
                row_r, row_c = A[r], A[col]
                for c in range(col, n + 1):
                    row_r[c] -= f * row_c[c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = A[i][n]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


def _lyap(A, Qn):
    """解 S = A S A^T + Qn（列优先 Kronecker：vec(ASA^T) = (A⊗A)vec(S)）。"""
    n = len(A)
    m = n * n
    M = [[0.0] * m for _ in range(m)]
    for i in range(n):
        for j in range(n):
            r = i + j * n
            M[r][r] = 1.0
            for k in range(n):
                aik = A[i][k]
                if aik == 0.0:
                    continue
                for l in range(n):
                    M[r][k + l * n] -= aik * A[j][l]
    rhs = [Qn[i][j] for j in range(n) for i in range(n)]
    v = _solve(M, rhs)
    S = [[v[i + j * n] for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m_ = 0.5 * (S[i][j] + S[j][i])
            S[i][j] = S[j][i] = m_
    return S


def _spectral_radius(A, iters=300):
    """Gelfand 公式的实用版：||A^k v||^(1/k) -> |lambda|max。

    每一步都把 v 归一化，所以 ||A^k v|| 是各步范数的连乘；在对数域累加
    再除以 k，数值上不会溢出也不会下溢。
    """
    n = len(A)
    v = [1.0 + 0.1 * i for i in range(n)]
    nv = math.sqrt(sum(t * t for t in v))
    v = [t / nv for t in v]
    logsum = 0.0
    for _ in range(iters):
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        nw = math.sqrt(sum(t * t for t in w))
        if nw <= 0.0:
            return 0.0
        logsum += math.log(nw)
        v = [t / nw for t in w]
    return math.exp(logsum / iters)


# ---------------------------------------------------------------- 估计器

def riccati(Q, R, phi=1.0, iters=5000, tol=1e-15):
    """一维稳态 Riccati：漂移 d_{k+1}=phi d_k + w，量测 y=d+v。

    返回 P_prior（先验方差）、g_k（Kalman 增益）、P_post（后验方差）、
    snr = P_prior/R、以及缩减因子 1/(1+snr)。
    R=0（量测无噪声）时后验方差精确为 0、snr 为无穷，单独处理而不是除零。
    """
    if R <= 0.0:
        P_prior = Q if phi >= 1.0 else 0.0
        return {"P_prior": P_prior, "P_post": 0.0, "g_k": 1.0,
                "snr": float("inf"), "ratio": 0.0, "ratio_alt": 0.0}
    P = Q if Q > 0 else R
    for _ in range(iters):
        P_prior = phi * phi * P + Q
        g = P_prior / (P_prior + R)
        P_new = (1.0 - g) * P_prior
        if abs(P_new - P) <= tol * max(1.0, abs(P_new)):
            P = P_new
            break
        P = P_new
    P_prior = phi * phi * P + Q
    g = P_prior / (P_prior + R)
    snr = P_prior / R
    return {"P_prior": P_prior, "P_post": P, "g_k": g, "snr": snr,
            "ratio": 1.0 / (1.0 + snr), "ratio_alt": 1.0 - g}


def filter_gate(phi=0.98, Q=1.0, R=4.0, N=200000, seed=0):
    """估计器验收：后验/先验方差比必须等于 1/(1+snr) = 1-g_K。

    这一条**不是**自己跟自己比：Riccati 给的是理论值，这里拿一个真正跑
    起来的滤波器量样本方差。滤波器用稳态增益（定常），所以两条误差序列
    都是平稳的，样本方差才有意义。
    """
    ss = riccati(Q, R, phi)
    g = ss["g_k"]
    rng = random.Random(seed)
    sw = math.sqrt(Q)
    sv = math.sqrt(R)
    d = 0.0
    dh = 0.0
    e_prior = []
    e_post = []
    for _ in range(N):
        d_prior = phi * dh
        d = phi * d + sw * rng.gauss(0.0, 1.0)
        y = d + sv * rng.gauss(0.0, 1.0)
        dh = d_prior + g * (y - d_prior)
        e_prior.append(d - d_prior)
        e_post.append(d - dh)
    v_prior = _var(e_prior)
    v_post = _var(e_post)
    ratio = v_post / v_prior if v_prior > 0 else float("nan")
    # 样本方差的相对标准差约 sqrt(2/N)；两条序列相关，容差取 3 倍
    tol = 3.0 * math.sqrt(2.0 / N)
    return {"theory_ratio": ss["ratio"], "alt_ratio": ss["ratio_alt"],
            "measured_ratio": ratio,
            "theory_prior": ss["P_prior"], "measured_prior": v_prior,
            "theory_post": ss["P_post"], "measured_post": v_post,
            "g_k": g, "snr": ss["snr"], "tol": tol,
            "ok": abs(ratio - ss["ratio"]) <= tol * ss["ratio"]}


# ---------------------------------------------------------------- 闭环参考值

def reference(phi=0.99, sigma_w=0.05, sigma_v=0.05, p=0.6, g_c=0.3,
              beta=None, Q=None, R=None):
    """闭环输出方差的**精确**参考值（离散 Lyapunov），不靠仿真。

    状态 z = (x, c, d, yhat_prev)：
        x_{k+1} = p x_k + (1-p) c_{k+1} + d_k       机床一阶跟随 + 漂移
        c_{k+1} = c_k - g_c yhat_k                  积分补偿
        d_{k+1} = phi d_k + w_k                     漂移
        yhat_k  = (1-beta) yhat_{k-1} + beta (x_k + v_k)   量测低通
    其中 beta 默认取标称随机游走模型的稳态 Kalman 增益。

    开环 = 补偿器不动（g_c 那条支路断掉），只有 x 与 d 两个状态。
    """
    Q = sigma_w * sigma_w if Q is None else Q
    R = sigma_v * sigma_v if R is None else R
    if beta is None:
        beta = riccati(Q, R, 1.0)["g_k"]
    k = (1.0 - p) * g_c
    A = [
        [p - k * beta, 1.0 - p, 1.0, -k * (1.0 - beta)],
        [-g_c * beta, 1.0, 0.0, -g_c * (1.0 - beta)],
        [0.0, 0.0, phi, 0.0],
        [beta, 0.0, 0.0, 1.0 - beta],
    ]
    u = [-k * beta, -g_c * beta, 0.0, beta]          # 量测噪声 v 的入口
    Qn = [[sigma_v * sigma_v * u[i] * u[j] for j in range(4)] for i in range(4)]
    Qn[2][2] += sigma_w * sigma_w
    S = _lyap(A, Qn)
    var_cl = S[0][0]

    Ao = [[p, 1.0], [0.0, phi]]
    Qo = [[0.0, 0.0], [0.0, sigma_w * sigma_w]]
    So = _lyap(Ao, Qo)
    var_ol = So[0][0]

    return {"var_closed": var_cl, "var_open": var_ol,
            "ratio": var_cl / var_ol if var_ol > 0 else float("nan"),
            "rho": _spectral_radius(A), "beta": beta,
            "poles_note": "rho 是 A 的谱半径（幂迭代估的），决定阶跃后的衰减率"}


# ---------------------------------------------------------------- 跑闭环

def run(ticks=400, period=0.002, clock=None, cost=0.0,
        phi=0.99, sigma_w=0.05, sigma_v=0.05, p=0.6, g_c=0.3, beta=None,
        limit=None, rate=None, step=(None, 0.0), slow=(), slow_factor=3.0,
        seed=0, x0=0.0):
    """按固定周期跑 ticks 拍，返回误差序列 + 时间账。

    cost        每拍模拟的计算耗时（秒）。虚拟时钟下必须给，否则每一拍都是 0。
    slow        故意变慢的拍号集合；这些拍额外 sleep(period*slow_factor)。
    limit       补偿量饱和值（绝对值），None 表示不限。
    rate        补偿量每拍最大变化量，None 表示不限。
    step        (k0, amp)：第 k0 拍起漂移叠加一个阶跃 amp。
    返回里的 missed / miss_idx 是**数出来的**，不是按标称周期推的。
    """
    clock = clock or WallClock()
    if beta is None:
        beta = riccati(sigma_w * sigma_w, sigma_v * sigma_v, 1.0)["g_k"]
    rng = random.Random(seed)
    k0, amp = step
    x = float(x0)
    c = 0.0
    d = 0.0
    yh = 0.0
    err = []
    cmd = []
    dts = []
    miss_idx = []
    sat = 0
    ratelim = 0
    for k in range(ticks):
        t0 = clock.now()
        # 1 量测
        y = x + sigma_v * rng.gauss(0.0, 1.0)
        # 2 估计（低通，稳态增益 beta）
        yh = (1.0 - beta) * yh + beta * y
        # 3 修正：积分作用（增量正比于估计误差），再限速、再限幅
        dc = -g_c * yh
        if rate is not None and abs(dc) > rate:
            dc = math.copysign(rate, dc)
            ratelim += 1
        c_new = c + dc
        if limit is not None and abs(c_new) > limit:
            c_new = math.copysign(limit, c_new)
            sat += 1
        c = c_new
        # 4 漂移推进；阶跃是叠加在机床输入上的**偏置**，不是加进漂移的
        #   增量里 —— 加进增量的话 AR(1) 会把 amp 累成 amp/(1-phi)，那就
        #    不是阶跃了。
        d = phi * d + sigma_w * rng.gauss(0.0, 1.0)
        bias = amp if (k0 is not None and k >= k0) else 0.0
        # 5 机床响应
        x = p * x + (1.0 - p) * c + d + bias
        err.append(x)
        cmd.append(c)
        # 6 计时：这一拍到底花了多久
        if cost:
            clock.sleep(cost)
        if k in slow:
            clock.sleep(period * slow_factor)
        dt = clock.now() - t0
        dts.append(dt)
        if dt > period:
            miss_idx.append(k)
    total = sum(dts)
    return {"ticks": ticks, "period": period, "cost": cost,
            "elapsed": total, "achieved": total / ticks if ticks else 0.0,
            "periods": dts,
            "p50": _pct(dts, 0.5), "p99": _pct(dts, 0.99), "max": max(dts),
            "missed": len(miss_idx), "miss_idx": miss_idx,
            "overrun": total / (ticks * period) if ticks else 0.0,
            "err": err, "cmd": cmd,
            "var_err": _var(err), "max_abs_err": max(abs(t) for t in err),
            "saturated": sat, "rate_limited": ratelim,
            "beta": beta, "settled": None}


# ---------------------------------------------------------------- 闭环验收

def loop_gate(phi=0.99, sigma_w=0.05, sigma_v=0.05, p=0.6, g_c=0.3,
              ticks=300000, seed=0, warmup=2000):
    """闭环输出方差：仿真 vs Lyapunov 精确解。

    注意这里**没有**要求 ratio < 1 —— 传感器脏、增益小的时候闭环更差，
    那是真实结论。要求的是仿真与参考值一致，好坏如实报。

    容差按**有效样本数**算，不按 ticks 算：闭环输出是强相关的（最慢的极点
    就在 0.99 附近），N 拍里独立信息只有 N(1-rho^2)/(1+rho^2) 拍。按 N 算
    容差会把自己卡死 —— 6 万拍时 4/sqrt(N)=1.6%，而实测偏差 2.0%，那是
    相关性的代价，不是模型错。加长到 24 万拍偏差就掉到 1.7e-4。
    """
    ref = reference(phi, sigma_w, sigma_v, p, g_c)
    r = run(ticks=ticks + warmup, period=1.0, clock=VirtualClock(),
            phi=phi, sigma_w=sigma_w, sigma_v=sigma_v, p=p, g_c=g_c,
            beta=ref["beta"], seed=seed)
    v = _var(r["err"][warmup:])
    rel = abs(v - ref["var_closed"]) / ref["var_closed"]
    rho = ref["rho"]
    n_eff = ticks * (1.0 - rho * rho) / (1.0 + rho * rho)
    tol = 4.0 * math.sqrt(2.0 / max(1.0, n_eff))
    return {"measured": v, "reference": ref["var_closed"],
            "var_open": ref["var_open"], "ratio": ref["ratio"],
            "rel": rel, "tol": tol, "n_eff": n_eff, "rho": rho,
            "ok": rel <= tol}


def crossover(phi=0.99, sigma_w=0.05, p=0.6, g_cs=None, sigma_vs=None):
    """闭环到底在什么条件下比开环差 —— 分界由 Lyapunov 精确算。

    两条线各扫一遍：固定噪声扫增益、固定增益扫噪声。返回每行的
    ratio = 闭环方差 / 开环方差，<1 才是赚了。
    """
    g_cs = g_cs or (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
    sigma_vs = sigma_vs or (0.0, 0.01, 0.05, 0.2, 1.0, 5.0)
    by_gain = []
    for g in g_cs:
        r = reference(phi, sigma_w, 0.05, p, g)
        by_gain.append({"g_c": g, "ratio": r["ratio"], "rho": r["rho"]})
    by_noise = []
    for sv in sigma_vs:
        r = reference(phi, sigma_w, sv, p, 0.3)
        by_noise.append({"sigma_v": sv, "ratio": r["ratio"]})
    win = [b["g_c"] for b in by_gain if b["ratio"] < 1.0]
    return {"by_gain": by_gain, "by_noise": by_noise,
            "gain_helps": (min(win), max(win)) if win else None}


# ---------------------------------------------------------------- 截止期

def deadline_gate(period=0.002, ticks=40, cost_frac=0.4, slow=(7, 8, 9),
                  slow_factor=3.0):
    """截止期验收：故意慢的拍必须被数成 miss。

    虚拟时钟下这是确定的：cost = 0.4*period 的拍不超时，被 slow 命中的
    拍花 0.4+3.0 = 3.4 倍周期，必然超时。所以要正好 |slow| 次 miss，
    且 miss 的拍号与 slow 完全相同 —— 不能多算、不能漏算、不能顺延。
    """
    p_ = period
    cost = p_ * cost_frac
    r = run(ticks=ticks, period=p_, clock=VirtualClock(), cost=cost,
            slow=set(slow), slow_factor=slow_factor, phi=0.99, sigma_w=0.01,
            sigma_v=0.01, g_c=0.2, seed=1)
    expect = set(slow)
    want = (cost * ticks + p_ * slow_factor * len(expect)) / ticks
    ok = (set(r["miss_idx"]) == expect and r["missed"] == len(expect)
          and abs(r["achieved"] - want) <= 1e-12 * want
          and r["achieved"] >= cost * (1.0 - 1e-12))
    return {"missed": r["missed"], "expected": len(expect),
            "miss_idx": r["miss_idx"], "achieved": r["achieved"],
            "period": p_, "cost": cost, "overrun": r["overrun"],
            "p50": r["p50"], "max": r["max"], "ok": ok}


def wall_timing(period=0.002, ticks=200, slow=(50, 51), slow_factor=4.0):
    """墙钟实测：这个循环在这台机器上一拍真实花多少微秒。

    这里不断言 miss 的个数 —— 操作系统会给什么就是什么。断言的是
    被 slow 命中的那几拍一定被记成 miss（那是必然超时的）。
    """
    r = run(ticks=ticks, period=period, clock=WallClock(), cost=0.0,
            slow=set(slow), slow_factor=slow_factor, phi=0.99, sigma_w=0.01,
            sigma_v=0.01, g_c=0.2, seed=2)
    hit = set(slow) & set(r["miss_idx"])
    return {"period_us": period * 1e6, "ticks": ticks,
            "p50_us": r["p50"] * 1e6, "p99_us": r["p99"] * 1e6,
            "max_us": r["max"] * 1e6, "achieved_us": r["achieved"] * 1e6,
            "missed": r["missed"], "overrun": r["overrun"],
            "slow_hit": sorted(hit), "slow_expected": sorted(slow),
            "ok": hit == set(slow)}


# ---------------------------------------------------------------- 阶跃

def step_gate(phi=0.99, sigma_w=0.0, sigma_v=0.0, p=0.6, g_c=0.3,
              amp=10.0, k0=20, ticks=400, band=0.5):
    """阶跃扰动：实测收敛拍数 vs 理想积分的下界。

    预测值取**无滞后、无滤波**的理想积分器：c 每拍把误差打掉 g_c 的比例，
    所以误差按 (1-g_c) 衰减，把 amp 压到 band 需要
        N_ideal = ln(amp/band) / ln(1/(1-g_c))
    这是下界（机床跟随滞后 p 和量测低通都会让它更慢），不是拟合值。

    为什么不用 A 的谱半径当预测：那个 4 状态里有漂移自己的极点 0.99，
    它**不被阶跃激励**（阶跃进的是 x，不进 d），谱半径于是给 340 拍，
    而实测 18 拍。用没被激励的模态预测收敛时间是个陷阱，这里踩过一次。
    """
    ideal = math.log(amp / band) / math.log(1.0 / (1.0 - g_c))
    r = run(ticks=ticks, period=1.0, clock=VirtualClock(), phi=phi,
            sigma_w=0.0, sigma_v=0.0, p=p, g_c=g_c, beta=0.618,
            step=(k0, amp), seed=3)
    settle = None
    for i in range(k0, len(r["err"])):
        if all(abs(t) <= band for t in r["err"][i:]):
            settle = i - k0
            break
    return {"settle": settle, "ideal": ideal, "band": band, "amp": amp,
            "peak": max(abs(t) for t in r["err"][k0:]),
            "overshoot": max(r["err"][k0:]) / amp,
            "ok": settle is not None and settle <= 3.0 * ideal + 3.0}


# ---------------------------------------------------------------- 汇总

def check(verbose=False):
    """四道门全跑一遍，返回结构化结果（CLI 负责排版）。"""
    out = {}
    out["filter"] = filter_gate()
    out["loop"] = loop_gate()
    out["cross"] = crossover()
    out["deadline"] = deadline_gate()
    out["step"] = step_gate()
    out["wall"] = wall_timing()
    if verbose:
        f = out["filter"]
        print("一 估计器：后验/先验 = 1/(1+snr) = 1-g_K")
        print("  理论 %.6f (= 1-%.6f)  实测 %.6f  (snr=%.3f)  -> %s"
              % (f["theory_ratio"], f["g_k"], f["measured_ratio"],
                 f["snr"], "过" if f["ok"] else "不过"))
        lp = out["loop"]
        print("二 闭环输出方差：Lyapunov 精确解 vs 仿真")
        print("  参考 %.6e  实测 %.6e  相对差 %.2e（容差 %.2e）  -> %s"
              % (lp["reference"], lp["measured"], lp["rel"], lp["tol"],
                 "过" if lp["ok"] else "不过"))
        print("  闭环/开环 = %.4f（<1 才是赚了）" % lp["ratio"])
        print("三 截止期（虚拟时钟，确定性）")
        d = out["deadline"]
        print("  标称周期 %.2f ms，故意慢的拍 %s，数出 miss %d 次，"
              "实际平均周期 %.3f ms  -> %s"
              % (d["period"] * 1e3, d["miss_idx"], d["missed"],
                 d["achieved"] * 1e3, "过" if d["ok"] else "不过"))
        s = out["step"]
        print("四 阶跃：实测 %s 拍收敛（超调 %.2f 倍），理想积分下界 %.1f 拍  -> %s"
              % (s["settle"], s["overshoot"], s["ideal"],
                 "过" if s["ok"] else "不过"))
        w = out["wall"]
        print("五 墙钟实测：周期 %.1f µs，实测中位 %.1f µs / p99 %.1f µs / "
              "最大 %.1f µs，miss %d 次"
              % (w["period_us"], w["p50_us"], w["p99_us"], w["max_us"],
                 w["missed"]))
    return out
