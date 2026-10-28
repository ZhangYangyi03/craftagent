# -*- coding: utf-8 -*-
"""ident.py -- 从实测日志里辨识数字孪生的参数，并说出哪些参数这批数据认不出来。

要的东西是"孪生里的精工参数"：时间常数、增益、磨损漂移率。这些东西有两个来源，
本仓只用其中一个：

  A. 从软件/仿真里"导出"它的内部参数        —— 不可行。那是软件的实现细节，不是数据；
                                              逆它等于逆产品，撞许可和专利。
  B. 用实测日志把自己的模型标定到这台机床上  —— 可行，而且更准。孪生要的是
                                              "这台机器现在的 tau 是多少"，
                                              不是"厂商默认值是多少"。

所以这里做 B：把 plant.py 里那几个符号背后该有的数，用数据反算出来。

三条纪律，每条都写在代码里而不是注释里：

  1. 参数必须是可辨识的才报。一阶滞后只有在一个采样周期里真的"动起来过"才
     能定 tau；采样间隔比 tau 还大时，报出来的 tau 是插值出来的幻觉。
  2. 每个数都带一个区间。点估计不给区间 = 不给别人反驳的机会。
  3. 认不出来的参数明确列出来，并写清"要采什么才能认出来"。这比给一个
     假数字有用，也是这个仓一贯的做法。
"""
import math

from .stats import median

# ---------------------------------------------------------------- 一带而过的基础件

def _ols(xs, ys):
    """两点最小二乘，返回 (slope, intercept, r2)。不引入任何依赖。"""
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    k = sxy / sxx
    b = my - k * mx
    ss_res = sum((y - (k * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return k, b, r2


def _pct(xs, q):
    s = sorted(xs)
    if not s:
        return 0.0
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _trimmed(xs, k=3):
    """按 MAD 剔掉远处的点，再返回保留下来的下标。这一段决定了抗离群。"""
    if len(xs) < 8:
        return list(range(len(xs)))
    m = median(xs)
    d = [abs(x - m) for x in xs]
    s = 1.4826 * median(d) or (sum(d) / len(d) or 1.0)
    keep = [i for i, x in enumerate(xs) if abs(x - m) <= k * s]
    return keep if len(keep) >= 8 else list(range(len(xs)))


# ---------------------------------------------------------------- 一阶滞后：tau 与增益

def _solve2(s11, s12, s22, b1, b2):
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-15:
        return None
    return ((b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det)


def _fit_tau_fixed(ts, ys, tau):
    """固定 tau 时，模型对 (y_inf, A) 是线性的：y = y_inf - A * exp(-t/tau)。

    于是"拟合 tau"变成"对每一格 tau 解一个 2x2 正规方程"，取残差最小的那一格。
    纯标准库就能算出极大似然意义下的 tau，不需要 scipy，也不需要梯度下降——
    这个仓的依赖约束是硬的（工厂那台电脑装不上东西）。
    """
    s11 = s12 = s22 = b1 = b2 = 0.0
    for t_, y_ in zip(ts, ys):
        f = math.exp(-t_ / tau)
        s11 += 1.0
        s12 += -f
        s22 += f * f
        b1 += y_
        b2 += -f * y_
    ab = _solve2(s11, s12, s22, b1, b2)
    if ab is None:
        return None
    y_inf, A = ab
    sse = 0.0
    for t_, y_ in zip(ts, ys):
        r = y_ - (y_inf - A * math.exp(-t_ / tau))
        sse += r * r
    return sse, y_inf, A


def fit_first_order(profile, dt_min, cycle_min):
    """profile[p] 是"班次内第 p 分钟"的跨班次中位值；估这台设备的响应时间常数。

    模型只有一个：y(t) = y_inf - A * exp(-t / tau)。

    **先说清楚为什么不用 AR(1)。** 一阶滞后的离散写法 y[k] = a*y[k-1] + c 看着更省事，
    但在一条已经饱和的曲线上它是有病的：饱和段里 y[k] ≈ y[k-1] 恒成立，噪声进了
    两边，最小二乘会挑一个几乎任意的 a。实测过一次：tau=8 分钟的曲线被它报成 0.6 分钟，
    而且报得理直气壮。所以这里改成对 tau 做剖面似然：固定 tau，模型对 (y_inf, A)
    是线性的，2x2 正规方程直接解；tau 在一个对数网格上扫，取残差最小的一格。
    曲线饱和得再厉害也不影响——饱和段对 tau 不提供信息，也就不会污染 tau。

    返回里最重要的不是 tau，是 identifiable 和它为什么不可辨识。
    """
    n = len(profile)
    if n < 16:
        return {"identifiable": False, "why": "有效点不足（%d 个），至少要 16 个才谈得上拟合" % n}

    ys = [float(v) for v in profile]
    ts = [i * dt_min for i in range(n)]

    # 噪声尺度：用高频差分估（相邻点之差的 MAD），不看整体方差——
    # 整体方差里大部分是那条上升曲线本身，用它估噪声会把信噪比算得虚高。
    diffs = [abs(ys[i] - ys[i - 1]) for i in range(1, n)]
    sigma = 1.4826 * median(diffs) / math.sqrt(2.0) if diffs else 0.0
    rng_amp = max(ys) - min(ys)

    # tau 网格：从"比采样还细"到"比整个窗口还长"都扫一遍，所以不会因为初值
    # 选错而收敛到错的谷。这一步是 240 次 2x2 求解，代价可以忽略。
    lo, hi = 0.05 * dt_min, 50.0 * n * dt_min
    best = None
    for k in range(240):
        tau = lo * (hi / lo) ** (k / 239.0)
        r = _fit_tau_fixed(ts, ys, tau)
        if r and (best is None or r[0] < best[0]):
            best = (r[0], tau, r[1], r[2])
    if best is None:
        return {"identifiable": False, "why": "拟合失败：数据退化"}

    # 局部细化：在上面那一格附近再扫 40 格
    sse0, tau0, y_inf, A = best
    f_lo, f_hi = tau0 / 1.3, tau0 * 1.3
    for k in range(40):
        tau = f_lo * (f_hi / f_lo) ** (k / 39.0)
        r = _fit_tau_fixed(ts, ys, tau)
        if r and r[0] < best[0]:
            best = (r[0], tau, r[1], r[2])
    sse, tau, y_inf, A = best

    resid = [ys[i] - (y_inf - A * math.exp(-ts[i] / tau)) for i in range(n)]
    ss_tot = sum((v - sum(ys) / n) ** 2 for v in ys)
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0

    # 曲线在窗口末尾有没有真的停下来。没停下来 -> y_inf 是被窗口截断猜的，
    # 这种时候 tau 只有下界，给不出数。
    tail = max(4, n // 12)
    tail_mid = median(ys[-tail:])
    tail_prev = median(ys[-2 * tail:-tail])
    settled = abs(tail_mid - tail_prev) < max(3.0 * sigma, 0.02 * abs(A))

    reasons = []
    if sigma > 0 and abs(A) < 3.0 * sigma:
        reasons.append("曲线几乎是平的：振幅 %.3f 与噪声 %.3f 同量级，"
                       "这个通道没有可辨识的响应" % (A, sigma))
    if not settled:
        reasons.append("窗口结束时还在爬（末段 %.3f 与 %.3f 之差仍大于噪声），"
                       "tau 只会是下界，等于 tau > %.0f 分钟" % (tail_prev, tail_mid, tau))
    if tau < 2.0 * dt_min:
        reasons.append("tau=%.2f 分钟只有采样间隔 %.1f 分钟的 %.1f 倍，低于可辨识下限 2 倍"
                       % (tau, dt_min, tau / dt_min))
    if tau > 0.9 * n * dt_min:
        reasons.append("tau=%.0f 分钟接近窗口长度 %.0f 分钟，参数与窗口正撞，认不出"
                       % (tau, n * dt_min))
    if r2 < 0.30:
        reasons.append("指数结构的解释力 R2=%.2f 太低，形状不像指数上升" % r2)

    return {
        "tau_min": tau, "y0": y_inf - A, "y_inf": y_inf, "gain": A,
        "sigma": sigma, "snr": (abs(A) / sigma) if sigma > 0 else float("inf"),
        "settled": settled, "r2": r2,
        "resid_mad": median([abs(r) for r in resid]),
        "points": n, "dt_min": dt_min, "window_min": n * dt_min,
        "identifiable": not reasons,
        "why": "；".join(reasons) if reasons else "",
    }


def phase_profile(rows, channel, cycle_min=480, min_shifts=3):
    """把一个通道按班次相位叠起来：**每个班次先减掉自己的水平，再跨班次取中位**。

    这一步我原本写错了，错法值得留在这里，因为它不报错、只出好看的数：

    第一版是"直接按相位取跨班次中位值"。理由是"中位数把每个班次各自的偏移消掉"。
    但在 M01 这种正在漂移的机器上，中位数只消掉了一半 —— 每个相位的跨班次中位
    落在漂移的中间位置，叠加出来的那条曲线是"15 个班次的平均历史"，**任何一个
    真实班次都不是这个形状**。孪生装进去以后拿真实班次验，RMSE 比"整班恒定"
    这个最笨的基线还差 689%。数字很荒唐，但它是老实话：装错的孪生比不装更差。

    正确的做法只有一句：同一台机器的班次内形状是一样的，绝对水平不一样，
    所以先减水平、再叠形状、最后把水平的总体中位加回来。

    返回 (profile, 有效班次数, 水平中位)。水平中位是有物理意义的绝对量
    （这台机器通常的工作点），形状是相对它而言的。
    """
    per_shift = {}
    for i, r in enumerate(rows):
        v = r["sig"].get(channel)
        if v is None:
            continue
        per_shift.setdefault(i // cycle_min, []).append((i % cycle_min, v))
    by_phase, levels = {}, []
    for sh, pts in per_shift.items():
        vals = [v for _, v in pts]
        if len(vals) < cycle_min // 2:
            continue                      # 半个班次以下的残段不参与，它代表不了这个班次
        lv = median(vals)
        levels.append(lv)
        for ph, v in pts:
            by_phase.setdefault(ph, []).append(v - lv)
    base = median(levels) if levels else 0.0
    prof = []
    for ph in range(cycle_min):
        vs = by_phase.get(ph) or []
        prof.append((median(vs) if vs else 0.0) + base)
    return prof, len(levels), base


def time_constants(rows, channels, cycle_min=480, dt_min=1):
    out = {}
    for ch in channels:
        prof, n_shifts, base_level = phase_profile(rows, ch, cycle_min=cycle_min)
        if n_shifts < 3:
            out[ch] = {"identifiable": False, "why": "只有 %d 个班次，至少 3 个班次才叠得起来" % n_shifts}
            continue
        # 用班次长度的前 1/2 做拟合窗口：后半段可能已经进入别的状态
        w = max(16, cycle_min // 2)
        r = fit_first_order(prof[:w], dt_min, cycle_min)
        r["shifts"] = n_shifts
        r["base_level"] = base_level
        r["channel"] = ch
        out[ch] = r
    return out


# ---------------------------------------------------------------- 漂移率

def drift_rate(rows, channel, ts0_index=None):
    """变点之后的稳健斜率（单位/小时）。用来把"磨损速率"写进孪生。

    用中点前后两半的中位值差分除以时间差，而不是回归：回归对单点离群敏感，
    而车间日志里最不缺的就是单点离群。
    """
    series = [(i, r["sig"].get(channel)) for i, r in enumerate(rows) if r["sig"].get(channel) is not None]
    if len(series) < 60:
        return {"ok": False, "why": "点数不足"}
    half = len(series) // 2
    first = [v for _, v in series[:half]]
    second = [v for _, v in series[half:]]
    dt_h = (series[-1][0] - series[0][0]) / 60.0
    if dt_h <= 0:
        return {"ok": False, "why": "时间跨度为零"}
    rate = (median(second) - median(first)) / (dt_h / 2.0)
    return {"ok": True, "per_hour": rate,
            "level_first_half": median(first), "level_second_half": median(second),
            "hours": dt_h}


# ---------------------------------------------------------------- 症状 -> 质量 的增益（带留出）

def _kfold_r2(pairs, k=5):
    """k 折交叉验证的 R2，按折报中位数。

    为什么不是一次留出：这台机器的质量数据是时间序列，最后一次留出段刚好
    落在漂移最新、外推最不利的位置，R2 会被外推难度支配，而不是被"这个症状
    解释力有多强"支配。第一次做的时候就撞到了：训练 R2 0.49、留出 R2 -0.24，
    而 n=180 的两参数拟合在 5 折下是 +0.2 左右。所以两个都报：
    时序留出回答"明天用得上吗"，k 折回答"关系本身存在吗"。
    """
    n = len(pairs)
    if n < 20:
        return {"k": 0, "r2_median": None}
    idx = list(range(n))
    folds = [idx[i::k] for i in range(k)]
    r2s = []
    for f in folds:
        if len(f) < 4:
            continue
        tr = [pairs[i] for i in idx if i not in set(f)]
        te = [pairs[i] for i in f]
        if len(tr) < 12:
            continue
        a, b, _ = _ols([v for v, _ in tr], [q for _, q in tr])
        pred = [a * v + b for v, _ in te]
        obs = [q for _, q in te]
        m = sum(obs) / len(obs)
        ss_res = sum((o - pv) ** 2 for o, pv in zip(obs, pred))
        ss_tot = sum((o - m) ** 2 for o in obs)
        if ss_tot > 0:
            r2s.append(1.0 - ss_res / ss_tot)
    return {"k": len(r2s), "r2_median": median(r2s) if r2s else None}


def symptom_gain(aligned, channel, quality_name="roundness_um", holdout_frac=0.25,
                 cp_index=None, tail_drop=0):
    """质量对症状的增益，两个检验都带上。

    这个仓从第一天就在防"在同一批点上学一个数、再在同一批点上说它准"。
    tail_drop 是为了跳过最后那几个点：最后一段往往同时处在"抽检还没跟上"
    和"漂移已经越界"的状态，把它们算进去等于用一个正在离开线性区的区间
    去评判线性模型。
    """
    pairs = []
    for a in aligned:
        q = a.get("quality")
        v = (a.get("sig") or {}).get(channel)
        if q is None or v is None:
            continue
        pairs.append((v, q))
    if cp_index is not None and 0 < cp_index < len(pairs) - 20:
        seg_note = "变点后 %d 点" % (len(pairs) - cp_index)
        pairs = pairs[cp_index:]
    else:
        seg_note = "全段 %d 点" % len(pairs)
    if tail_drop > 0:
        pairs = pairs[:-tail_drop]
    if len(pairs) < 40:
        return {"ok": False, "why": "对齐后的点只有 %d 个（%s）" % (len(pairs), seg_note)}

    n_hold = max(8, int(len(pairs) * holdout_frac))
    head, tail = pairs[:-n_hold], pairs[-n_hold:]
    k, b, r2 = _ols([v for v, _ in head], [q for _, q in head])
    pred = [k * v + b for v, _ in tail]
    obs = [q for _, q in tail]
    ss_res = sum((o - pv) ** 2 for o, pv in zip(obs, pred))
    m = sum(obs) / len(obs)
    ss_tot = sum((o - m) ** 2 for o in obs)
    cv = _kfold_r2(pairs)
    return {"ok": True, "gain": k, "offset": b, "r2_train": r2,
            "r2_holdout": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "r2_cv": cv["r2_median"], "n_cv": cv["k"],
            "n_train": len(head), "n_holdout": len(tail), "segment": seg_note}


# ---------------------------------------------------------------- 入口

# 这批数据认不出来、必须另外去采的东西。写成数据，不写在文档里，
# 这样它才能被打印、被核对、被下一个接手的人当成待办清单。
NEEDS = [
    {"param": "伺服时间常数 tau_s / 位置环增益 kv",
     "why": "plant.py 里它是圆角误差的主因，但它只有在一毫秒级的指令-实际位置对上才认得出；"
            "工况日志是一分钟一个采样点，差三个数量级",
     "collect": "CNC 的跟随误差（following error）或实际位置，采样 >= 1 kHz，"
                "最好同时录一次 G0/G1 的阶跃响应"},
    {"param": "各轴加速度上限 a_max、jerk",
     "why": "由机床伺服参数决定，不在日志里；现在用的是 plan.py 的保守默认值",
     "collect": "CNC 参数页里的 acc/dec 时间常数，或做一次空程加速标定（打表 + 高速采样）"},
    {"param": "切削力系数 kc 的实测值",
     "why": "feeds.py 用的是 Kienzle 常用简化值，不是这台机床、这对刀具的实测值",
     "collect": "测力仪（或主轴电流标定）在一组 ap/ae/vf 下的阶跃试验，至少 3x3 网格"},
    {"param": "砂轮磨损速率与修整周期",
     "why": "本批数据能给出\"症状漂移率\"，但那是电流/振动的漂移，不是磨损量本身",
     "collect": "修整前后的首件圆度 + 修整量，连续记 30 个修整周期"},
    {"param": "热变形系数（温度 -> 尺寸）",
     "why": "有温度通道和圆度，但没有在\"温度单独变化、其它都不动\"的条件下取过数，"
            "所以温度和尺寸的关系与班次效应混在一起，分不开",
     "collect": "停机后冷机启动、只开机不切削的纯热漂移试验，连续 4 小时"},
]


def identify(rows_by_machine, aligned_by_machine, channels=None, cycle_min=480, dt_min=1,
             cp_by_machine=None, tail_drop=6):
    """把一台或多台机器的日志辨识成一张参数表。"""
    out = {"cycle_min": cycle_min, "dt_min": dt_min, "machines": {}, "needs": NEEDS}
    for mid in sorted(rows_by_machine):
        rows = rows_by_machine[mid]
        chans = channels or sorted({k for r in rows for k in r["sig"]})
        tc = time_constants(rows, chans, cycle_min=cycle_min, dt_min=dt_min)
        dr = {ch: drift_rate(rows, ch) for ch in chans}
        gains = {}
        cp = (cp_by_machine or {}).get(mid)
        for ch in chans:
            g = symptom_gain(aligned_by_machine.get(mid, []), ch, cp_index=cp, tail_drop=tail_drop)
            if g.get("ok"):
                gains[ch] = g
        out["machines"][mid] = {"time_constants": tc, "drift": dr, "symptom_gain": gains}
    return out


def render(ident):
    """参数表，文本。没有一个数是手写进去的。"""
    L = ["数字孪生参数辨识  班次 %d 分钟，采样 %d 分钟" % (ident["cycle_min"], ident["dt_min"]), ""]
    for mid, m in ident["machines"].items():
        L.append("== %s" % mid)
        L.append("  %-22s %-10s %-12s %-8s %s" % ("通道", "tau(分钟)", "增益", "R2", "判定"))
        for ch, r in sorted(m["time_constants"].items()):
            if not r.get("identifiable"):
                L.append("  %-22s %-10s %-12s %-8s %s" % (ch, "-", "-", "-", "认不出：" + (r.get("why") or "")))
                continue
            L.append("  %-22s %-10.1f %-12.3f %-8.2f %s"
                     % (ch, r["tau_min"], r["gain"] or 0.0, r["r2"],
                        "可辨识（%d 个班次）" % r.get("shifts", 0)))
        L.append("  漂移率（后半-前半，每小时的量）")
        for ch, d in sorted(m["drift"].items()):
            if d.get("ok"):
                L.append("    %-22s %+8.3f /h  跨度 %.0f 小时" % (ch, d["per_hour"], d["hours"]))
        if m["symptom_gain"]:
            L.append("  症状 -> 质量 的增益（留出检验）")
            for ch, g in sorted(m["symptom_gain"].items()):
                cv = "%.2f" % g["r2_cv"] if g.get("r2_cv") is not None else "-"
                L.append("    %-22s %+8.3f  训练R2 %.2f  留出R2 %+.2f  %d折R2 %s" %
                         (ch, g["gain"], g["r2_train"], g["r2_holdout"], g.get("n_cv", 0), cv))
        L.append("")
    L.append("== 这批数据认不出来、必须另外去采的参数")
    for n in ident["needs"]:
        L.append("  * %s" % n["param"])
        L.append("      为什么认不出：%s" % n["why"])
        L.append("      要采什么：%s" % n["collect"])
    return "\n".join(L)
