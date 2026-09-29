"""归因：谁先动，谁跟着动，动多少能把质量拉回来。
方法上必须先降采样：一分钟一个点的差分，噪声（0.15 um）远大于漂移（0.015 um/分钟），
信噪比被打穿，互相关出来全是 0.02 的噪声值。按 30 分钟取块中位数之后，漂移才浮出来。
这是相关性排序 + 物理常识，不是因果证明 —— 报告和回答里都必须写死这句。
读数量（温度、湿度、冷却液温度）不是旋钮：它排第一时，话术是"去查它上游的旋钮"。"""
from .stats import median, mad, pearson, ols_slope

READOUTS = ("temp_c", "coolant_temp_c", "ambient_rh")

# 读数量 -> 它上游可能被谁改动了。"去查哪个旋钮"这条线索就是这张表给的。
# 表里放的是"可能要查的",不是"一定是他"——报告必须照这个措辞写。
UPSTREAM = {
    "temp_c":         ["coolant_flow_l_min", "spindle_speed_rpm", "duty_cycle_pct",
                       "wheel_dress_depth_mm", "开机预热是否到位"],
    "coolant_temp_c": ["coolant_flow_l_min", "冷却液箱体温度/是否刚补液", "过滤是否堵"],
    "ambient_rh":     ["车间空调/除湿", "厂房门窗与夜班交接", "压缩空气含水"],
}

def upstream_of(channel, knobs=None):
    """给一个读数量，返回"该去查哪些上游旋钮"的候选，并标出其中哪些本批有记录。"""
    cands = list(UPSTREAM.get(channel, []))
    have = set(knobs or [])
    return [{"knob": k, "recorded": k in have} for k in cands]

def block_median(series, block=30):
    out = []
    for i in range(0, len(series), block):
        w = [v for v in series[i:i+block] if v is not None]
        out.append(median(w) if w else None)
    return out

def lead_lag(target, driver, block=30, max_lag_blocks=8, min_n=40):
    """块中位数 -> 一阶差分 -> 互相关。返回 (lag_min, r)，lag>0 表示 driver 领先 target。"""
    tb = block_median(target, block); db = block_median(driver, block)
    pairs = [(a, b) for a, b in zip(tb, db) if a is not None and b is not None]
    if len(pairs) < min_n + 2: return 0, 0.0
    t = [p[0] for p in pairs]; d = [p[1] for p in pairs]
    dt = [t[i+1]-t[i] for i in range(len(t)-1)]
    dd = [d[i+1]-d[i] for i in range(len(d)-1)]
    m = len(dt); best = (0, 0.0)
    for lag in range(-max_lag_blocks, max_lag_blocks + 1):
        if lag >= 0: a = dt[lag:]; b = dd[:m-lag]
        else:        a = dt[:m+lag]; b = dd[-lag:]
        if len(a) < min_n: continue
        r = pearson(a, b)
        if abs(r) > abs(best[1]): best = (lag*block, round(r, 3))
    return best

def attribute(channels, quality_series, quality_cps, window=180, r_min=0.15, block=30, min_n=40, knobs=None):
    """打分 = |r| * (1 + 正领先分钟/120)。|r| 低于 r_min 不算候选，
    免得"滞后 240 分钟但几乎不相关"靠大滞后刷分。"""
    res = []
    anchor = quality_cps[0][0] if quality_cps else None
    for name, series in channels.items():
        lag, r = lead_lag(quality_series, series, block=block, min_n=min_n)
        eff = None
        if anchor is not None:
            lo = max(0, anchor-window); hi = min(len(series), anchor+window)
            before = [v for v in series[lo:anchor] if v is not None]
            after = [v for v in series[anchor:hi] if v is not None]
            if len(before) > 5 and len(after) > 5:
                s = 1.4826*mad([v for v in series if v is not None]) or 1.0
                eff = round((median(after)-median(before))/s, 2)
        below = abs(r) < r_min
        up = upstream_of(name, knobs) if name in READOUTS else []
        res.append({"channel": name, "lead_lag_min": lag, "corr": r,
                    "upstream_knobs": up,
                    "effect_after_anchor": eff,
                    "score": 0.0 if below else round(abs(r)*(1.0 + max(lag, 0)/120.0), 4),
                    "below_threshold": below, "is_readout": name in READOUTS})
    res.sort(key=lambda d: (-d["score"], -abs(d["corr"])))
    for i, d in enumerate(res, 1): d["rank"] = i
    return {"anchor_index": anchor, "candidates": res, "r_min": r_min, "block_min": block,
            "method": "块中位数(%d 分钟) -> 一阶差分 -> 互相关。正滞后=该信号领先质量。" % block}

def recommend(quality_series, driver_series, driver_name, spec, unit="", role="symptom", start=0):
    """线性回归把"质量回到合格线"换算成"这个信号要改多少"。只做量级参考。

    三种角色的话术必须分开，否则报告会自相矛盾：
      knob    —— 实际设定参数（进给、转速）。给"改成多少"。
      symptom —— 量出来的症状（电流、振动）。它自己不可调，它是故障的表现；
                 给的数是"要让质量回来，这个症状得降到什么水平"，对应的是换砂轮/修整这类动作。
      readout —— 环境读数量（温度、湿度）。它变了说明它上游的旋钮变了，去查那个旋钮。
    """
    pairs = [(d, q) for d, q in zip(driver_series[start:], quality_series[start:])
             if d is not None and q is not None]
    base = {"channel": driver_name, "unit": unit, "role": role, "start_index": start}
    if len(pairs) < 50:
        base.update({"direction": "-", "delta": 0.0, "current": 0.0, "target": 0.0,
                     "basis": "变点后有效样本不足（%d 个），不给建议。" % len(pairs)})
        return base
    d = [p[0] for p in pairs]; q = [p[1] for p in pairs]
    b, _ = ols_slope(q, d)
    cur = median(d); q_med = median(q)
    if abs(b) < 1e-9:
        base.update({"direction": "-", "delta": 0.0, "current": round(cur, 3),
                     "target": round(cur, 3),
                     "basis": "回归系数接近 0，这个信号和质量没有可用的线性关系，不给建议。"})
        return base
    delta = (spec - q_med) / b
    capped = abs(delta) > abs(cur)
    if capped: delta = abs(cur) * (1 if delta > 0 else -1)
    head = ("变点后 %d 个采样点线性回归：%s 每变 1 %s，圆度变 %.4f um；"
            "当前圆度中位数 %.2f um，规格 %.2f um。"
            % (len(pairs), driver_name, unit, b, q_med, spec))
    if role == "knob":
        tail = ("这是可调量，可以按上面这个量级去试；但仍是线性外推，一次只动这一个量，动完复测首件。")
    elif role == "symptom":
        tail = ("注意 %s 是量出来的症状，不是可调的旋钮：这个数不是让你去调它，"
                "而是说症状要降到这个水平，质量才会回来——对应的是修整/换砂轮这类动作，"
                "不是改设定值。真正的旋钮（进给、转速、修整量）本批数据里没有记录。" % driver_name)
    else:
        up = upstream_of(driver_name)
        miss = [u["knob"] for u in up if not u["recorded"]]
        tail = ("注意 %s 是环境读数量：它变了说明它上游的某个旋钮变了，"
                "要查的是那个旋钮，不是把 %s 调走。" % (driver_name, driver_name))
        if up:
            tail += ("该去查的旋钮（按可能性排）：%s。"
                     % "、".join(u["knob"] for u in up))
            if miss:
                tail += ("其中 %s 本批数据里没有记录，所以只能给到线索、给不到数值——"
                         "这是下一步要到现场采的第一批参数。" % "、".join(miss))
    base.update({"direction": "降低" if delta < 0 else "提高", "delta": round(abs(delta), 3),
                 "current": round(cur, 3), "target": round(cur + delta, 3),
                 "capped": capped, "slope_per_unit": round(b, 5),
                 "basis": head + tail})
    return base
