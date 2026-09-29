"""评分器：craftagent 的方法，能不能恢复师傅的判断？

这是这个项目真正的产品。目标不是"十个师傅的经验"，而是一套可复现的标注方法：
师傅指认一次，机器同步记一次，然后拿机器算的东西去核对师傅说的话。
分数低的地方就是方法该改的地方——而不是师傅记错了。

五问，每一问都能被证伪：
  1 人 blame 的那个通道，机器有没有把它排进候选（前 N）？  —— 名次
  2 人说的"从这时起不对"，机器检出的变点落在他的窗口里吗？ —— 窗口命中
  3 人预期的领先量，机器测出来差多少？                      —— 领先量误差
  4 人对上的症候，机器匹配到的是不是同一个？                —— 症候一致
  5 人声明的边界，分析有没有守（预热段有没有被排除）？       —— 边界遵守
"""
import datetime as _dt
from .stats import median

def _cand_rank(cands, channel):
    for c in cands:
        if c["channel"] == channel:
            return c
    return None

def _in_window(ts, a, b):
    if not ts or not a or not b: return None
    try:
        t = str(ts)[:19]; a = str(a)[:19]; b = str(b)[:19]
        return a <= t <= b
    except Exception:
        return None

def grade(rep, rec, machine=None, topn=3, lead_tol_min=40, masked=None):
    """masked: 一份按 boundary.invalid_when 裁过段重算的报告；给了才判"边界遵守"这一问。"""
    """rep 是 pipeline 的产出，rec 是一条标注。返回可证伪的五问明细。"""
    mid = machine or (rec.get("situation") or {}).get("machine")
    if not mid or mid not in rep["machines"]:
        return {"ok": False, "why": "报告里没有机台 %s；是不是跑错数据了" % mid, "questions": []}
    m = rep["machines"][mid]
    j = rec.get("judgement") or {}
    b = rec.get("boundary") or {}
    ch = j.get("blamed_channel")

    qs = []
    def add(name, ok, detail):
        qs.append({"name": name, "ok": (None if ok is None else bool(ok)), "detail": detail})

    cands = [c for c in m["attribution"]["candidates"] if not c["below_threshold"]]
    c = _cand_rank(cands, ch) if ch else None
    if not ch:
        add("通道名次", False, "标注里没有 blamed_channel，没法核对")
    elif c is None:
        add("通道名次", False, "师傅指的是 %s，机器没把它排进候选（|r| 低于 %s）——要么方法漏了，要么这个通道没采"
            % (ch, m["attribution"]["r_min"]))
    else:
        add("通道名次", c["rank"] <= topn,
            "师傅指的是 %s，机器排第 %d（前 %d 算命中）；r=%s，领先 %+d 分钟"
            % (ch, c["rank"], topn, c["corr"], c["lead_lag_min"]))

    qcps = m["change_points"].get(rep["quality_channel"]) or []
    hit = None
    if qcps and (j.get("window_start") or j.get("window_end")):
        hit = _in_window(qcps[0]["ts"], j["window_start"] or "0000", j["window_end"] or "9999")
        add("变点落在师傅的窗口里", hit,
            "师傅说 %s ~ %s 开始不对；机器检出 %s" % (j.get("window_start"), j.get("window_end"), qcps[0]["ts"]))
    elif qcps:
        add("变点位置", None, "师傅没给窗口，跳过；机器检出 %s（效应量 %.1f）" % (qcps[0]["ts"], qcps[0]["score"]))
    else:
        add("变点位置", False, "机器在 %s 上没检出变点，而师傅认为有——方法漏报" % rep["quality_channel"])

    lead_true = b.get("leads_expected_min")
    if lead_true is not None and c is not None:
        err = c["lead_lag_min"] - lead_true
        add("领先量误差 <= %d 分钟" % lead_tol_min, abs(err) <= lead_tol_min,
            "师傅预期 %+d 分钟；机器测出 %+d 分钟（差 %+d）" % (lead_true, c["lead_lag_min"], err))
    elif lead_true is not None:
        add("领先量误差", False, "师傅给了预期领先量，但机器没把该通道排进候选，无从比较")
    else:
        add("领先量", None, "师傅没给预期领先量，跳过")

    syn_said = (j.get("syndrome") or "").strip()
    syn_got = (m.get("syndromes") or [{}])[0].get("name") if m.get("syndromes") else None
    if syn_said:
        add("症候一致", syn_said == syn_got, "师傅说 %s；机器匹配 %s" % (syn_said, syn_got or "（没有匹配）"))
    else:
        add("症候", None, "师傅没填症候名，跳过；机器匹配 %s" % (syn_got or "（无）"))

    warm = [w for w in (b.get("invalid_when") or []) if ("开机" in w or "预热" in w or "换批" in w or "首批" in w)]
    if warm:
        if masked is None:
            add("边界遵守（预热/换批段）", None,
                "师傅声明这些条件下不成立：%s。要判这一问，得按它裁段重算一遍（--auto-mask）。" % "、".join(warm))
        else:
            mm = masked["machines"].get(mid) or {}
            mc = [x for x in (mm.get("attribution") or {}).get("candidates", []) if not x["below_threshold"]]
            rank2 = next((x for x in mc if x["channel"] == ch), None)
            top1_before = cands[0]["channel"] if cands else None
            top1_after = mc[0]["channel"] if mc else None
            syn2 = (mm.get("syndromes") or [{}])[0].get("name") if mm.get("syndromes") else None
            same_top = (top1_before == top1_after)
            same_syn = (syn_got == syn2)
            # 判据用"结论有没有变"，不用"领先量差多少"：田间没有领先量的真值，
            # 师傅说的那个数本身就是被测对象，拿它当尺子会把自己量歪。
            add("边界遵守（预热/换批段）", same_top and same_syn,
                "守住师傅说的「不算」段（每班次丢 %d 分钟）重算：头号嫌疑 %s -> %s，症候 %s -> %s。%s"
                % (masked.get("warmup_skip_min") or 0, top1_before, top1_after,
                   syn_got or "（无）", syn2 or "（无）",
                   "结论没变——这条边界对本次判断不构成影响，但仍要写进档案，换批/夜班场景用得上。"
                   if (same_top and same_syn) else
                   "结论变了——这条边界是必须守的，之前那份报告不裁段就是错的。"))
            if rank2 is not None and c is not None:
                qs.append({"name": "边界前后领先量对照", "ok": None,
                           "detail": "%s 的领先量 %+d -> %+d 分钟（差 %d）。%s"
                           % (ch, c["lead_lag_min"], rank2["lead_lag_min"],
                              abs(rank2["lead_lag_min"] - c["lead_lag_min"]),
                              ("有合成真值可对：真值 %+d 分钟，裁段后那份更准。"
                               % (rep.get("ground_truth") or {}).get("lead_min")
                               if (rep.get("ground_truth") or {}).get("lead_min") is not None else
                               "田间没有领先量的真值，两个数都留着，别挑好看的那个。"))})
    else:
        add("边界", None, "师傅没声明边界（这正是要追问的地方：什么条件下你会改判）")

    scored = [q for q in qs if q["ok"] is not None]
    return {"ok": True, "machine": mid, "record": rec.get("id"),
            "questions": qs, "scored": len(scored),
            "passed": sum(1 for q in scored if q["ok"]),
            "note": "没答的一律算 None，不冒充通过。分数低的地方先怀疑方法，再怀疑师傅。"}

def grade_all(rep, recs, masked=None, **kw):
    rs = [grade(rep, r, masked=masked, **kw) for r in recs]
    tot = sum(r.get("passed", 0) for r in rs); den = sum(r.get("scored", 0) for r in rs)
    return {"records": rs, "passed": tot, "scored": den,
            "rate": round(tot/den, 3) if den else None,
            "note": "这是方法对师傅判断的恢复率。分母只算能被证伪的问项。"}
