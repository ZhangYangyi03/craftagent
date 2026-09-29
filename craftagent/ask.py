"""中文问答层。规则优先，答不出来就直说。
本地大模型（GGUF/llama.cpp）是可选增强，不是依赖：工厂那台电脑多半装不上它。"""
import json, os

def _pick(rep, machine, question=""):
    """显式指定 > 问题里提到的机台号 > 在漂的第一台 > 第一台。
    问题里带机台号却不认，就会答成另一台机——这种错最伤信任，所以放在第二优先。"""
    ms = rep["machines"]
    if machine and machine in ms: return machine, ms[machine]
    for mid in ms:
        if mid and mid in (question or ""): return mid, ms[mid]
    d = rep["verdict"]["drifting"]
    if d: return d[0], ms[d[0]]
    k = next(iter(ms)); return k, ms[k]

def _lab(rep, ch):
    sig = rep.get("profile", {}).get("signals", {})
    return (sig.get(ch) or {}).get("label", ch)

def summary_text(rep):
    v = rep["verdict"]
    L = ["规格 %.2f %s；机台 %d 台。" % (rep["spec"], rep["spec_unit"], len(rep["machines"])),
         "在漂的：%s；正常的：%s。" % ("、".join(v["drifting"]) or "无", "、".join(v["stable"]) or "无")]
    for mid, m in rep["machines"].items():
        o = m["out_of_spec"]
        L.append("  %s：超差 %d/%d（%.1f%%）" % (mid, o["count"], o["total"], o["ratio"]*100))
        cps = m["change_points"].get(rep["quality_channel"]) or []
        if cps:
            L.append("     质量变点 %s，效应量 %.1f 倍段内波动" % (cps[0]["ts"], cps[0]["score"]))
            t = m.get("trend") or {}
            if t.get("slope_per_hour"): L.append("     变点后以 %.2f um/小时 上升" % t["slope_per_hour"])
            best = next((c for c in m["attribution"]["candidates"] if not c["below_threshold"]), None)
            if best:
                L.append("     头号嫌疑 %s：领先 %d 分钟，r=%s" % (_lab(rep, best["channel"]),
                         best["lead_lag_min"], best["corr"]))
            syn = (m.get("syndromes") or [])
            if syn: L.append("     匹配症候：%s" % syn[0]["name"])
    return "\n".join(L)

def llm_answer(question, context):
    """有本地 GGUF 就让它润色，没有返回 None，交给规则回答兜底。不联网。"""
    try:
        from llama_cpp import Llama
    except Exception:
        return None
    path = os.environ.get("CRAFTAGENT_GGUF") or \
        r"C:\Users\china\AppData\Local\autoforge\ruzheng-q4.gguf"
    if not os.path.exists(path): return None
    try:
        llm = Llama(model_path=path, n_ctx=2048, verbose=False)
        out = llm("下面是车间数据分析结果，请用一句中文回答工程师的问题。\n数据：%s\n问题：%s\n答："
                  % (json.dumps(context, ensure_ascii=False)[:1500], question), max_tokens=160)
        return out["choices"][0]["text"].strip()
    except Exception:
        return None

def answer(question, rep, machine=None):
    q = (question or "").strip()
    has = lambda *ws: any(w in q for w in ws)
    mid, m = _pick(rep, machine, q)
    o = m["out_of_spec"]; cands = m["attribution"]["candidates"]
    good = [c for c in cands if not c["below_threshold"]]
    cps = m["change_points"].get(rep["quality_channel"]) or []

    if has("哪台", "哪一台", "几台", "所有", "巡检"):
        d = rep["verdict"]["drifting"]
        if not d: return "样本期内没有机台出现超差或质量变点，全部正常。"
        out = ["在漂的是 %s。" % "、".join(d)]
        for k in d:
            b = next((c for c in rep["machines"][k]["attribution"]["candidates"]
                      if not c["below_threshold"]), None)
            if b: out.append("%s 头号嫌疑 %s，领先质量 %d 分钟。" % (k, _lab(rep, b["channel"]), b["lead_lag_min"]))
        return " ".join(out)

    if has("根因", "为什么", "原因", "怎么回事", "谁", "嫌疑", "问题"):
        if not cps: return "%s 没有检出质量变点，没有根因可归。" % mid
        if not good: return "%s 检出了质量变点，但没有哪个信号的相关系数过线（阈值 %s），给不出根因。" % (mid, rep["machines"][mid]["attribution"]["r_min"])
        b = good[0]
        s = ("%s：最可能是 %s —— 领先质量 %d 分钟，互相关 r=%s，变点后变化 %s 倍段内波动。"
             % (mid, _lab(rep, b["channel"]), b["lead_lag_min"], b["corr"], b["effect_after_anchor"]))
        if len(good) > 1:
            n = good[1]
            s += "%s 紧随其后（领先 %d 分钟，r=%s）。" % (_lab(rep, n["channel"]), n["lead_lag_min"], n["corr"])
        syn = m.get("syndromes") or []
        if syn:
            s += "对上档案里的「%s」：%s" % (syn[0]["name"], syn[0]["action"])
        else:
            s += "没有对上档案里的已知症候，别硬猜；把这次的证据补进 profile.json。"
        return s + " 注意这是相关性排序，不是因果证明。"

    if has("超差", "不合格", "超标", "废", "合格率"):
        if o["count"] == 0: return "%s 样本内没有超差件（共 %d 件抽检）。" % (mid, o["total"])
        return ("%s 超差 %d 件，占 %.1f%%；第一件出现在 %s，最集中的时段是 %s。"
                % (mid, o["count"], o["ratio"]*100, o["first_ts"], o["cluster_ts"]))

    if has("正常吗", "有没有问题", "没问题", "为什么正常", "稳不稳"):
        if mid in rep["verdict"]["drifting"]:
            return "%s 在漂，不是正常机台：%s。这台不是正常机台。" % (mid, (m.get("syndromes") or [{}])[0].get("name", "有质量变点"))
        good = [c for c in cands if not c["below_threshold"]]
        return ("%s 被判正常，理由是：抽检 %d 件全部在 %.2f %s 以内，样本期内没有检出质量变点。"
                "同时它最强的相关信号只有 %s（r=%s），低于 %.2f 的候选阈值——"
                "也就是说，连「可疑」都够不上。"
                % (mid, o["total"], rep["spec"], rep["spec_unit"],
                   (_lab(rep, good[0]["channel"]) if good else "无"),
                   (good[0]["corr"] if good else "—"),
                   m["attribution"]["r_min"]))

    if has("怎么办", "建议", "调", "参数", "改", "设定", "拧"):
        if not cps: return "%s 没有质量变点，不给调整建议——没有漂移的地方，调多少都是编的。" % mid
        ad = m.get("adjust")
        if not ad: return "%s 没有可用的候选信号，给不出建议。" % mid
        role = ad.get("role", "symptom")
        if role == "knob":
            return "可调量参考：把 %s %s约 %.2f %s（%s → %s）。%s" % (
                _lab(rep, ad["channel"]), ad["direction"], ad["delta"], ad["unit"],
                ad["current"], ad["target"], ad["basis"])
        if role == "symptom":
            return ("%s 是量出来的症状，不是旋钮，所以不能直接调它。它要降到约 %.2f %s（现在 %.2f），"
                    "质量才会回到 %.2f %s 以内。对应动作是修整/换砂轮这类，不是改设定值。"
                    % (_lab(rep, ad["channel"]), ad["target"], ad["unit"], ad["current"],
                       rep["spec"], rep["spec_unit"]))
        return "%s 是环境读数量，不是旋钮：它动了说明上游的旋钮动了，要查那个旋钮。" % _lab(rep, ad["channel"])

    if has("什么时候", "何时", "开始漂", "时间"):
        if not cps: return "%s 的质量在样本期内没有显著变点。" % mid
        return "%s 的质量变点在 %s（第 %d 分钟那个采样点，效应量 %.1f）。" % (
            mid, cps[0]["ts"], cps[0]["index"], cps[0]["score"])

    if has("多久", "趋势", "走势", "速度", "触线", "还能用"):
        t = m.get("trend") or {}
        if not t: return "%s 没有变点，没有趋势可报。" % mid
        s = "%s 变点后圆度以 %.2f um/小时 上升。" % (mid, t["slope_per_hour"])
        if t.get("already_over"): s += "最近 2 小时中位数 %.2f um，已经越过 %.2f 的规格线。" % (t["last_median"], rep["spec"])
        elif t.get("hours_to_spec"): s += "按这个速度 %.1f 小时后触到 %.2f um。" % (t["hours_to_spec"], rep["spec"])
        if t.get("anchor_band_min"): s += "（数据噪声下，变点本身有 ±%.0f 分钟的定位带。）" % t["anchor_band_min"]
        return s

    if has("方法", "原理", "可靠", "准不准", "怎么算", "凭什么"):
        return "方法：%s 本项目的做法是先埋真值再验证：合成数据里把「电流先动、质量滞后 40 分钟」写死，跑完看方法能不能恢复出来。当前验收结果在 out/verify.json。" % rep["method"]

    if has("数据", "采什么", "要什么"):
        kp = m.get("knob_channels_present") or []
        return ("这批数据的信号：%s。可调旋钮：%s。三坐标抽检 %s。"
                "要让系统给出「参数改成多少」，必须把 CNC 的设定参数也采进来。"
                % ("、".join(_lab(rep, c["channel"]) for c in cands), "、".join(kp) or "没有采到",
                   (m.get("sampling") or {}).get("cmm_every_min", "?")))

    out = llm_answer(q, {"machine": mid, "out_of_spec": o, "candidates": cands[:3],
                         "change_points": cps[:1], "trend": m.get("trend"),
                         "syndromes": m.get("syndromes")})
    return out if out else ("这个问题超出规则回答的范围。可以问：哪台在漂 / 为什么 / 怎么办 / "
                            "什么时候开始 / 趋势如何 / 数据采了什么。\n" + summary_text({"verdict": rep["verdict"],
                            "spec": rep["spec"], "spec_unit": rep["spec_unit"], "machines": {mid: m},
                            "quality_channel": rep["quality_channel"], "profile": rep.get("profile", {})}))
