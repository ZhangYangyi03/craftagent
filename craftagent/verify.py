"""拿合成数据里埋的真值做验收：该查出来的查出来，不该报的不报。
没有真值的项目，方法对不对全靠嘴，所以这一步不能省。"""
import json
from . import synth as S

def check(rep, synth_info, tol_min=40):
    """返回 (通过与否, 明细)。三项：漂移机台检出 / 健康机台不误报 / 领先量在容差内。"""
    truth = synth_info["truth"]; mid = truth["machine"]
    res = {"checks": [], "passed": True}
    def add(name, ok, detail):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok: res["passed"] = False

    drift = mid in rep["verdict"]["drifting"]
    add("检出漂移机台", drift, "真值 %s；判定漂移 %s" % (mid, rep["verdict"]["drifting"]))

    others = [m for m in rep["machines"] if m != mid]
    false_alarm = [m for m in others if m in rep["verdict"]["drifting"]]
    add("健康机台不误报", not false_alarm, "健康机台 %s；误报 %s" % (others, false_alarm or "无"))

    m = rep["machines"][mid]
    qcps = m["change_points"].get(rep["quality_channel"]) or []
    if qcps:
        det = qcps[0]["index"]; true = synth_info["drift_index"] + truth["lead_min"]
        add("质量变点位置误差 <= %d 分钟" % tol_min, abs(det - true) <= tol_min,
            "检出 %d，真值 %d（差 %d 分钟，采样 10 分钟一个抽检点）" % (det, true, det - true))
    else:
        add("质量变点位置", False, "没有检出质量变点")

    cps = {k: (v[0]["index"] if v else None) for k, v in m["change_points"].items()
           if k != rep["quality_channel"] and v}
    driver = truth["driver"]
    if driver in cps and qcps:
        lead = qcps[0]["index"] - cps[driver]
        add("驱动信号领先质量（真值 %d 分钟）" % truth["lead_min"], abs(lead - truth["lead_min"]) <= tol_min,
            "检出领先 %d 分钟" % lead)
    best = next((c for c in m["attribution"]["candidates"] if not c["below_threshold"]), None)
    add("驱动信号进入候选且领先量为正", best is not None and best["lead_lag_min"] >= 0,
        "头号候选 %s，领先 %s 分钟，r=%s" % (best["channel"], best["lead_lag_min"], best["corr"]) if best else "无候选")
    return res
