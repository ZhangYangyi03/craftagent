"""工艺档案：把一个工序的物理常识和老师傅经验写成配置，代码不写死任何工艺。
这是整个项目里唯一需要行业知识的地方，也是别人抄不走的地方。"""
import json, os

DEFAULT = {
    "process": "外圆磨削（滚道）",
    "quality": {"channel": "roundness_um", "spec": 8.5, "unit": "um", "direction": "max"},
    "signals": {
        "spindle_current_a": {"role": "symptom", "label": "主轴电流", "unit": "A"},
        "vibration_mm_s":    {"role": "symptom", "label": "振动",     "unit": "mm/s"},
        "temp_c":            {"role": "readout", "label": "主轴温度", "unit": "C"},
        "coolant_temp_c":    {"role": "readout", "label": "冷却液温度", "unit": "C"},
        "ambient_rh":        {"role": "readout", "label": "环境湿度", "unit": "%"},
    },
    "knobs": ["wheel_dress_depth_mm", "feed_rate_mm_min", "spindle_speed_rpm", "coolant_flow_l_min"],
    "syndromes": [
        {"name": "砂轮钝化或堵塞",
         "symptoms": ["spindle_current_a", "vibration_mm_s"],
         "min_corr": 0.5, "lead_min_range": [0, 90], "quality_delay_min": 10,
         "action": "停机金刚石笔修整砂轮（0.02 mm/次），修整后首件复测圆度，再决定要不要动进给",
         "master_note": "老师傅经验：电流和振动一起抬、温度过一会儿才跟涨，十有八九是砂轮钝了。先修砂轮，别先动进给——动了进给就分不清是谁的功劳。"},
        {"name": "冷却不足 / 热变形",
         "symptoms": ["temp_c", "coolant_temp_c"],
         "min_corr": 0.4, "lead_min_range": [0, 180], "quality_delay_min": 5,
         "action": "查冷却液流量与过滤，确认喷嘴没堵；连续加工前保证空转预热到位",
         "master_note": "老师傅经验：一早开机头两小时不能算，机床还没热透。要看趋势得看上午十点以后的。"},
    ],
}

def load(path=None):
    if not path:
        return json.loads(json.dumps(DEFAULT))
    with open(path, encoding="utf-8") as f:
        prof = json.load(f)
    base = json.loads(json.dumps(DEFAULT))
    for k, v in prof.items():
        if k == "signals":
            base["signals"].update(v)
        else:
            base[k] = v
    return base

def roles_of(profile):
    return {name: s.get("role", "symptom") for name, s in profile["signals"].items()}

def label_of(profile, ch):
    s = profile["signals"].get(ch)
    return (s or {}).get("label", ch)

def unit_of_profile(profile, ch):
    s = profile["signals"].get(ch)
    return (s or {}).get("unit", "")

def match_syndrome(profile, cands, quality_cps):
    """按"哪些症状同时动了、动了多早"去匹配档案里的已知症候。
    匹配不上就老实说匹配不上，不要编一个原因。"""
    hits = []
    q_delay = None
    if quality_cps:
        q_idx = quality_cps[0][0]
    for syn in profile.get("syndromes", []):
        want = set(syn["symptoms"])
        got = {c["channel"]: c for c in cands
               if c["channel"] in want and not c["below_threshold"]
               and abs(c["corr"]) >= syn.get("min_corr", 0.0)}
        if len(got) < len(want):
            continue
        lo, hi = syn.get("lead_min_range", [-1e9, 1e9])
        if not all(lo <= got[w]["lead_lag_min"] <= hi for w in want):
            continue
        score = sum(abs(got[w]["corr"]) for w in want) / len(want)
        hits.append({"name": syn["name"], "score": round(score, 3),
                     "evidence": {w: {"lead_lag_min": got[w]["lead_lag_min"],
                                      "corr": got[w]["corr"]} for w in want},
                     "action": syn["action"], "master_note": syn.get("master_note", "")})
    hits.sort(key=lambda h: -h["score"])
    return hits
