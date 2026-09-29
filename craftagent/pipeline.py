"""把一个机台的一批数据跑成一份报告。"""
from .stats import median, stdev, deseasonalize, ols_slope, unit_of, scale_of
from .changepoint import find_changes
from .align import load_machine_log, load_cmm, align, interp_to_grid
from .attribute import attribute, recommend, READOUTS
from . import profile as profile_mod

def busiest_hour(items, key="ts"):
    if not items: return None
    b = {}
    for it in items:
        t = it[key]; b[(t.date(), t.hour)] = b.get((t.date(), t.hour), 0) + 1
    (d, h), n = max(b.items(), key=lambda kv: kv[1])
    return "%s %02d:00-%02d:00（%d 件）" % (d, h, h + 1, n)

def analyse_machine(rows, cmm, spec, min_size=240, window=180, knobs=None,
                    quality_name="roundness_um", block=30, prof=None):
    ts = [r["ts"] for r in rows]
    sig_names = sorted(rows[0]["sig"].keys())
    channels = {k: [r["sig"].get(k) for r in rows] for k in sig_names}
    aligned, dropped = align(rows, cmm)
    q_grid = interp_to_grid(ts, aligned)

    bad = [c for c in cmm if c["quality"] > spec]
    oos = {"count": len(bad), "total": len(cmm),
           "ratio": round(len(bad) / max(1, len(cmm)), 4),
           "first_ts": bad[0]["ts"].isoformat(sep=" ") if bad else None,
           "cluster_ts": busiest_hour(bad) if bad else None,
           "spec": spec, "unit": "um"}

    resid = {k: deseasonalize(ts, [v if v is not None else 0.0 for v in s])
             for k, s in channels.items()}
    resid_q = deseasonalize(ts, [v if v is not None else 0.0 for v in q_grid])
    cps = {k: find_changes(v, min_size=min_size) for k, v in resid.items()}
    q_cps = find_changes(resid_q, min_size=min_size)

    def out(lst):
        return [{"index": i, "ts": ts[i].isoformat(sep=" "), "score": s} for i, s in lst]
    cp_report = {k: out(v) for k, v in cps.items()}
    cp_report[quality_name] = out(q_cps)

    attr = attribute(resid, resid_q, q_cps, window=window, block=block)
    for c in attr["candidates"]:
        c["unit"] = unit_of(c["channel"])

    trend = {}
    if q_cps:
        a = q_cps[0][0]
        ys = [v for v in q_grid[a:] if v is not None]
        xs = [i / 60.0 for i in range(len(ys))]
        slope, intercept = ols_slope(ys, xs)
        res_std = stdev([ys[i] - (intercept + slope * xs[i]) for i in range(len(ys))])
        last = median(ys[-120:]) if len(ys) >= 120 else (ys[-1] if ys else 0.0)
        anchor_band_min = (round(abs(res_std / slope) * 60, 1)
                           if abs(slope) > 1e-9 else None)
        trend = {"anchor_ts": ts[a].isoformat(sep=" "),
                 "slope_per_hour": round(slope, 3),
                 "last_median": round(last, 3),
                 "resid_std": round(res_std, 3),
                 "hours_to_spec": (round((spec - last) / slope, 2) if slope > 1e-9 else None),
                 "already_over": last > spec,
                 "anchor_band_min": anchor_band_min}

    drv = next((c for c in attr["candidates"] if c["rank"] == 1 and not c["below_threshold"]), None)
    start = q_cps[0][0] if q_cps else 0
    # 没有质量变点就不给调整建议：没有漂移的地方，"怎么调"是编出来的。
    role = (prof or {}).get("signals", {}).get(drv["channel"], {}).get("role", "symptom") if drv else None
    adjust = (recommend(q_grid, channels.get(drv["channel"], []), drv["channel"], spec,
                        unit_of(drv["channel"]), role=role, start=start)
              if (drv and q_cps) else None)
    syn = profile_mod.match_syndrome(prof, attr["candidates"], q_cps) if q_cps else []

    return {"machine_id": rows[0]["id"], "n_machine": len(rows), "n_cmm": len(cmm),
            "aligned": len(aligned), "dropped": dropped, "out_of_spec": oos,
            "change_points": cp_report, "attribution": attr, "trend": trend,
            "adjust": adjust, "syndromes": syn, "spec": spec,
            "knob_channels_present": [k for k, v in (prof or {}).get("signals", {}).items()
                                       if v.get("role") == "knob" and k in channels],
            "sampling": {"cmm_every_min": (round(len(ts) / len(aligned), 1) if aligned else None)}}

def run(machine_path, cmm_path, quality=None, spec=8.5, min_size=240,
        knobs=None, block=30, profile=None):
    prof = profile_mod.load(profile)
    machine = load_machine_log(machine_path)
    cmm, qname = load_cmm(cmm_path, quality=quality)
    if spec is None: spec = prof["quality"]["spec"]
    knobs = knobs if knobs is not None else prof.get("knobs")
    ids = sorted({r["id"] for r in machine} | {c["id"] for c in cmm})
    per, drifting, stable = {}, [], []
    for mid in ids:
        rows = [r for r in machine if r["id"] == mid]
        cs = [c for c in cmm if c["id"] == mid]
        if not rows: continue
        ms = min_size if len(rows) > 4 * min_size else max(60, len(rows) // 8)
        rep = analyse_machine(rows, cs, spec, min_size=ms, window=180, knobs=knobs,
                              quality_name=qname, block=block, prof=prof)
        per[mid] = rep
        if rep["out_of_spec"]["count"] > 0 or (rep["change_points"].get(qname) or []):
            drifting.append(mid)
        else:
            stable.append(mid)
    return {"spec": spec, "spec_unit": "um", "quality_channel": qname,
            "machine_log": machine_path, "cmm_report": cmm_path,
            "machines": per, "verdict": {"drifting": drifting, "stable": stable},
            "profile": prof, "knobs": list(knobs) if knobs else [c for c in
                     sorted({c for m in per.values() for c in [x["channel"] for x in m["attribution"]["candidates"]]})
                     if c not in READOUTS],
            "readouts": [c for c in READOUTS],
            "method": ("去班次日周期 -> 中位数/MAD 标准化 -> CUSUM 二分变点 -> 两段均值精修 -> "
                       "差分互相关定领先滞后 -> 线性回归换算调整量。相关性排序，不是因果证明。"),
            "ground_truth": None}
