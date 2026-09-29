"""现场跟岗：师傅指认一次，机器同步记一次。

用法上就三件事：
  stamp   师傅在机边指"这台不对"的那一刻，把当时的读数快照 + 他的原话记下来
  fill    回到办公室把六要素补全（判断/动作/边界/结果）
  grade   拿机器算出来的东西，去核对师傅说的话

stamp 的要点是"同一时刻"：师傅说"就现在，声音发闷"的那一刻，信号是什么值，
以后才有得比。事后回忆填 TS 是最容易把方法验歪的地方。
"""
import json, os
from datetime import datetime
from . import annotation as A

def snapshot(rep, machine, tail_min=180):
    """把最近 tail_min 分钟的读数取中位数，作为"师傅指认那一刻"的信号快照。"""
    m = (rep.get("machines") or {}).get(machine)
    if not m:
        return {"error": "报告里没有 %s" % machine}
    out = {}
    for c in m["attribution"]["candidates"]:
        out[c["channel"]] = {"corr_vs_quality": c["corr"], "lead_lag_min": c["lead_lag_min"],
                             "effect_after_anchor": c["effect_after_anchor"],
                             "is_readout": c["is_readout"], "rank": c["rank"]}
    return {"anchor_ts": m["trend"].get("anchor_ts"), "quality_signal": out}

def stamp(machine, say, rep_path, out_dir="annotations", ts=None, master="", extra=None):
    """一条 A 记录：情境+线索先落地，判断/动作/边界等师傅说完再补。"""
    rec = A.blank(machine=machine)
    now = ts or datetime.now().isoformat(sep=" ", timespec="seconds")
    rec["situation"]["ts_start"] = now
    rec["clues"]["observed"] = say
    rec["source"]["master"] = master
    rep = json.load(open(rep_path, encoding="utf-8")) if os.path.exists(rep_path) else {}
    rec["clues"]["observed_reading_at_stamp"] = snapshot(rep, machine) if rep else None
    if extra:
        for k, v in extra.items():
            rec[k] = v
    p = os.path.join(out_dir, "annotations.jsonl")
    bad = A.append(rec, p)
    return {"id": rec["id"], "path": p, "open_items": bad}
