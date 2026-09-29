"""把机床日志和三坐标报告对到同一条时间轴上。
这一步没有模型，也是整个项目真正难的部分：格式各家一套，时间还不一定对得齐。"""
import csv
from bisect import bisect_left
from datetime import datetime

_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y%m%d%H%M%S")

def parse_ts(s):
    s = str(s).strip()
    for f in _FORMATS:
        try: return datetime.strptime(s, f)
        except ValueError: continue
    raise ValueError("无法解析的时间戳: %r（在 align._FORMATS 里加一条格式）" % s)

def load_machine_log(path, ts_col="ts", id_col="machine_id"):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ts = parse_ts(r[ts_col])
            sig = {}
            for k, v in r.items():
                if k in (ts_col, id_col) or v is None or str(v).strip() == "":
                    continue
                try: sig[k] = float(v)
                except ValueError: pass
            rows.append({"ts": ts, "id": (r.get(id_col, "M01") or "M01").strip(), "sig": sig})
    rows.sort(key=lambda r: r["ts"])
    return rows

def load_cmm(path, ts_col="ts", id_col="machine_id", quality=None):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows: return []
    if quality is None:
        quality = next((c for c in rows[0] if c.endswith("_um")), None) or \
                  next(c for c in rows[0] if c not in (ts_col, id_col, "serial"))
    out = []
    for r in rows:
        try: q = float(r[quality])
        except (KeyError, ValueError): continue
        out.append({"ts": parse_ts(r[ts_col]), "id": (r.get(id_col, "M01") or "M01").strip(),
                    "serial": r.get("serial", ""), "quality": q})
    out.sort(key=lambda r: r["ts"])
    return out, quality

def align(machine, cmm, tol_minutes=15):
    """每个三坐标结果挂上最近时刻的工况；超容差的丢弃并计数。
    用 bisect 而不是逐条 min()，否则 7000 点 x 300 条抽检就是两百万次比较。"""
    if not machine:
        return [], len(cmm)
    mts = [m["ts"] for m in machine]
    aligned = []; dropped = 0; lags = []
    for c in cmm:
        j = bisect_left(mts, c["ts"])
        cand = [k for k in (j - 1, j) if 0 <= k < len(mts)]
        k = min(cand, key=lambda k: abs((mts[k] - c["ts"]).total_seconds()))
        lag = (mts[k] - c["ts"]).total_seconds() / 60.0
        if abs(lag) > tol_minutes:
            dropped += 1; continue
        rec = dict(c); rec["sig"] = machine[k]["sig"]; rec["lag_min"] = round(lag, 2)
        aligned.append(rec); lags.append(lag)
    return aligned, dropped

def interp_to_grid(ts, aligned, tol_minutes=25):
    """把抽检的质量插值到工况的分钟网格上，给变点检测用。
    抽样点是稀疏的，不插值变点检测就没法用；报告里必须标明它是插值出来的。"""
    pts = sorted(((c["ts"], c["quality"]) for c in aligned), key=lambda p: p[0])
    if not pts: return []
    ptt = [p[0] for p in pts]; ptv = [p[1] for p in pts]
    out = []
    for t in ts:
        j = bisect_left(ptt, t)
        if j <= 0:
            v = ptv[0] if (ptt[0] - t).total_seconds() <= tol_minutes * 60 else None
        elif j >= len(pts):
            v = ptv[-1] if (t - ptt[-1]).total_seconds() <= tol_minutes * 60 else None
        else:
            t0, v0 = ptt[j-1], ptv[j-1]; t1, v1 = ptt[j], ptv[j]
            span = (t1 - t0).total_seconds()
            f = 0.0 if span == 0 else (t - t0).total_seconds() / span
            v = v0 + (v1 - v0) * f
        out.append(v)
    return out
