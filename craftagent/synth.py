"""合成演示数据：两台磨床，M01 砂轮钝化后漂移，M02 健康。
一分钟一个采样点，6 天。日周期按 8 小时班次（开机预热），周期 480 分钟整除 1440，
所以按时钟去周期是成立的。
埋进去的真值：M01 主轴电流与振动同刻起升，质量滞后 40 分钟，温度滞后 10 分钟。
方法若检出"电流领先约 40 分钟"，说明它恢复出了真值。"""
import os, math, random
from datetime import datetime, timedelta

T0 = datetime(2026, 9, 1, 8, 0, 0)
MIN_PER_DAY = 1440
DAYS = 5
N = DAYS * MIN_PER_DAY
DRIFT_AT = 3 * MIN_PER_DAY + 14 * 60 + 20      # 第 4 天 14:20
LEAD_DRIVER = 0
LEAD_TEMP = 10
LEAD_QUALITY = 40
QUAL_DRIFT_PER_H = 0.90
CUR_DRIFT_PER_H = 0.45
VIB_DRIFT_PER_H = 0.18
CMM_EVERY_MIN = 10
CMM_NOISE = 0.15
TRUTH = {"machine": "M01", "driver": "spindle_current_a", "lead_min": LEAD_QUALITY,
         "leads": {"spindle_current_a": 0, "vibration_mm_s": 0, "temp_c": -10,
                   "roundness_um": 40},
         "note": "砂轮钝化：主轴电流与振动同步上升，温度 10 分钟后跟涨，圆度 40 分钟后超差。"}

def _series(mid, i, rng):
    phase = (i % 480) / 480.0                  # 8 小时班次，整除 1440
    warm = 1.0 - math.exp(-phase * 16)
    cur = 12.0 + 0.60 * warm + rng.gauss(0, 0.09)
    vib = 1.80 + 0.20 * warm + rng.gauss(0, 0.045)
    tmp = 24.0 + 3.00 * warm + rng.gauss(0, 0.14)
    cool = 20.0 + 0.80 * warm + rng.gauss(0, 0.18)
    rh = 45.0 + rng.gauss(0, 1.10)
    q = 6.00 + 0.40 * warm + rng.gauss(0, 0.10)
    if mid == "M01" and i >= DRIFT_AT:
        k = (i - DRIFT_AT) / 60.0
        cur += CUR_DRIFT_PER_H * k
        vib += VIB_DRIFT_PER_H * k
        if i >= DRIFT_AT + LEAD_TEMP:
            tmp += 0.25 * ((i - DRIFT_AT - LEAD_TEMP) / 60.0)
        if i >= DRIFT_AT + LEAD_QUALITY:
            q += QUAL_DRIFT_PER_H * ((i - DRIFT_AT - LEAD_QUALITY) / 60.0)
    return cur, vib, tmp, cool, rh, q

def write(outdir="data", seed=7):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(seed)
    mp = os.path.join(outdir, "machine_log.csv")
    cp = os.path.join(outdir, "cmm_report.csv")
    nm = nc = 0
    with open(mp, "w", encoding="utf-8", newline="") as f, \
         open(cp, "w", encoding="utf-8", newline="") as g:
        f.write("ts,machine_id,spindle_current_a,vibration_mm_s,temp_c,coolant_temp_c,ambient_rh\n")
        g.write("ts,machine_id,serial,roundness_um\n")
        for mid in ("M01", "M02"):
            sn = 0
            for i in range(N):
                ts = T0 + timedelta(minutes=i)
                cur, vib, tmp, cool, rh, q = _series(mid, i, rng)
                f.write("%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n"
                        % (ts.isoformat(), mid, cur, vib, tmp, cool, rh))
                nm += 1
                if i % CMM_EVERY_MIN == 0:
                    sn += 1
                    g.write("%s,%s,%s%05d,%.3f\n"
                            % ((ts + timedelta(minutes=3)).isoformat(), mid, mid, sn,
                               q + rng.gauss(0, CMM_NOISE)))
                    nc += 1
    return {"machine": nm, "cmm": nc, "machine_path": mp, "cmm_path": cp,
            "truth": TRUTH, "drift_start": (T0 + timedelta(minutes=DRIFT_AT)).isoformat(sep=" "),
            "drift_index": DRIFT_AT, "cmm_every_min": CMM_EVERY_MIN, "cmm_noise": CMM_NOISE}
