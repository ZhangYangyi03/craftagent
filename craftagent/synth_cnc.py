# -*- coding: utf-8 -*-
"""synth.py -- 造程序，跑虚拟机床，得到和工厂完全同形的两张表。

这一层存在的理由和 craftagent 原有的 synth 一样：**没有真值的方法只能靠嘴**。
所以这里的真值是**造的时候就写死**的：哪一行 G 代码把进给抬了上去、
抬了多少、质量在它之后第几分钟开始走坏。后面 verify 拿这些真值去考自己的方法。

机床模型只有一条物理假设，写在这里而不是藏在代码里：

    每根轴对指令位置是一阶滞后  x_act' = (x_cmd - x_act) / tau

这条假设产生的**跟随误差**就是拐角处的圆角误差的物理来源：速度在拐角突变，
滞后让实际位置切进角里，切进去多少（微米）就是表面质量丢失多少。
它不是拟合出来的，是 tau 和加速度算出来的。

信号分三类，和 craftagent 的工艺档案一致：
    knob     进给倍率、主轴转速（这本就是 G 代码里的 F 和 S）
    symptom  主轴电流、振动
    readout  冷却液温度、跟随误差
"""
import csv
import math
import os
import random
from datetime import datetime, timedelta

from . import gcode as gcode_mod
from . import plan as plan_mod

PLANT = {
    'tau_s': 0.008,            # 伺服时间常数
    'dt_s': 0.001,             # 积分步长
    'k_round_um': 1.0,         # 1 微米跟随误差 -> 1 微米圆角误差（定义，不是拟合）
    'q_base_um': 4.0,          # 无误差时的名义圆度
    'q_noise_um': 0.25,
    'vib_base_mm_s': 0.35,
    'vib_per_a': 0.0022,       # 振动随实际加速度
    'cur_base_a': 3.2,
    'cur_per_mm_s': 0.055,     # 主轴电流随切削速度
    'cool_base_c': 21.0,
    'cool_drift_c_per_h': 0.9,
}

KNOB_MAIN = 'feed_override_pct'


# ---------------------------------------------------------------- 程序生成
def pocket_program(rows=6, cols=6, pitch_mm=12.0, depth_mm=-1.5, feed=900.0,
                   plunge=150.0, rapid=1500.0, corner_feed=None, arc_corners=True,
                   spindle=8000, spike=None):
    """一个矩形槽的往复走刀，拐角处可圆可尖。

    corner_feed=None 表示后处理器没有在拐角降速（这正是真机上最常见的那种
    CAM 输出，也是这个仓要量的那件事）；给一个数就是在拐角降速到它。

    spike=(line_index, factor) 是把这条程序"弄坏"的那一处：某个进给字
    被乘上一个系数。真值就记在这里，不给后面任何模块看。
    """
    L = ['G21 G90', 'G0 Z5.0 F%.0f' % rapid, 'M3 S%d' % spindle,
         'G0 X0 Y0', 'G1 Z%.2f F%.0f' % (depth_mm, plunge)]
    y = 0.0
    for i in range(rows):
        x_end = (cols - 1) * pitch_mm if i % 2 == 0 else 0.0
        f = feed if corner_feed is None else feed
        L.append('G1 X%.3f Y%.3f F%.0f' % (x_end, y, f))
        if i < rows - 1:
            y += pitch_mm
            if arc_corners and corner_feed is not None:
                # 圆角过渡 + 拐角降速：CAM 正常的做法
                L.append('G1 Y%.3f F%.0f' % (y, corner_feed))
            else:
                L.append('G1 Y%.3f F%.0f' % (y, f))
    L.append('G0 Z5.0')
    L.append('M5')
    if spike:
        idx, factor = spike
        body = [ln for ln in L]
        # 在切削段里挑一处改 F
        cut = [i for i, ln in enumerate(body) if ln.startswith('G1 X')]
        i = cut[idx % len(cut)]
        parts = body[i].split()
        body[i] = ' '.join(p if not p.startswith('F') else 'F%.0f' % (float(p[1:]) * factor)
                           for p in parts)
        L = body
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- 机床物理
def simulate(segs, pl, plant=None, t0=0.0):
    """按计划表积分虚拟机床，返回 (每秒一条的信号序列, 每次走刀的质量样本)。

    积分步长 1 ms；每个通道按 1 秒聚合，这样它就是工厂那台机器一分钟一行的
    同一形态，只是这一台是"一秒一行"——因为一条程序的时长就是几十秒。
    """
    P = dict(PLANT)
    P.update(plant or {})
    dt = P['dt_s']
    tau = P['tau_s']
    rows = pl['rows']
    cmd = [0.0, 0.0, 0.0]
    act = [0.0, 0.0, 0.0]
    vel_prev = [0.0, 0.0, 0.0]
    t = t0
    series = []          # 每秒聚合
    bucket = {'t': t0, 'v': 0.0, 'a': 0.0, 'err': 0.0, 'n': 0, 'feed': 0.0, 'spindle': 0.0,
              'pass_max_err': 0.0, 'pass_i': 0}
    passes = []
    cur_pass_max = 0.0
    for row in rows:
        s = row['seg']
        ve, vx = row['v_entry'], row['v_exit']
        L = s.length
        if L <= 1e-9:
            continue
        a = pl['params']['a_max_mm_s2']
        d_acc = (row['v_peak'] ** 2 - ve ** 2) / (2 * a) if row['v_peak'] > ve else 0.0
        d_dec = (row['v_peak'] ** 2 - vx ** 2) / (2 * a) if row['v_peak'] > vx else 0.0
        d_dec = min(d_dec, max(0.0, L - d_acc))
        d_cru = max(0.0, L - d_acc - d_dec)
        n_steps = max(1, int(round(row['t_s'] / dt)))
        # 弧长 -> 位置的解析映射，避免累积误差
        p0, p1 = s.start, s.end
        for k in range(n_steps):
            u = (k + 1) / n_steps
            d = u * L
            # 这一段在弧长 d 处的目标速度
            if d <= d_acc and d_acc > 1e-12:
                v = math.sqrt(max(0.0, ve * ve + 2 * a * d))
            elif d >= L - d_dec and d_dec > 1e-12:
                v = math.sqrt(max(0.0, vx * vx + 2 * a * max(0.0, L - d)))
            else:
                v = row['v_peak']
            frac = d / L
            if s.kind == 'arc':
                ang = s.radius and 0.0
                a0 = math.atan2(s.start[1] - s.centre[1], s.start[0] - s.centre[0])
                sweep = L / s.radius if s.radius else 0.0
                ang = a0 + (sweep * frac if s.ccw else -sweep * frac)
                tgt = [s.centre[0] + s.radius * math.cos(ang),
                       s.centre[1] + s.radius * math.sin(ang),
                       s.start[2] + (s.end[2] - s.start[2]) * frac]
            else:
                tgt = [p0[i] + (p1[i] - p0[i]) * frac for i in range(3)]
            cmd = tgt
            for i in range(3):
                act[i] += (cmd[i] - act[i]) * (dt / tau) if tau > 0 else (cmd[i] - act[i])
            # 实际速度与加速度（有限差分）
            vl = [(act[i] - (vel_prev[i] * 0 + act[i])) for i in range(3)]
            v_now = [(act[i] - (cmd[i] - (cmd[i] - act[i]))) for i in range(3)]  # placeholder
            err = math.sqrt(sum((cmd[i] - act[i]) ** 2 for i in range(3))) * 1000.0  # um
            bucket['err'] = max(bucket['err'], err)
            cur_pass_max = max(cur_pass_max, err)
            bucket['v'] = max(bucket['v'], v)
            bucket['a'] = max(bucket['a'], v * v / max(1e-6, s.radius) if s.kind == 'arc' else 0.0)
            bucket['feed'] = max(bucket['feed'], s.feed or 0.0)
            bucket['spindle'] = max(bucket['spindle'], s.spindle or 0.0)
            bucket['n'] += 1
            bucket['t'] += dt
            if bucket['t'] - bucket['t0'] if 't0' in bucket else False:
                pass
            if bucket['n'] * dt >= 1.0:
                series.append(_flush(bucket, P))
                bucket = {'t': bucket['t'], 'v': 0.0, 'a': 0.0, 'err': 0.0, 'n': 0,
                          'feed': 0.0, 'spindle': 0.0}
        # 一段结束 = 一次走刀结束：记一条质量样本（真机就是一件/一道）
        passes.append({'line': s.line, 't': t0 + sum(r['t_s'] for r in rows[:rows.index(row) + 1]),
                       'max_follow_um': round(cur_pass_max, 3),
                       'q_um': round(P['q_base_um'] + P['k_round_um'] * cur_pass_max, 3)})
        cur_pass_max = 0.0
    if bucket['n'] > 0:
        series.append(_flush(bucket, P))
    return series, passes


def _flush(b, P):
    a_act = b['a']
    return {'t': b['t'],
            'speed_mm_s': round(b['v'], 3),
            'accel_mm_s2': round(a_act, 2),
            'follow_err_um': round(b['err'], 2),
            'feed_cmd_mm_min': round(b['feed'], 1),
            'spindle_cmd_rpm': round(b['spindle'], 0)}

