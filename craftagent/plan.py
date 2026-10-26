# -*- coding: utf-8 -*-
"""plan.py -- 前瞻 + 拐角限速：把程序变成一张"每段能跑多快"的表。

两条约束分开写，因为它们拒绝的是不同的东西，混在一起就无法说清是谁拦的：

    向心加速度   v <= sqrt(a_max * r)         圆弧，半径 r 是这一段自己的
    拐角偏差     v <= sqrt(a_max * d / (1 - cos))   两段交接处，按转角折算

junction deviation（GRBL 那一套）算的是一个允许的最大拐角偏差 d 下，
过拐角时两段平均方向变化所需的速度上限；它的好处是连续可导、不需要
把每个角分类成"尖角/圆角"。向心加速度那条是圆弧自己的物理，不是拐角，
两者取小。

前瞻是**反推**的：从最后一段往前，后一段的入口速度决定了前一段的出口速度：

    v_out <= sqrt(v_in_next^2 + 2 * a_max * length)

正推（从头往后算"下个弯能不能刹住"）在长程序上对每个弯都要重扫一遍，
这就是环形缓冲前瞻器要在实时板上做的东西；这里的程序是离线整条的，
所以用一遍反推，O(n)，代价是它假定整条程序都已经读进来——这个前提写在
报告里，不许含糊。

限速表只是"许诺"，不是"事实"：它是否被遵守由 gate.py 逐段核对。
"""
import math

from . import gcode as gcode_mod

# 一台中档三轴加工中心，参数写在一个地方，谁要改就改这里
DEFAULT = {
    'a_max_mm_s2': 1500.0,      # 允许的加/减速度
    'jerk_dev_mm': 0.02,        # junction deviation，允许的拐角偏差
    'v_max_mm_s': 200.0,        # 快速移动的机器上限
    'v_feed_max_mm_s': 100.0,   # 进给上限（切削方向不要超过它）
    'r_min_mm': 0.5,            # 刀具能走的最小圆弧半径
}


def _cos_turn(t_in, t_out):
    """两段交接处方向改变角的余弦。等于 1 表示直行，等于 -1 表示原路折返。"""
    c = sum(a * b for a, b in zip(t_in, t_out))
    n1 = math.sqrt(sum(a * a for a in t_in))
    n2 = math.sqrt(sum(b * b for b in t_out))
    if n1 <= 0 or n2 <= 0:
        return 1.0
    return max(-1.0, min(1.0, c / (n1 * n2)))


def corner_speed(t_in, t_out, a_max, dev):
    """过这个交接处允许的最大速度。转角越小，越接近直行，允许越快。"""
    c = _cos_turn(t_in, t_out)
    half = 0.5 * (1.0 - c)                 # = sin^2(转弯半角)
    if half <= 1e-12:
        return None                        # 直行，这一处不设限
    return math.sqrt(a_max * dev / half)


def entry_speed(seg, a_max, dev, v_cap):
    """这一段自己允许的速度：圆弧有向心约束，直线没有。"""
    v = v_cap
    if seg.kind == 'arc' and seg.radius and seg.radius > 0:
        v = min(v, math.sqrt(a_max * seg.radius))
    return v


def plan(segs, params=None, v_start=0.0, v_end=0.0):
    """给每段算 (v_entry, v_peak, v_exit) 和耗时。

    三个数组要分清，否则报告里会出现"计划速度"和"实际速度"两个数对不上的怪事：

        cap[i]      这一段自己允许的速度上限（进给 / 机器 / 向心加速度取小）
        node[i]     第 i 个节点（段落之间）允许通过的速度，i=0 是程序起点
        entry/exit  某一段真正的进出口速度，由两面夹出来

    反推给上限，正推给定值，两边都取小才是这条曲线的真相。
    """
    p = dict(DEFAULT)
    p.update(params or {})
    a = p['a_max_mm_s2']
    dev = p['jerk_dev_mm']
    n = len(segs)
    if n == 0:
        return {'params': p, 'rows': [], 'node': [], 'binding': [],
                't_total_s': 0.0, 'n_segments': 0}

    # 1. 每段自己的速度上限
    cap = []
    for s in segs:
        # G 代码的 F 是 mm/min，这里的每个速度都是 mm/s。单位不换，
        # 报告里就会出现"进给 900 mm/s"这种一眼假但公式全对的数字。
        f = (s.feed / 60.0) if s.feed and s.feed > 0 else p['v_max_mm_s']
        cap.append(entry_speed(s, a, dev, min(f, p['v_max_mm_s'])))

    # 2. 每个节点允许的速度
    node = [None] * (n + 1)
    binding = [None] * (n + 1)
    node[0], binding[0] = v_start, {'rule': 'program_start'}
    node[n], binding[n] = v_end, {'rule': 'program_end'}
    for i in range(1, n):
        v = corner_speed(segs[i - 1].t_out, segs[i].t_in, a, dev)
        c = _cos_turn(segs[i - 1].t_out, segs[i].t_in)
        turn = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        if v is None:
            node[i] = None
            binding[i] = {'rule': 'straight', 'turn_deg': round(turn, 2)}
        else:
            node[i] = v
            binding[i] = {'rule': 'corner', 'turn_deg': round(turn, 2),
                          'prev_line': segs[i - 1].line, 'next_line': segs[i].line}

    def limit_at(i):
        """节点 i 的硬上限：过弯速度、两侧段的自身上限，三者取小。"""
        cands = []
        if node[i] is not None:
            cands.append(node[i])
        if i > 0:
            cands.append(cap[i - 1])
        if i < n:
            cands.append(cap[i])
        return min(cands) if cands else p['v_max_mm_s']

    # 3. 反推：能从下一个节点减速到的最大入口速度
    hi = [0.0] * (n + 1)
    hi[n] = node[n]
    for i in range(n - 1, -1, -1):
        hi[i] = min(limit_at(i), math.sqrt(max(0.0, hi[i + 1] ** 2 + 2.0 * a * segs[i].length)))
    hi[0] = min(hi[0], v_start)

    # 4. 正推：能从上一个节点加速到的速度
    lo = [0.0] * (n + 1)
    lo[0] = hi[0]
    for i in range(n):
        lo[i + 1] = min(hi[i + 1], math.sqrt(max(0.0, lo[i] ** 2 + 2.0 * a * segs[i].length)))

    rows = []
    for i, s in enumerate(segs):
        ve, vx = min(lo[i], hi[i]), min(lo[i + 1], hi[i + 1])
        L = s.length
        # 梯形/三角形速度曲线，加速度给定，求峰值和耗时
        v_cap_here = cap[i]
        vm = math.sqrt(max(0.0, (2.0 * a * L + ve * ve + vx * vx) / 2.0))
        if vm > v_cap_here:                 # 峰值不能超过这一段自己的上限
            vm = v_cap_here
        d_acc = (vm * vm - ve * ve) / (2.0 * a)
        d_dec = (vm * vm - vx * vx) / (2.0 * a)
        if d_acc + d_dec > L:
            scale = L / (d_acc + d_dec)
            d_acc *= scale
            d_dec *= scale
            vm = math.sqrt(max(0.0, ve * ve + 2.0 * a * d_acc))
        d_cru = max(0.0, L - d_acc - d_dec)
        t = (2.0 * d_acc / max(1e-9, ve + vm) if d_acc > 1e-9 else 0.0) + \
            (2.0 * d_dec / max(1e-9, vx + vm) if d_dec > 1e-9 else 0.0) + \
            (d_cru / vm if (d_cru > 1e-9 and vm > 1e-9) else (0.0 if d_cru <= 1e-9 else float('inf')))
        rows.append({'line': s.line, 'kind': s.kind,
                     # G 代码号原样带下去：G0（快速定位）和 G1（切削进给）
                     # 的语义完全不同。把两者压成同一种 'line' 之后，
                     # 质量预测会把空行程也按规格评判，一个健康程序
                     # 就会因为"快速移动太粗"被判报废。
                     'motion': getattr(s, 's', None),
                     'rapid': getattr(s, 's', None) == 0,
                     'length_mm': round(L, 4),
                     'feed': s.feed, 'v_entry': round(ve, 4), 'v_peak': round(vm, 4),
                     'v_exit': round(vx, 4), 't_s': round(t, 6),
                     'd_acc': round(d_acc, 6),
                     'limit_by': (binding[i].get('rule') if binding[i] else None),
                     'seg': s})
    return {'params': p, 'rows': rows, 'node': node, 'binding': binding,
            't_total_s': round(sum(r['t_s'] for r in rows), 6), 'n_segments': n}


def apply(params_or_none=None, **kw):
    p = dict(DEFAULT)
    p.update(params_or_none or {})
    p.update(kw)
    return p
