# -*- coding: utf-8 -*-
"""plant.py -- 虚拟机床：把插补轨迹变成两个物理量，再从物理量算质量。

只有一条物理假设，写在这里而不是藏在代码里：

    每根轴对指令位置是一阶滞后   x_act' = (x_cmd - x_act) / tau

对 x_c = a t^2 / 2 这条指令，它的解析解是

    e(t) = a * tau^2 * ( t/tau - 1 + exp(-t/tau) )        [mm]

两个极限都有意义，而且都必须被说出来，因为它们是两件不同的事：

    加速期间      e ~ a*tau*t        线性增长，长度取决于这一小段跑多久
    匀速之后      e -> a*tau^2 = a/kv^2   稳态，与时间无关

**圆角误差用的是加速期间那一支，不是稳态那支。** 拐角处的加速度只作用
在很短一段上，用稳态值会把误差算小几十倍（实测差 24 倍），而算小的方向
恰好是"看起来一切良好"的方向，这就是它值得单独写一段注释的原因。

这台机床的信号按 craftagent 工艺档案的三类分：
    knob     进给速率、主轴转速（G 代码里本来就有 F 和 S）
    symptom  主轴电流、振动
    readout  冷却液温度、跟随误差
"""
import math

from . import plan as plan_mod

PLANT = {
    # 真实加工中心的位置环带宽在 30~60 Hz 量级，kv = 1/tau 是 190~380 1/s。
    # 原来写 8 ms（kv=125）是一台很软的机床，算出来的误差比规格大几十倍，
    # 那种"什么程序都不合格"的模型没有区分度，也就挡不住任何东西。
    'tau_s': 0.0008,         # 伺服时间常数 -> kv = 1250 1/s（高速加工中心量级）
    'q_base_um': 2.6,        # 无误差时的名义圆度
    # 质量不能是任何一个通道的仿射变换。写成 q = q_base + k*err 时，
    # 跟随误差通道就是质量的精确线性像，它在归因里必然排第一——
    # 那不是"发现了跟随误差是原因"，那是把我自己写的那一行公式念了一遍。
    # 这是循环论证，也正是 craftagent 从头到尾在防的那件事。
    #
    # 所以质量走另一条独立的路径：刀具在拐角处的**实际曲率突变**
    # 造成的残留高度，与段长、进给、加速度都有非线性关系，
    # 跟随误差只是其中一个输入，不是全部。
    # 取 2.6 而不是 4：这台机床在**合规进给**下必须产出合格件（roundness < spec 8.5），
    # 否则整条程序从头到尾都超差，变点检测就失去了'哪里开始变坏'这个问题的前提。
    # 一个什么程序都做不合格的模型，是没有区分度的模型。
    'q_noise_um': 0.22,
    'k_round_um': 1.0,
    'r_eff_mm': 12.0,        # 有效刀尖半径（弓高与过冲都按它算）
    'vib_base_mm_s': 0.35,
    'vib_per_a': 0.0016,
    'cur_base_a': 3.2,
    'cur_per_mm_s': 0.048,
    'cool_base_c': 21.0,
    'seed': 7,
    # 测量噪声。这不是装饰：没有噪声时，振动、电流、跟随误差都是进给速率的
    # **确定性单调函数**，四个通道与质量的相关都等于 1，谁排第一完全由
    # 并列打破规则决定。那种数据上的"归因"没有信息量——它复述了一遍
    # 我写死的公式，而不是从数据里发现关系。
    # 有了噪声，"进给是旋钮、其余是症状"这个层级才在数据里真的存在：
    # 症状带噪声，进给是编码器读数基本不带，于是进给的相关更高、排得更前。
    'vib_noise_rel': 0.06,
    'cur_noise_rel': 0.02,
    'err_noise_rel': 0.04,
    'rng_seed': 4242,
}


def err_accel_phase(a_mm_s2, t_s, tau_s):
    """加速期间的跟随误差（mm），闭式解。t 是这一小段已经走了多久。"""
    if tau_s <= 0 or t_s <= 0:
        return 0.0
    return a_mm_s2 * tau_s * tau_s * (t_s / tau_s - 1.0 + math.exp(-t_s / tau_s))


def err_steady(a_mm_s2, tau_s):
    """加速结束、匀速之后的稳态跟随误差（mm）：e = a/kv^2。"""
    kv = 1.0 / tau_s if tau_s > 0 else 1.0
    return a_mm_s2 / (kv * kv)


def corner_error_um(row, tau_s, a_cap=1500.0):
    """一段走刀在拐角处的跟随误差（um），用它算圆角损失。

    这一段的速度在 v_entry 与 v_peak 之间变化，加速度 a 在段内恒定；
    误差取加速期间闭式解在这一段时长处的值——段短则误差小，段长则逼近稳态。
    线段用的是它自己的段长，圆弧用的是弧长，这一点没有区别：
    伺服只知道"以加速度 a 跑这么久会落后多少"，不知道指令画的是直线还是弧。
    """
    L = row['length_mm']
    if L <= 1e-9 or row['t_s'] <= 0:
        return 0.0
    ve, vp, vx = row['v_entry'], row['v_peak'], row['v_exit']
    a = min(a_cap, (vp * vp - ve * ve) / (2.0 * max(1e-9, L * 0.5)))
    if a <= 1e-9:
        return 0.0
    d_acc = (vp * vp - ve * ve) / (2.0 * a)
    t_acc = 2.0 * d_acc / max(1e-9, (ve + vp)) if d_acc > 1e-12 else 0.0
    t_acc = min(t_acc, row['t_s'])
    if t_acc <= 0.0:
        return 0.0
    # 取这一小段里误差的最大值（t=t_acc 处），不是段末值：
    # 段末如果已经进入匀速段，误差早就收敛到稳态，拿它当拐角误差会漏报。
    return err_accel_phase(a, t_acc, tau_s) * 1000.0


def actual_accel(row, a_cap=1500.0):
    """这一段内真正用到的加速度（mm/s^2）。

    必须从规划器给的 d_acc（加速距离）反算：
        v_peak^2 = v_entry^2 + 2*a*d_acc
    而不是 dv / (t_s/2)。后者把**整段时长**当成加速时长，
    一段 60 mm 的行程加速只占 0.07 mm（0.005 s 中的 0.005 s），
    却被算成 2 秒，加速度被低估了几十倍，于是误差被低估、
    质量被算好——方向上总是"看起来没问题"，这是最坏的一种错。
    """
    d = row.get('d_acc')
    if d is None or d <= 1e-12:
        return 0.0
    ve, vp = row['v_entry'], row['v_peak']
    a = (vp * vp - ve * ve) / (2.0 * d)
    return min(a, a_cap) if a > 0 else 0.0


def machine_pass(rows, plan_params=None, plant=None):
    """按计划表逐段算物理量与质量。返回每次走刀一条记录 = 抽样点。

    确定性：同一个 rng_seed 给同一台机床同样的噪声。
    这一条是必需的，否则"重跑一次数字就变"会让所有验收都失去意义——
    一个每次运行都不同的 bench 只能证明它能运行，不能证明它正确。
    """
    import random
    P = dict(PLANT)
    P.update(plant or {})
    rng = random.Random(P['rng_seed'])
    a_max = (plan_params or plan_mod.DEFAULT)['a_max_mm_s2']
    out = []
    t = 0.0
    for r in rows:
        err = corner_error_um(r, P['tau_s'], a_max)
        # 残留高度：刀具路径在拐角处的弓高 + 伺服滞后造成的过冲，两者非线性叠加
        L = max(1e-6, r['length_mm'])
        scallo = (r['v_peak'] ** 2) / (P['r_eff_mm'] * 8.0)      # 曲率引起的弓高
        over = err * err / (2.0 * P['r_eff_mm'])                 # 伺服过冲的二次项
        q = P['q_base_um'] + P['k_round_um'] * (scallo + over) + 0.35 * err
        a_peak = actual_accel(r, a_max)
        vib = P['vib_base_mm_s'] + P['vib_per_a'] * a_peak * 1000.0
        cur = P['cur_base_a'] + P['cur_per_mm_s'] * r['v_peak']
        vib *= 1.0 + rng.gauss(0.0, P['vib_noise_rel'])
        cur *= 1.0 + rng.gauss(0.0, P['cur_noise_rel'])
        err_m = err * (1.0 + rng.gauss(0.0, P['err_noise_rel']))
        # 快速定位（G0）不切材料，圆度对它没有意义。
        # 让它按 F2000 算出来一个 52 um 的"质量"，只会把一个健康程序
        # 判成报废——而车间里没有人会因为空行程快而说零件不合格。
        cutting = bool(r['feed']) and 0 < r['feed'] <= 6000.0 and r['kind'] in ('line', 'arc')
        is_rapid = r.get('rapid', False) or (r['feed'] or 0) > 6000.0
        out.append({
            'line': r['line'], 'kind': r['kind'],
            'cutting': (not is_rapid) and cutting,
            't_s': round(t + r['t_s'], 6),
            'v_peak': round(r['v_peak'], 4), 'v_entry': round(r['v_entry'], 4),
            'a_peak': round(a_peak, 3),
            'follow_err_um': round(err_m, 4),
            'roundness_um': round(q, 4) if ((not is_rapid) and cutting) else None,
            'vib_mm_s': round(vib, 4),
            'spindle_current_a': round(cur, 4),
            'feed_mm_min': round((r['feed'] or 0.0), 1),
        })
        t += r['t_s']
    return out
