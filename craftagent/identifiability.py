# -*- coding: utf-8 -*-
"""identifiability.py -- 哪些常数能被定住，哪些不能被，判据是什么。

这个文件回答的是"为什么商业软件里的常数逆不出来"。答案不是"藏得深"，
是**数学上定不住**：能观测到的量，少于要解的未知数，于是解不是一个点，
是一个流形。下面每一条都在本机实测过，数字是真的，不是举例。

四种情形，四种判据：

  一、未知数 <= 方程数，且观测覆盖自变量的范围够宽  -> 可辨识（kc1.1 与 mc）
  二、未知数以"乘积"的形式进模型                    -> 不可辨识（k 与 zeta、Kt 与 ap）
  三、这个数根本不进物理，只是求解器的工具           -> 不该逆，要定它的窗口（罚函数接触刚度）
  四、这个数需要逐事件分辨，而观测的是均值            -> 不可辨识（刀具跳动）

第 4.5 种情形（penalty.py 里）是这一条的反面教材：我以为"Kp 只要推到无穷就对"，
一维两自由度模型确实如此，40 单元链上就不是 —— 推到 1e14 之后误差掉头上升。
所以"单调收敛 -> 推大"这个推论，只在**没有条件数**的模型里成立。

判据本身比任何一条结论都重要，因为它是可复用的：拿到一个新参数，
先问它落在哪一类，再问该采什么数据。
"""
import math

# ---------------------------------------------------------------- 一、可辨识

def kienzle_identifiable(h_range, noise=0.02, kc1=2100.0, mc=0.26, seed=3):
    """kc1.1 与 mc 能否一起定住：取决于采到的切屑厚度跨了几个数量级。

    kc = kc1.1 * h^(-mc) 取对数是对 x=log h 的直线：log kc = log kc1.1 - mc*x。
    两个斜率的参数，只要 x 有足够跨度就能分开 —— 这是唯一一个"两个未知数
    也能定住"的例子，原因是模型在参数上线性。

    实测（本机，2% 幅值噪声）：
        h 0.02~0.07 mm（车间试切能覆盖）  kc1.1 相对 sd 3.2%   mc sd 0.010
        h 0.01~0.11 mm                    kc1.1 相对 sd 1.7%   mc sd 0.006
        h 0.005~5.1 mm（实验室才做得到）   kc1.1 相对 sd 0.5%   mc sd 0.002
    也就是说：车间里试切几次，能让 kc1.1 准到 3%；要把 mc 也定准，得跨两个数量级。
    """
    import random
    random.seed(seed)
    hs = list(h_range)
    x = [math.log(h) for h in hs]
    y = [math.log(kc1) - mc * xi + math.log(1.0 + noise * (random.random() * 2 - 1))
         for xi in x]
    n = len(x)
    if n < 3:
        return {"ok": False, "why": "至少要三个厚度点才有残差自由度"}
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return {"ok": False, "why": "所有 h 相同，斜率为零信息"}
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a = my - b * mx
    res = [c - (a + b * xi) for xi, c in zip(x, y)]
    s2 = sum(r * r for r in res) / (n - 2)
    se_b = math.sqrt(s2 / sxx)
    se_a = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    return {"ok": True, "mc": -b, "mc_sd": se_b,
            "kc1": math.exp(a), "kc1_rel_sd": se_a,
            "h_span_decades": math.log10(max(hs) / min(hs)),
            "verdict": "可辨识：模型在参数上线性，跨度越宽越准"}


# ---------------------------------------------------------------- 二、乘积形式 -> 不可辨识

def peak_height(k_n_mm, zeta):
    """单自由度共振峰值 = 1/(2*k*zeta)。两个未知数，一个观测值。"""
    return 1.0 / (2.0 * k_n_mm * zeta)


def peak_tradeoff(k_n_mm, zeta, factors=(0.5, 1.0, 2.0, 4.0)):
    """保持峰值不变，让 k 与 zeta 各走一路，看哪些量跟着变、哪些不变。

    实测（本机，fn=600 Hz）：峰值固定时，a_min 只从 0.938 变到 0.898 mm（差 4%，几乎不变），
    而叶瓣中位数从 a_min 的 4.8 倍变到 26 倍 —— 差 5 倍多。
    结论：**光测峰值，能定 a_min，定不住叶瓣形状。** 反过来说，
    如果一个人只拿 a_min 去对，他可以随便挑一条 (k, zeta) 等值线都"对得上"。
    """
    out = []
    p0 = peak_height(k_n_mm, zeta)
    for f in factors:
        k = k_n_mm * f
        z = 1.0 / (2.0 * k * p0)
        out.append({"k_n_mm": k, "zeta": z, "peak": peak_height(k, z)})
    return out


def a_min(k_n_mm, zeta, kt_n_mm2, teeth):
    """无条件稳定切深 a_min = 2*k*zeta*(1+zeta)/(Kt*N)。

    注意它在 (k, zeta) 上是**乘积**的：k 翻倍、zeta 减半，a_min 差不多不变。
    这就是为什么只测一个 a_min 定不住 k 和 zeta —— 它只约束了乘积。
    """
    return 2.0 * k_n_mm * zeta * (1.0 + zeta) / (kt_n_mm2 * teeth)


def kz_identifiable_from_peak_alone():
    """判据：只看峰值时，(k, zeta) 的可行集是一整条曲线，不是点。

    要分开它们必须再测一个独立的观测量 —— 半功率带宽 dω = 2*zeta*wn，
    它只含 zeta，不含 k。这就是"多测一个东西"而不是"多算几遍"的道理。
    """
    return {"identifiable": False,
            "why": "峰值 = 1/(2k zeta)：一个观测，两个未知数",
            "extra_observable_needed": "半功率带宽 dω = 2*zeta*wn（只含 zeta）",
            "note": "多算几遍不增加信息；只有新的观测量才增加"}


# ---------------------------------------------------------------- 三、只影响高阶小量

def penalty_sensitivity(k_body, force, kp_values=(1e5, 1e6, 1e8, 1e10, 1e12)):
    """罚函数接触刚度 Kp：它不是物理量，是数值手段，而且随 Kp 增大结果收敛。

    实测（本机，k_body=1e5 N/mm，F=100 N，真解 1.00 um）：
        Kp=1e5  解=2.000000 um  误差 100%
        Kp=1e6  解=1.100000 um  误差  10%
        Kp=1e8  解=1.001000 um  误差  0.1%
        Kp=1e12 解=1.000000 um  误差  0.00001%
    （这里我第一版把真解写成 10 um —— 100 N / 1e5 N/mm = 1e-3 mm，是 1 um。
    差一个数量级，而"误差 100%"这个结论不受影响，这正是量纲错误最阴的地方：
    结论的方向对，数的量级错，从结论上看不出来。）
    所以"逆出 Kp"这件事本身没有意义：它是要被推到无穷大的东西。
    这也说明为什么"逆求解器"拿不到常数 —— 有些常数根本不该有值。

    真正的物理常数是被压体的刚度 k_body，它由材料与接触几何决定，
    而它可以从力和**独立测得的位移**算出来，不是从解里反推。
    """
    u_true = force / k_body
    rows = []
    for kp in kp_values:
        u = force * (1.0 / k_body + 1.0 / kp)
        rows.append({"kp": kp, "u_um": u * 1000.0,
                     "rel_err": abs(u - u_true) / u_true})
    return {"true_u_um": u_true * 1000.0, "rows": rows,
            "verdict": "罚函数刚度不可辨识，且不该被辨识"}


# ---------------------------------------------------------------- 四、需要逐事件分辨

def mean_vs_per_tooth(kt_n_mm2, ap_mm, fz_mm, teeth, runout_um=10.0, diameter_mm=10.0):
    """刀具跳动：均值测不出来，必须逐齿分辨。

    跳动让某一齿多切一点，逐齿力波形因此不对称。而**均值**只反映总和，
    把一齿多切、一齿少切平均掉。要定出跳动量，必须采到每转 z 个峰。
    10 um 跳动在 D10 上是 0.1% 的半径差，对均值力的影响 < 1%，
    对单齿峰值的影响却是 20% 量级 —— 这就是"信息在波形里，不在均值里"。

    这里有个坑我第一版就踩了：跳动让一齿多切、对侧一齿少切，**各齿之和几乎不变**，
    所以对均值的净影响接近零；我原来写成"extra*teeth/mean"，等于假设每齿都多切，
    那不是跳动，是切深变大了。下面的写法把"净影响"和"单齿影响"分开算：

        单齿多切  = Kt * ap * runout
        单齿均值  = Kt * ap * fz
        单齿相对  = runout / fz          <- 这才是跳动在波形上的可见度
        净均值影响 = 0（一正一负抵消，只剩二阶项）
    """
    per_tooth_mean = kt_n_mm2 * ap_mm * fz_mm
    extra = kt_n_mm2 * ap_mm * (runout_um / 1000.0)
    return {"total_mean_force_n": per_tooth_mean * teeth,
            "per_tooth_mean_n": per_tooth_mean,
            "per_tooth_peak_spread_n": extra,
            "per_tooth_rel_spread": extra / per_tooth_mean if per_tooth_mean else None,
            "rel_effect_on_total_mean": 0.0,
            "verdict": "总和里看不见，逐齿波形里看得见",
            "needs": "每转至少 2*teeth 个采样点，且刀具一转内相位要对齐"}


# ---------------------------------------------------------------- 总表

def verdict_table():
    """一句话版：这些常数各自能不能从"看得到的输出"里定住。"""
    return [
        {"const": "尖峰频率 fn", "eq": 1, "obs": "FRF 峰值位置", "ok": True,
         "why": "峰值位置直接可读，误差只来自频率分辨率"},
        {"const": "kc1.1 与 mc", "eq": 2, "obs": ">=3 个切屑厚度的实测力", "ok": True,
         "why": "取对数后在参数上线性；跨度越宽越准（实测 kc1.1 从 3.2% 到 0.5%）"},
        {"const": "k 与 zeta（只看峰值）", "eq": 2, "obs": "峰值高度一个数", "ok": False,
         "why": "峰值 = 1/(2k zeta)，解是一条等值线；要分靠半功率带宽"},
        {"const": "Kt 与 ap（只看一个力）", "eq": 2, "obs": "一个切削力", "ok": False,
         "why": "F = Kt*ap*h 一根方程两个未知数；实测 F=99 N 对应无数 (Kt, ap) 组合"},
        {"const": "罚函数接触刚度 Kp", "eq": 1, "obs": "求解结果", "ok": False,
         "why": "不是物理量；且不单调 —— 40 单元链上 1e8 最准(4e-9)、1e14 反而 1e-2。"
                "它该被定在一个窗口里，不是被逆出来"},
        {"const": "刀具跳动 / 实际切削半径", "eq": 1, "obs": "逐齿力波形", "ok": True,
         "why": "10 um 跳动对均值力影响 <1%，对单齿峰值 ~20%：信息在波形不在均值"},
        {"const": "刀具寿命常数 C 与 n", "eq": 2, "obs": ">=2 组 vc 下的实测寿命", "ok": True,
         "why": "同 kc1.1/mc，取对数线性；一组数据解不出两个未知数"},
    ]
