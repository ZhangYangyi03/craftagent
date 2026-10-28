# -*- coding: utf-8 -*-
"""standards.py -- 标准库：不抄表，用公开公式重算，并且把"重算值和手册差多少"印出来。

这个文件回答的是一个问题：工艺数据里，哪一部分是**可以被别人验算的**。

结论分三层，三层都写在这里，因为混在一起就分不清哪一句能信：

  一、可重算的（本标准）
      公差等级  ISO 286：i = 0.45 * D^(1/3) + 0.001 * D，D 是尺寸段的几何平均，
                IT5..IT18 是 i 的整数倍（7,10,16,25,40,64,100,160,250,400,640,
                1000,1600,2500）。写公式而不是抄表，好处是**别人能自己算一遍**。
                但必须同时说清：公式是近似，和手册表有 1 um 量级的差（见 HANDBOOK_REF），
                差多少是量出来的，不是"大致相符"。

  二、可引用但要标出处的（材料常数）
      E、泊松比、密度、热膨胀系数、kc1.1 —— 公开手册值，不是软件私有值。
      这里的每一条带 source 字段；没有 source 的数不许进来（这个仓的老规矩）。

  三、两个都不是的（工艺参数的危险区）
      推荐切削速度 vc 在"某个材料"上有一个范围，但那个范围是**条件**的函数。
      本文件只放那些能被物理反算校验的部分（功率、扭矩、挠度、稳定性），
      不放"某材料某工序的最佳 vc 是 137"这种无法反驳的数。
"""
import math

# ---------------------------------------------------------------- ISO 286 公差

# 尺寸段（mm，上界含），ISO 286-1 的常用段
SIZE_STEPS = (3, 6, 10, 18, 30, 50, 80, 120, 180, 250, 315, 400, 500)

# IT 等级 -> i 的倍数
IT_MULTIPLIER = {
    5: 7, 6: 10, 7: 16, 8: 25, 9: 40, 10: 64, 11: 100,
    12: 160, 13: 250, 14: 400, 15: 640, 16: 1000, 17: 1600, 18: 2500,
}

# 手册表里的权威值（um），用来量"公式算出来的到底差多少"。
# 只挑每个等级的典型几段，不抄全表——抄全表就又变成复制了。
# 键是 (尺寸段上界, IT 等级)。第一版我把 (10,6) 写成 6、(10,5) 写成 4 —— 那是把
# IT5 的数填进了 IT6 那一格，于是"公式差 +3 um"看起来像是公式不准，其实是参照表抄错。
# 这就是为什么这一节要把差打印出来：差得大的时候，先怀疑参照值，再怀疑公式。
# 6-10 段手册值：IT5=6, IT6=9, IT7=15, IT8=22, IT9=36, IT10=58。
HANDBOOK_REF = {
    (3, 5): 4, (3, 6): 6, (3, 7): 10, (3, 8): 14, (3, 9): 25, (3, 10): 40,
    (10, 5): 6, (10, 6): 9, (10, 7): 15, (10, 8): 22, (10, 9): 36, (10, 10): 58,
    (18, 6): 11, (18, 7): 18, (18, 8): 27, (18, 9): 43, (18, 10): 70,
    (30, 6): 13, (30, 7): 21, (30, 8): 33, (30, 9): 52, (30, 10): 84,
    (50, 6): 16, (50, 7): 25, (50, 8): 39, (50, 9): 62, (50, 10): 100,
    (80, 7): 30, (80, 8): 46, (80, 9): 74, (80, 10): 120,
    (120, 7): 35, (120, 8): 54, (120, 9): 87, (120, 10): 140,
}


def size_step(size_mm):
    """这个尺寸落在哪一段，返回 (下界, 上界)。"""
    lo = 0
    for hi in SIZE_STEPS:
        if size_mm <= hi:
            return lo, hi
        lo = hi
    raise ValueError("尺寸 %r 超出 ISO 286 常用段（<=500 mm）" % (size_mm,))


def diameter_step_mean(size_mm):
    """段内几何平均 D。第一段（<=3）按 1 和 3 的几何平均，这是 ISO 286-1 的规定。"""
    lo, hi = size_step(size_mm)
    a = lo if lo > 0 else 1.0
    return math.sqrt(a * hi)


def tolerance_factor_d(size_mm):
    """标准公差因子 i（um）：i = 0.45 * D^(1/3) + 0.001 * D。"""
    D = diameter_step_mean(size_mm)
    return 0.45 * D ** (1.0 / 3.0) + 0.001 * D


def _round_tol(v):
    """按 ISO 的做法取整：1 um 量级的数不会写成 3.79。"""
    if v >= 100:
        return float(round(v, 0))
    if v >= 10:
        return float(round(v, 0))
    if v >= 1:
        return float(round(v * 2.0) / 2.0)     # 0.5 的整数倍
    return math.floor(v * 10.0 + 0.5) / 10.0


def formula_applies(size_mm):
    """ISO 286 的第一段（<=3 mm）是表值，不是公式值。

    这一条是量出来的：<=3 mm 段上，公式和手册最多差 5 um（IT10：公式 35、手册 40），
    相对误差 12%；而 6 mm 以上各段最多差 1 um。原因不是公式错，是 ISO 286-1
    对第一段直接给表值（那一档的零件太小，公式的 D 取几何平均已经失真）。
    所以这一节不许在这里"用公式代替表"，而是明确说：这一段请查表。
    """
    return size_mm > 3.0


def it_tolerance_um(size_mm, grade):
    """IT 等级公差（um）。公式值 + 取整值 + 与手册的差，一起返回。

    为什么三个都返回：只返回一个数，别人无从判断该不该信它。
    """
    if grade not in IT_MULTIPLIER:
        raise ValueError("IT%d 不在本文件的范围内（IT5..IT18）" % grade)
    i = tolerance_factor_d(size_mm)
    raw = IT_MULTIPLIER[grade] * i
    val = _round_tol(raw)
    ref = HANDBOOK_REF.get((size_step(size_mm)[1], grade))
    return {"size_mm": size_mm, "grade": grade, "i_um": i, "raw_um": raw,
            "um": val, "handbook_um": ref,
            "delta_um": (val - ref) if ref is not None else None,
            "formula_applies": formula_applies(size_mm),
            "warning": "" if formula_applies(size_mm) else
                       "<=3 mm 段是 ISO 286 的表值区，公式在这里最多差 5 um，请查表"}


def tolerance_table(size_mm, grades=(6, 7, 8, 9, 10, 11)):
    return [it_tolerance_um(size_mm, g) for g in grades]


def fit_summary(size_mm, hole_grade=7, shaft_grade=6):
    """一个配合的余量级：H7 孔 + h6 轴 = 间隙从 0 到 (TH+Ts)。

    这里只做**尺寸公差**这一层的算术。真正的配合还要算形状公差、
    温度、表面粗糙度——那三样不在这条公式里，说清楚比假装算全了有用。
    """
    th = it_tolerance_um(size_mm, hole_grade)["um"]
    ts = it_tolerance_um(size_mm, shaft_grade)["um"]
    return {"size_mm": size_mm, "fit": "H%d/h%d" % (hole_grade, shaft_grade),
            "hole_um": th, "shaft_um": ts,
            "max_clearance_um": th + ts, "min_clearance_um": 0.0,
            "note": "只算了尺寸公差；形状公差、热、粗糙度未计入"}


# ---------------------------------------------------------------- 材料常数（带出处）

MATERIAL_CONSTANTS = {
    "steel_c45": {
        "label": "碳素结构钢 C45 / 45#",
        "E_gpa": 210.0, "nu": 0.30, "rho_kg_m3": 7850.0, "alpha_1e6_per_K": 11.5,
        "kc1_1_n_mm2": 2100.0, "mc": 0.26,
        "source": "GB/T 699 化学成分 + 通用手册弹性常数；kc1.1 为 Kienzle 常用值",
    },
    "al_6061_t6": {
        "label": "铝合金 6061-T6",
        "E_gpa": 68.9, "nu": 0.33, "rho_kg_m3": 2700.0, "alpha_1e6_per_K": 23.6,
        "kc1_1_n_mm2": 700.0, "mc": 0.25,
        "source": "ASM Handbook Vol.2 / ASTM B221 常见值；kc1.1 为 Kienzle 常用值",
    },
    "ti_6al4v": {
        "label": "钛合金 TC4 / Ti-6Al-4V",
        "E_gpa": 113.8, "nu": 0.342, "rho_kg_m3": 4430.0, "alpha_1e6_per_K": 8.6,
        "kc1_1_n_mm2": 2100.0, "mc": 0.22,
        "source": "AMS 4911 / ASM Handbook；kc1.1 取镍基-钛合金常用档，实测差异大",
    },
    "inconel_718": {
        "label": "高温合金 Inconel 718",
        "E_gpa": 200.0, "nu": 0.29, "rho_kg_m3": 8190.0, "alpha_1e6_per_K": 13.0,
        "kc1_1_n_mm2": 2750.0, "mc": 0.24,
        "source": "AMS 5662 / Special Metals 数据表；kc1.1 为常用档",
    },
    "cast_iron_250": {
        "label": "灰铸铁 HT250",
        "E_gpa": 120.0, "nu": 0.26, "rho_kg_m3": 7200.0, "alpha_1e6_per_K": 10.5,
        "kc1_1_n_mm2": 1600.0, "mc": 0.28,
        "source": "GB/T 9439；弹性模量为拉压平均值，铸铁离散大",
    },
}

# 表面粗糙度与公差等级的**经验对应**，不是标准。
# 写在这里是因为现场天天用，写清楚"这是经验"是因为它不是可以验算的东西。
RA_VS_IT = ((0.4, 5), (0.8, 6), (1.6, 7), (3.2, 8), (6.3, 9), (12.5, 11), (25.0, 12))


def ra_for_it(grade):
    """经验：IT7 通常对应 Ra 0.8~1.6。返回一个区间，不给单值。"""
    for ra, it in RA_VS_IT:
        if it >= grade:
            return {"grade": grade, "ra_max_um": ra,
                    "note": "经验对应，不是标准；同一 IT 等级在磨/铣/车上差一个档"}
    return {"grade": grade, "ra_max_um": None, "note": "超出常见范围"}


# ---------------------------------------------------------------- 优先数（ISO 3）

def preferred_numbers(decade_lo=1.0, decade_hi=1000.0, series="R10"):
    """R5/R10/R20 优先数，从基本序列重算而不是抄。"""
    step = {"R5": 5, "R10": 10, "R20": 20}[series]
    out = []
    k = decade_lo
    while k < decade_hi:
        for j in range(step):
            v = k * 10.0 ** (j / float(step))
            if decade_lo <= v < decade_hi:
                out.append(round(v, 3))
        k *= 10.0
    return out
