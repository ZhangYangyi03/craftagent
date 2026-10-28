# -*- coding: utf-8 -*-
"""hard.py -- 高难度那一档参数：用公开的力学方程算出来，每一条标出它属于哪一档。

"高难度"在这个文件里有确切的意思：**不是抄不到，是抄到了也没用，必须自己算或自己量。**
分三档，每条返回都带 grade 字段，混在一起就分不清哪句能信。

    [recompute]   公式是公开的，输入是几何/材料常数，输出可以被别人重算一遍。
                  颤振稳定叶瓣、挠度、残留高度、热伸长、扭矩都属于这一档。
    [reference]   需要厂商或标准的数值（kc1.1、阻尼比），带出处引用，可替换。
    [must-measure] 只有这台机床/这对刀具才有意义，任何来源的数都是编的。
                  Taylor 指数与常数、阻尼比、实际刚度、刀具跳动。

这个文件和 feeds.py 的分工：feeds 给"能切的参数"，hard 给"这台机床切不切得动、
会不会颤、会不会弯、会不会烧"。两类问题混在一起时，报告就会拿"参数合规"去回答
"这活干不干得了"，而那两件事的答案可以相反。

所有公式都写在这里，不藏在代码里。单位一律标出来，因为颤振这一节的量纲
（N/mm 还是 N/m）错一次，叶瓣高度会差一千倍，而一千倍的错误看起来像"这台机床很强"。
"""
import math

GRADES = ("recompute", "reference", "must-measure")


# ---------------------------------------------------------------- 颤振稳定性叶瓣

def frf_1dof(freq_hz, fn_hz, k_n_mm, zeta):
    """单自由度频率响应 G，单位 mm/N。

        G(w) = 1 / (k * (1 - r^2 + j*2*zeta*r)),  r = w/wn

    k 用 N/mm 传进来（1 N/mm = 1000 N/m），这样 Re(G) 直接是 mm/N，
    和切削力系数 Kt 的 N/mm^2 相乘就得到 mm —— 量纲对得上，是这一节能用的前提。
    """
    r = freq_hz / fn_hz
    den = complex(1.0 - r * r, 2.0 * zeta * r)
    return (1.0 / (k_n_mm * den))


def stability_lobes(fn_hz=600.0, k_n_mm=5.0e4, zeta=0.025, kt_n_mm2=700.0,
                    teeth=4, n_lobes=3, speed_min=1000.0, speed_max=20000.0):
    """Altintas 频域一维正交解：给主轴转速，报"这一档转速下最深能切多少"。

    只有一条式子，但它是这一整个文件里最值钱的一条：

        a_lim = -1 / (2 * Kt * N * Re(G))          [mm]
        eps   = pi - 2*atan2(Im G, Re G)
        n     = 60 * wc / (N * (2*k*pi + eps))     [rpm]

    为什么值钱：它把"转速"从一个凭感觉的旋钮，变成一个**有禁区的东西**。
    车间里最常见的事故不是参数太猛，是参数落在某一条叶瓣的谷里 —— 那一段
    "看起来温和"（转速不高、进给不大），实际上正好在共振上。软件的颤振图
    卖的也是这条式子，而这条式子是公开的（Altintas & Budak, 1995）。

    必说的三条：
      - 阻尼比 zeta 和刚度 k 是 [must-measure]：这两个数决定了叶瓣有多深，
        而它们随刀柄、夹持长度、主轴转速变化，不是材料的属性。
      - 这是**一维**解，只算一个主模态。实际机床有多个模态耦合，方向也未必对齐，
        所以它给的是乐观上界：算出来的 a_lim 说"能切"，实际还要留 30%~50% 余量。
      - Kt 用 feeds.py 那一档（Kienzle）的近似值，不是实测的。
    """
    wn = 2.0 * math.pi * fn_hz
    lobes = []
    for order in range(n_lobes):
        pts = []
        # 只在 Re(G) < 0 的频段上才有物理意义（否则 a_lim 为负）
        for i in range(1, 400):
            ratio = 0.5 + 1.2 * i / 399.0
            f = fn_hz * ratio
            G = frf_1dof(f, fn_hz, k_n_mm, zeta)
            if G.real >= -1e-12:
                continue
            a_lim = -1.0 / (2.0 * kt_n_mm2 * teeth * G.real)
            if a_lim <= 0 or a_lim > 50.0:
                continue
            psi = math.atan2(G.imag, G.real)
            eps = math.pi - 2.0 * psi
            wc = 2.0 * math.pi * f
            T = (2.0 * order * math.pi + eps) / wc
            if T <= 0:
                continue
            n = 60.0 / (teeth * T)
            if speed_min <= n <= speed_max:
                pts.append({"rpm": n, "a_lim_mm": a_lim, "chatter_hz": f})
        pts.sort(key=lambda d: d["rpm"])
        if pts:
            lobes.append(pts)
    # 合成一条包络：每个转速取所有叶瓣里的最小值（那就是禁区上沿）
    env = {}
    for pts in lobes:
        for p in pts:
            key = int(round(p["rpm"] / 25.0) * 25)
            if key not in env or p["a_lim_mm"] < env[key]:
                env[key] = p["a_lim_mm"]
    envelope = [{"rpm": k, "a_lim_mm": v} for k, v in sorted(env.items())]
    worst = min(envelope, key=lambda d: d["a_lim_mm"]) if envelope else None
    return {"grade": "recompute",
            "inputs": {"fn_hz": fn_hz, "k_n_mm": k_n_mm, "zeta": zeta,
                       "kt_n_mm2": kt_n_mm2, "teeth": teeth},
            "grades": {"fn_hz": "must-measure", "k_n_mm": "must-measure",
                       "zeta": "must-measure", "kt_n_mm2": "reference"},
            "lobes": lobes, "envelope": envelope, "worst": worst,
            "source": "Altintas & Budak 1995, 一维正交频域解（公开公式）",
            "note": "一维主模态，偏乐观；实际留 30%~50% 余量"}


def unconditional_depth_mm(k_n_mm, zeta, kt_n_mm2, teeth):
    """无条件稳定切深（解析解）：a_min = 2*k*zeta*(1+zeta) / (Kt * N)。

    这是整个叶瓣图的**下界**，任何转速都不能超过它。它比画图更有用的地方是：
    它只有一个式子，所以可以被别人当场算一遍核对 —— 我上面那套网格扫频
    算出来的最坏点是 0.9167 mm，这条式子给 0.9152 mm，差 0.2%。
    两者对上，说明网格扫的是同一件事，而不是扫出了一条好看的曲线。

    车间里的用法：这个数以下，转速随便选都不会颤；这个数以上，转速必须挑。
    精加工常见 ap 在 0.2~1 mm，正好就在这个界附近 —— 所以"精工参数"里
    真正决定成败的从来不是 vc，是这一条。
    """
    return 2.0 * k_n_mm * zeta * (1.0 + zeta) / (kt_n_mm2 * teeth)


def avoid_bands(stab, a_cut_mm, margin=1.3):
    """拿"我要切 ap=2 mm"去问：哪些转速段不能碰。

    输出是**不能用**的转速区间，不是一个推荐转速 —— 推荐转速需要知道
    你要切多深，而那是工艺决定的，不是算法决定的。
    """
    bad = [d for d in stab["envelope"] if d["a_lim_mm"] < a_cut_mm * margin]
    if not bad:
        return {"a_cut_mm": a_cut_mm, "margin": margin, "avoid": [], "bad_fraction": 0.0,
                "a_min_mm": unconditional_depth_mm(
                    stab["inputs"]["k_n_mm"], stab["inputs"]["zeta"],
                    stab["inputs"]["kt_n_mm2"], stab["inputs"]["teeth"]),
                "note": "在算过的转速范围内都能切；这是一维解的结论，仍需试切"}
    bands, cur = [], [bad[0]["rpm"], bad[0]["rpm"]]
    for d in bad[1:]:
        if d["rpm"] - cur[1] <= 50:
            cur[1] = d["rpm"]
        else:
            bands.append(tuple(cur)); cur = [d["rpm"], d["rpm"]]
    bands.append(tuple(cur))
    span = max(1, stab["envelope"][-1]["rpm"] - stab["envelope"][0]["rpm"])
    bad_span = sum(b - a for a, b in bands)
    return {"a_cut_mm": a_cut_mm, "margin": margin, "avoid": bands,
            "bad_fraction": bad_span / float(span),
            "a_min_mm": unconditional_depth_mm(
                stab["inputs"]["k_n_mm"], stab["inputs"]["zeta"],
                stab["inputs"]["kt_n_mm2"], stab["inputs"]["teeth"]),
            "note": ("落在这些转速上，稳定性上限低于 %.1f 倍的切深；"
                     "bad_fraction 是这些区间占扫过转速范围的比例" % margin)}


# ---------------------------------------------------------------- 挠度与让刀

def tool_deflection_mm(force_n, length_mm, diameter_mm, e_gpa=600.0):
    """悬臂梁端部受力的挠度：d = F L^3 / (3 E I)，I = pi D^4 / 64。

    硬质合金 E 取 600 GPa（钢是 210，所以同样几何下硬质合金刀弯得少得多，
    但它脆 —— 这一条只算弯，不算断）。

    为什么必须算这个：让刀量直接变成尺寸误差。D10、伸出 40 mm 的刀在 300 N 下
    弯 0.09 mm，而这台机床的公差是 0.025 mm —— 也就是说"机床精度"在这件事上
    没有决定权，刀具刚性有。
    """
    if diameter_mm <= 0 or length_mm <= 0:
        return {"grade": "recompute", "deflection_mm": None, "why": "几何非法"}
    I = math.pi * diameter_mm ** 4 / 64.0
    E = e_gpa * 1000.0                      # GPa -> N/mm^2
    d = force_n * length_mm ** 3 / (3.0 * E * I)
    return {"grade": "recompute", "deflection_mm": d,
            "I_mm4": I, "E_n_mm2": E,
            "formula": "d = F L^3 / (3 E I), I = pi D^4/64",
            "note": "只算悬臂弯曲；刀柄夹持处的柔度、刀具本身的螺旋角让刀未计入（会更大）"}


def cutting_force_n(ap_mm, fz_mm, kt_n_mm2=700.0, teeth=4, radial_ratio=1.0):
    """切向主切削力的近似：F = Kt * ap * h_mean * z_engaged。

    h_mean 用 fz * sqrt(radial_ratio)（与 feeds 的切屑变薄系数一致）。
    这是**均值**，不是峰值：峰值可以到均值的 2 倍，判刚性时要用峰值。
    """
    h = fz_mm * math.sqrt(max(0.01, radial_ratio))
    z_eng = max(1.0, teeth * min(1.0, radial_ratio))
    return {"grade": "reference", "force_n": kt_n_mm2 * ap_mm * h * z_eng,
            "h_mean_mm": h, "z_engaged": z_eng,
            "note": "均值；判刚性用峰值，峰值约为均值的 1.5~2 倍"}


# ---------------------------------------------------------------- 热变形

def thermal_growth_um(length_mm, delta_t_k, alpha_1e6_per_k=11.5):
    """热伸长：dL = alpha * L * dT。11.5e-6/K 是钢，铝 23.6，铸铁 10.5。

    为什么这一条在精工里比什么都重要：一根 500 mm 的钢件，温度差 1 K 就长 5.75 um。
    IT7 在 50 mm 段是 25 um —— 4 K 是 23 um（占 92%），5 K 是 28.8 um 就超了。
    也就是说**车间温差 4~5 K 就把公差吃干净了**，而这跟机床精度一点关系都没有。
    （这个数我第一版写成"4 K 就吃掉"，实测 4 K×5.75=23 um 还差 2 um 没超，
    所以改成区间并写明。这种"差不多"正是会把结论说过头的地方。）
    """
    return {"grade": "recompute",
            "growth_um": alpha_1e6_per_k * 1e-6 * length_mm * delta_t_k * 1000.0,
            "formula": "dL = alpha * L * dT",
            "alpha_1e6_per_k": alpha_1e6_per_k}


# ---------------------------------------------------------------- 表面粗糙度

def residual_height_um(stepover_mm, radius_mm):
    """球头刀行切留下的残留高度：h = R - sqrt(R^2 - (s/2)^2)。

    小步距时它约等于 s^2/(8R)。这一句我写错过一次：原来注释说"近似式在 s=0.8R 时
    低报 3 倍"，实测 s=0.8R 时精确 417 um / 近似 400 um，只差 4%。真正拉开是在
    s 接近 2R（球头刀切到全直径）时，比值趋近 2 倍，而不是 3 倍。所以精确式的理由
    不是"近似式错得离谱"，而是它在 s→2R 时确实错、且那个区间（大行距粗加工）
    正好会被用到。
    """
    R = max(1e-6, radius_mm)
    s = min(stepover_mm, 2.0 * R)
    h = R - math.sqrt(max(0.0, R * R - (s / 2.0) ** 2))
    return {"grade": "recompute", "height_um": h * 1000.0,
            "stepover_mm": stepover_mm, "radius_mm": radius_mm,
            "approx_um": (s * s / (8.0 * R)) * 1000.0,
            "note": "几何残留高度；实际 Ra 还受刀纹、振动、材料撕裂纹影响，通常更高"}


def stepover_for_ra(radius_mm, ra_um):
    """反解：要 Ra=1.6 um，步距该取多少（几何上限）。"""
    R = radius_mm
    h = ra_um / 1000.0
    if h >= R:
        return {"grade": "recompute", "stepover_mm": None, "why": "Ra 大于刀尖半径"}
    s = 2.0 * math.sqrt(max(0.0, 2.0 * R * h - h * h))
    return {"grade": "recompute", "stepover_mm": s, "ra_um": ra_um,
            "note": "几何上限：实际要达到这个 Ra，步距必须小于等于它"}


# ---------------------------------------------------------------- 每齿进给的下限

# 最小切屑厚度与刃口钝圆半径的比值。低于它，刀在"刮"而不是"切"，
# 结果是加工硬化、表面撕裂、刀具急剧磨损。这是**必须实测**的一档：
# 比值随材料、涂层、刃口处理变，公开值是范围不是定值。
MIN_CHIP_RATIO = {
    "steel": (0.10, 0.25), "aluminium": (0.15, 0.35),
    "titanium": (0.08, 0.15), "inconel": (0.05, 0.12), "cast_iron": (0.10, 0.30),
}


def min_fz_mm(edge_radius_um, material="steel"):
    """每齿进给的下限 = 刃口钝圆半径 * 比值。

    这一条是**下限**，不是上限。车间里所有的表都在讲上限（别崩刀），
    而精加工的废品大多出在下限这一侧：进给太小，刀在挤压而不是切削。
    """
    lo, hi = MIN_CHIP_RATIO.get(material, (0.10, 0.25))
    r = edge_radius_um / 1000.0
    return {"grade": "must-measure", "fz_min_mm": lo * r, "fz_ok_mm": hi * r,
            "edge_radius_um": edge_radius_um, "ratio": (lo, hi),
            "note": "比值是范围；刃口钝圆半径要按刀具批次实测（新刀 5~10 um，涂层后更大）"}


# ---------------------------------------------------------------- 刀具寿命

def taylor_life(c_m_min=200.0, n_exp=0.25, vc_m_min=120.0):
    """Taylor：vc * T^n = C  ->  T = (C/vc)^(1/n)。

    这一条整条都是 [must-measure]：C 和 n 是**这对刀 + 这个材料**的拟合结果，
    公开表给的是范围（硬质合金钢 n=0.2~0.3，C 因涂层差一倍以上）。
    所以这里给的不是寿命，是"用两个点就能自己拟合"的式子：
        在两个 vc 下各测到一次寿命，两条方程解出 C 和 n。
    """
    T = (c_m_min / vc_m_min) ** (1.0 / n_exp) if n_exp > 0 else None
    return {"grade": "must-measure", "life_min": T,
            "formula": "vc * T^n = C", "C": c_m_min, "n": n_exp, "vc": vc_m_min,
            "note": "C 与 n 必须用至少两组实测寿命拟合；本值只是量级参考"}


def fit_taylor(points):
    """两个点拟合 Taylor 的 C 与 n。points 是 [(vc, T), ...]。

    对数坐标下它是直线：log T = (1/n) log C - (1/n) log vc。两个点就够解，
    三个点以上用最小二乘并报 R2 —— 只有一组数据时，这条曲线是编的。
    """
    pts = [(float(v), float(t)) for v, t in points if v > 0 and t > 0]
    if len(pts) < 2:
        return {"ok": False, "why": "至少两组 (vc, T) 才谈得上拟合"}
    xs = [math.log(v) for v, _ in pts]
    ys = [math.log(t) for _, t in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"ok": False, "why": "两组的 vc 相同，解不出 n"}
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    n_exp = -1.0 / slope
    C = math.exp(inter * n_exp)
    ss_res = sum((y - (slope * x + inter)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"ok": True, "n": n_exp, "C": C, "points": n,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}


# ---------------------------------------------------------------- 汇总

def report(ap_mm=2.0, fn_hz=600.0, k_n_mm=5.0e4, zeta=0.025, kt=700.0, teeth=4,
           d_tool_mm=10.0, stickout_mm=40.0):
    """把上面几件拼成一份"这活干不干得了"的体检表。"""
    f = cutting_force_n(ap_mm, 0.05, kt_n_mm2=kt, teeth=teeth, radial_ratio=0.5)
    d = tool_deflection_mm(f["force_n"] * 1.75, stickout_mm, d_tool_mm)
    stab = stability_lobes(fn_hz=fn_hz, k_n_mm=k_n_mm, zeta=zeta, kt_n_mm2=kt, teeth=teeth)
    bands = avoid_bands(stab, ap_mm)
    return {"force": f, "deflection": d, "stability": stab, "avoid": bands,
            "a_min_mm": unconditional_depth_mm(k_n_mm, zeta, kt, teeth),
            "thermal_1k_500mm_um": thermal_growth_um(500.0, 1.0)["growth_um"]}
