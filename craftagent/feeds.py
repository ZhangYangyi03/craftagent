# -*- coding: utf-8 -*-
"""feeds.py -- 切削参数手册：**查表**，不是让模型自己编数字。

这个文件存在的理由只有一条，而且它是这一层最重要的设计决定：

    大模型算算术题会算错，而且错得看不出来。

让它"根据材料、直径、齿数算主轴转速"，它会给你一个像模像样的 3827 rpm。
那个数字可能是对的，也可能是把 vc=80 代进了 vc=180 的式子——两者都是
四位有效数字，都写在正确的量纲里，**从结果上看不出来**。而车间里的
后果是不一样的：轻则表面烧伤，重则断刀。

所以这一层的分工是：

    模型负责说"这是什么材料、什么工序、什么刀具"（分类，它可以做）
    这张表负责把分类变成数字（查表，它不该做）

表里的每一个数都有一个出处标注，没有出处的数不放进来。这是"可被反驳"
的最低要求：说得出这个数从哪来，别人才查得动。

三条公式写在这里，因为这个仓的习惯是物理不藏在代码里：

    转速      n = 1000 * vc / (pi * D)        vc 是切削线速度 m/min
    每齿进给  fz  按材料与刀具直径取（表）
    进给      vf = n * z * fz                 z 是齿数

    每转进给  fn = z * fz        （钻孔/攻丝用，与转速无关）

而切削功率，因为这个仓一直在意的原因——**它必须能反过来验算**：

    Pc = ap * ae * vf * kc / (60 * 10^6)      [kW]
    kc 是单位切削力 N/mm^2，按材料取（表），随切屑厚度变薄而升高

功率这一条是这张表唯一能自我检验的部分：如果查出来的 F 让这台机床
超功率，那么要么表错了、要么这个活这台机床干不了。它把"AI 生成的
参数对不对"从"看起来合理"变成了一个能算的数。
"""
import math


# ---------------------------------------------------------------- 材料
# vc 单位 m/min。范围取自公开的硬质合金立铣刀推荐值，按"钢-不锈钢-铸铁-
# 铝-钛"的常见分档。取的是区间中偏低的一侧：车间里烧刀比慢工更贵。
#
# 每个材料还带一个 kc（单位切削力 N/mm^2）——它不出自切削手册，
# 出自 Kienzle 公式的常用简化值。kc 随切屑厚度变薄而升高（尺寸效应），
# 这里给的是 h~0.1mm 附近的名义值，尺寸效应在 power() 里单独算。
MATERIALS = {
    'P': {  # 钢
        'label': '碳钢/合金钢',
        'vc_m_min': {'HSS': 25.0, 'carbide': 90.0, 'coated': 140.0},
        'fz_base_mm': 0.055,
        'kc_n_mm2': 2100.0,
        'kc_exponent': 0.26,
    },
    'M': {  # 不锈钢
        'label': '不锈钢',
        'vc_m_min': {'HSS': 15.0, 'carbide': 50.0, 'coated': 80.0},
        'fz_base_mm': 0.040,
        'kc_n_mm2': 2500.0,
        'kc_exponent': 0.24,
    },
    'K': {  # 铸铁
        'label': '铸铁',
        'vc_m_min': {'HSS': 20.0, 'carbide': 70.0, 'coated': 110.0},
        'fz_base_mm': 0.060,
        'kc_n_mm2': 1600.0,
        'kc_exponent': 0.28,
    },
    'N': {  # 铝
        'label': '铝合金',
        'vc_m_min': {'HSS': 120.0, 'carbide': 400.0, 'coated': 500.0},
        'fz_base_mm': 0.080,
        'kc_n_mm2': 700.0,
        'kc_exponent': 0.25,
    },
    'S': {  # 钛/高温合金
        'label': '钛合金/高温合金',
        'vc_m_min': {'HSS': 10.0, 'carbide': 30.0, 'coated': 50.0},
        'fz_base_mm': 0.030,
        'kc_n_mm2': 2100.0,
        'kc_exponent': 0.22,
    },
}

# 刀具材料。选错这一项，转速会差 4 倍，而两者看起来都"有道理"。
TOOL_MATERIALS = ('HSS', 'carbide', 'coated')

# 工序。粗铣与精铣不是"同一件事的两个档位"：
# 粗铣要切除率，精铣要表面质量，而它们对进给的约束来自不同的地方。
OPERATIONS = ('rough', 'finish', 'slot', 'drill')


class Tool(object):
    """一把刀。直径、齿数、可用长度。"""

    def __init__(self, diameter_mm=10.0, flutes=4, material='carbide',
                 stickout_mm=30.0, name=None):
        self.diameter_mm = float(diameter_mm)
        self.flutes = int(flutes)
        self.material = material
        self.stickout_mm = float(stickout_mm)
        self.name = name or ('D%.0f-%dF-%s' % (diameter_mm, flutes, material))

    def __repr__(self):
        return '<Tool %s>' % self.name


class Machine(object):
    """机床的功率与转速包络。参数决策必须落在这里面，否则说什么都白搭。"""

    def __init__(self, spindle_max_rpm=12000.0, spindle_min_rpm=100.0,
                 power_kw=11.0, efficiency=0.80, feed_max_mm_min=6000.0):
        self.spindle_max_rpm = spindle_max_rpm
        self.spindle_min_rpm = spindle_min_rpm
        self.power_kw = power_kw
        self.efficiency = efficiency
        self.feed_max_mm_min = feed_max_mm_min


# 每齿进给的直径分档。刀具厂样本给的是"某个直径区间对应的每齿进给"，
# 不是一条穿过原点的直线。表里存的是 D=基准直径时的值，
# 其它直径按 STEPS 里相邻档位取，中间不插直线——插直线就等于又回到
# "按比例缩放"那个错。
FZ_STEPS = ((3.0, 0.45), (6.0, 0.75), (10.0, 1.00), (16.0, 1.25), (25.0, 1.45))


def fz_for_diameter(fz_base_mm, diameter_mm):
    """按直径分档取每齿进给。基准值对应 D=10 mm 那一档。

    大于最大档按最大档，小于最小档按最小档：两者都是**端部**而不是外推。
    外推一个从没被验证过的区间，是让"看起来合理"替掉"有人验过"。
    """
    if diameter_mm <= 0:
        return fz_base_mm
    scale = FZ_STEPS[-1][1]
    for d, sc in FZ_STEPS:
        if diameter_mm <= d:
            scale = sc
            break
    return fz_base_mm * scale


# ---------------------------------------------------------------- 公式
def spindle_rpm(vc_m_min, diameter_mm):
    """n = 1000 vc / (pi D)。"""
    if diameter_mm <= 0:
        return 0.0
    return 1000.0 * vc_m_min / (math.pi * diameter_mm)


def feed_mm_min(rpm, flutes, fz_mm):
    """vf = n * z * fz。"""
    return rpm * flutes * fz_mm


def chip_thinning_factor(radial_engagement_ratio):
    """径向切深的切屑变薄系数。

    侧铣时径向切深 ae 小于刀具直径 D，实际切屑厚度比每齿进给小，
    于是同一进给量下的切削力反而下降。这个系数存在的意义是：
    不修正它，精铣的进给会被压得过低，而"保守"看起来总是对的。
    """
    a = max(0.0, min(1.0, radial_engagement_ratio))
    if a >= 1.0:
        return 1.0
    if a <= 1e-6:
        return 1.0
    # sin 近似：实际最大切屑厚度 ≈ fz * sin(径向接触角)
    return max(0.35, math.sqrt(a))


def specific_force(kc_n_mm2, exponent, mean_chip_mm):
    """Kienzle：kc = kc1.1 * h^-mc。切屑越薄，单位切削力越高。"""
    h = max(1e-4, mean_chip_mm)
    return kc_n_mm2 * (h ** (-exponent))


def power_kw(ap_mm, ae_mm, vf_mm_min, kc_n_mm2):
    """Pc = ap * ae * vf * kc / (60e6)  [kW]。

    60 是把 mm/min 换成 mm/s，1e6 是把 N·mm/s 换成 kW。
    两个常数写在这里而不是藏在代码里：单位换错一次，
    算出来的功率会差 60 倍，而 60 倍的错误在所有自查里都像"这台机床不行"。
    """
    return ap_mm * ae_mm * (vf_mm_min / 60.0) * kc_n_mm2 / 1.0e6


# ---------------------------------------------------------------- 查表
class Choice(object):
    """一次参数决策的结果，连同它为什么是这个数。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def as_gcode_words(self):
        return 'M3 S%.0f' % round(self.rpm), 'F%.0f' % round(self.feed_mm_min)

    def __repr__(self):
        return ('<Choice %s %s n=%.0f rpm vf=%.0f mm/min Pc=%.2f kW%s>'
                % (self.material, self.op, self.rpm, self.feed_mm_min,
                   self.power_kw, ' CLAMPED' if self.clamped else ''))


def choose(material='P', op='rough', tool=None, machine=None,
           ap_mm=1.0, ae_mm=None, radial_engagement_ratio=None,
           conservative=1.0):
    """给定材料/工序/刀具/机床，查出一组合规的 S 与 F。

    conservative 是一个乘到进给上的系数（<1 更保守），存在的原因是：
    表给的是**可行**参数，不是**最优**参数，而这两者之间的余量属于工艺员。
    把它做成显式参数，而不是悄悄乘一个 0.8——那种"看不见的余量"
    是事后没人说得清为什么进给是 480 而不是 600 的原因。

    返回 Choice；被机床包络夹住的项记在 clamped 里，说明**谁夹的**。
    """
    tool = tool or Tool()
    machine = machine or Machine()
    if material not in MATERIALS:
        raise KeyError('unknown material group %r (have %s)'
                       % (material, ', '.join(sorted(MATERIALS))))
    mat = MATERIALS[material]
    if tool.material not in mat['vc_m_min']:
        raise KeyError('tool material %r has no vc for %s'
                       % (tool.material, material))
    if ae_mm is None:
        if radial_engagement_ratio is not None:
            ae_mm = radial_engagement_ratio * tool.diameter_mm
        else:
            ae_mm = (tool.diameter_mm if op == 'slot' else 0.5 * tool.diameter_mm)
    if radial_engagement_ratio is None:
        radial_engagement_ratio = ae_mm / max(1e-9, tool.diameter_mm)

    vc = mat['vc_min' if 'vc_min' in mat else 'vc_m_min'][tool.material]
    # 工序修正：粗铣可以稍快（切得深、不怕表面），精铣反而要慢一点求表面，
    # 因为精铣的切深小、切屑薄，同样的转速会让刀刃在硬化层里摩擦。
    op_factor = {'rough': 1.0, 'finish': 0.85, 'slot': 0.6, 'drill': 0.7}.get(op, 1.0)
    vc *= op_factor

    rpm_raw = spindle_rpm(vc, tool.diameter_mm)
    clamped = []
    rpm = rpm_raw
    if rpm > machine.spindle_max_rpm:
        rpm = machine.spindle_max_rpm
        clamped.append('rpm_high')
    if rpm < machine.spindle_min_rpm:
        rpm = machine.spindle_min_rpm
        clamped.append('rpm_low')

    # 每齿进给：按直径分档查表，**不是**按直径线性缩放。
    #
    # 第一版写成 fz = 比例 * D（比例 0.020），于是 D10 钢粗铣得到 fz=0.20 mm，
    # vf=1621 mm/min。那个数落在门的进给上限之内、功率也只有 0.52 kW，
    # 所有自查都会说"合规"——而 0.20 mm/齿对一把 D10 硬质合金刀是断刀的进给。
    # 错的地方在假设：fz 随直径涨，但涨得比线性慢得多（刃口钝圆半径约为常数，
    # 大刀具靠刚性多切一点，不是按比例多切）。刀具厂给的是分档表，就按分档查。
    fz = fz_for_diameter(mat['fz_base_mm'], tool.diameter_mm)
    # 精铣的每齿进给必须小于刃口钝圆半径量级，否则表面是"刮"出来的
    if op == 'finish':
        fz *= 0.55
    elif op == 'slot':
        fz *= 0.45          # 满槽切屑排不出，进给必须降
    elif op == 'drill':
        fz *= 0.5

    fz *= max(0.05, conservative)
    factor = chip_thinning_factor(radial_engagement_ratio)
    fz_eff = fz * factor

    vf_raw = feed_mm_min(rpm, tool.flutes, fz_eff)
    vf = vf_raw
    if vf > machine.feed_max_mm_min:
        vf = machine.feed_max_mm_min
        clamped.append('feed_high')

    # 功率校核。这一条是这张表唯一能自我反驳的地方：查出来的参数
    # 若让机床超功率，那"表说可行"就是假的。夹进给，并记下是功率夹的。
    mean_chip = fz_eff * factor
    kc = specific_force(mat['kc_n_mm2'], mat['kc_exponent'], mean_chip)
    pc = power_kw(ap_mm, ae_mm, vf, kc)
    pc_avail = machine.power_kw * machine.efficiency
    while pc > pc_avail and vf > 1.0:
        vf *= 0.9
        pc = power_kw(ap_mm, ae_mm, vf, kc)
        clamped.append('power')
    # 去重但保留首次出现的顺序，方便报表
    seen, cl = set(), []
    for c in clamped:
        if c not in seen:
            seen.add(c); cl.append(c)

    return Choice(material=material, op=op, tool=tool, machine=machine,
                  vc_m_min=vc, rpm=rpm, rpm_raw=rpm_raw,
                  fz_mm=fz_eff, fz_raw=fz, feed_mm_min=vf, feed_raw=vf_raw,
                  feed_per_rev_mm=tool.flutes * fz_eff,
                  ap_mm=ap_mm, ae_mm=ae_mm, kc_n_mm2=kc, power_kw=pc,
                  power_available_kw=pc_avail, clamped=cl,
                  radial_engagement_ratio=radial_engagement_ratio)


def table():
    """把手册打成一张能打印的表。参数决策要能被审阅，不能只在代码里。"""
    rows = []
    for key in ('P', 'M', 'K', 'N', 'S'):
        m = MATERIALS[key]
        rows.append((key, m['label'],
                     m['vc_m_min']['HSS'], m['vc_m_min']['carbide'],
                     m['vc_m_min']['coated'], m['kc_n_mm2']))
    return rows
