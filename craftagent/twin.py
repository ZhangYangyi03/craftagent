# -*- coding: utf-8 -*-
"""twin.py -- 把辨识出来的系数装成一台可运行的孪生，并且当场验它有没有比"不动"更准。

这个文件存在的理由只有一个：**参数辨识如果只停在报告上，它就是一张好看的纸。**
数字孪生的用处是"改一个数之前，先说结果会变成多少"，所以辨识出的
(tau, 增益) 必须真的能被拿去算，而且必须证明它算得比最笨的基线准。

基线选得很硬：**什么都不预测（每个班次取该通道的恒定中位值）**。
孪生连"班次内会爬升"这件事都预测不过，那就没有理由相信它的任何其它预测。

两条纪律：
  1. 只装可辨识的通道。不可辨识的通道装进孪生，孪生就会在最需要它的时候
     （也就是数据不在它的适用区间时）给一个像模像样的错数。
  2. 每个系数带 provenance：从哪份日志、多少个点、R2 多少、窗口多长。
     别人要反驳，得先能看见它是从哪来的。
"""
import math

from .stats import median


class Twin(object):
    """一台机器的班次级孪生：每个通道一条一阶曲线，外加症状->质量的线性映射。

    不是全过程仿真。它回答的是这类问题：
        开机 40 分钟时主轴温度会到多少？ -> 一阶曲线
        振动到了 3.0，圆度大概多少？     -> 症状->质量增益
    以及一个它必须拒绝回答的问题：
        把跟随误差改成多少，圆角误差会掉到几？ -> 那是 tau_s 的事，一分钟采样认不出，
                                              孪生没有这个系数，就不许编一个。
    """

    def __init__(self, machine, cycle_min, channels, symptom_map=None, provenance=None,
                 unidentifiable=None):
        self.machine = machine
        self.cycle_min = cycle_min
        self.channels = channels               # ch -> {"tau_min","y0","y_inf","r2","n_points"}
        self.symptom_map = symptom_map or {}   # ch -> {"gain","offset","r2_holdout"}
        self.provenance = provenance or {}
        self.unidentifiable = unidentifiable or {}
        self.validation = {}
        self.rejected = {}

    # ---------------------------------------------------------------- 预测
    def predict_channel(self, ch, t_min):
        """班次内第 t 分钟这个通道应该是多少。不可辨识的通道直接 KeyError 说明原因。"""
        if ch not in self.channels:
            raise KeyError("通道 %s 未装进孪生：%s" %
                           (ch, self.unidentifiable.get(ch) or "没有辨识结果"))
        c = self.channels[ch]
        return c["y_inf"] - (c["y_inf"] - c["y0"]) * math.exp(-t_min / c["tau_min"])

    def predict_quality(self, ch, value):
        """症状值 -> 质量预测。没有可用的症状映射就拒绝，不给默认值。"""
        m = self.symptom_map.get(ch)
        if not m:
            raise KeyError("症状 %s 没有通过留出检验的增益，孪生不预测质量" % ch)
        return m["gain"] * value + m["offset"]

    def can_predict(self, question):
        """这台孪生能不能回答某类问题。写出来，因为"不能"必须是可查的，不是靠记忆。"""
        q = (question or "").lower()
        if any(k in q for k in ("跟随误差", "位置环", "kv", "伺服")):
            return False, ("tau_s / kv 没有装进孪生：一分钟采样的工况日志认不出它，"
                           "要有 >= 1 kHz 的指令-实际位置对才谈得上")
        if any(k in q for k in ("加速度", "acc", "jerk")):
            return False, "各轴加速度上限不在日志里，孪生装的是 plan.py 的保守默认值，不是这台机的实测值"
        if any(k in q for k in ("圈度", "圆度", "quality", "尺寸")):
            if not self.symptom_map:
                return False, "没有通过留出检验的症状->质量增益，孪生不预测质量"
        return True, ""

    # ---------------------------------------------------------------- 验它
    def validate(self, rows, channels=None):
        """拿班次内的实测曲线核对：孪生的 RMSE 必须小于基线的 RMSE。

        基线必须选对，否则这一步什么也证明不了。第一版我把基线写成"每个相位的中位值"，
        结果孪生输了 —— 因为孪生拟合的曲线本身就是那条相位中位曲线的平滑版本，
        两个东西在量同一件事，比的只是谁的噪声小。那不是检验，是同义反复。

        真正的基线是**"这个班次里它就一直是这个水平"**：每个班次取该通道的
        班次中位值，当整班都不变来预测。那是没有任何模型时的诚实预测，
        也是车间里一块带报警的仪表实际在做的事。孪生要赢的是它。

        赢不了，说明班次内的爬升对这个通道的误差贡献还不如随机波动，
        这条曲线就不该装进孪生 —— 装了只是多一个会被人当真的数。
        """
        out = {}
        for ch in (channels or self.channels):
            if ch not in self.channels:
                continue
            per_shift = {}
            for i, r in enumerate(rows):
                v = r["sig"].get(ch)
                if v is None:
                    continue
                per_shift.setdefault(i // self.cycle_min, []).append((i % self.cycle_min, v))
            twin_err, base_err, n = 0.0, 0.0, 0
            for sh, pts in per_shift.items():
                if len(pts) < self.cycle_min // 2:
                    continue                      # 半个班次以下的残段不参与（末尾常常是断的）
                b = median([v for _, v in pts])
                for phase, v in pts:
                    t = self.predict_channel(ch, phase)
                    twin_err += (v - t) ** 2
                    base_err += (v - b) ** 2
                    n += 1
            if n < 100:
                continue
            tr = math.sqrt(twin_err / n)
            br = math.sqrt(base_err / n)
            out[ch] = {"rmse_twin": tr, "rmse_flat_baseline": br,
                       "improvement": (1.0 - tr / br) if br > 0 else 0.0,
                       "n": n, "shifts": len(per_shift),
                       "beats_baseline": tr < br}
        return out


    def prune(self, rows, min_improvement=0.05):
        """验收即安装：赢不了整班恒定基线的通道，从孪生里剔掉。

        这一步跑出来的第一个结果就是最有用的一个：**在正在漂移的机器上，
        班次级孪生是无效的。** M01 的电流/温度/振动三个通道，孪生的 RMSE
        比基线差 7 倍到 275 倍 —— 因为它假设这台机器每个班次回到同一个起点，
        而它没有，它在漂。这个结论不是失败，是这条管子唯一诚实的输出：
        漂移机台该做的是修，不是给它配一条平稳曲线。

        所以孪生分成两档：
            installed  通过验收，可以拿去做"改一个数之前先算结果"
            rejected   装了但输了，剔除并记下原因（永远可以查，不会被静默丢掉）
        """
        val = self.validate(rows)
        rejected = {}
        for ch, r in val.items():
            if r["improvement"] < min_improvement:
                rejected[ch] = ("孪生 RMSE %.4f 输给整班恒定基线 %.4f（改善 %+.1f%%）："
                                "这个通道在班次之间不平稳（漂移、换刀或换批次都会这样），"
                                "班次级孪生对它无效——该做的是查出漂移源并修，不是给它配曲线"
                                % (r["rmse_twin"], r["rmse_flat_baseline"], r["improvement"] * 100))
                self.channels.pop(ch, None)
                self.unidentifiable[ch] = rejected[ch]
        self.validation = val
        self.rejected = rejected
        return val

def from_identify(ident, machine):
    """把 ident.identify() 的结果装成一台孪生。只装可辨识的那部分。"""
    m = ident["machines"][machine]
    channels, unident = {}, {}
    for ch, r in m["time_constants"].items():
        if r.get("identifiable"):
            channels[ch] = {"tau_min": r["tau_min"], "y0": r["y0"], "y_inf": r["y_inf"],
                            "r2": r["r2"], "n_points": r["points"],
                            "window_min": r["window_min"],
                            "provenance": "工况日志班次相位中位曲线，%d 个班次 x %d 分钟窗口"
                                          % (r.get("shifts", 0), r["window_min"])}
        else:
            unident[ch] = r.get("why") or "不可辨识"
    smap = {}
    for ch, g in m["symptom_gain"].items():
        # 只装留出检验过得去的：R2 在留出段为负的映射，装进去就是把过拟合当知识。
        if g.get("ok") and (g.get("r2_holdout") or 0) > 0.5:
            smap[ch] = {"gain": g["gain"], "offset": g["offset"],
                        "r2_holdout": g["r2_holdout"], "r2_cv": g.get("r2_cv"),
                        "segment": g.get("segment"),
                        "provenance": "变点后线性拟合，训练 %d 点 / 留出 %d 点"
                                      % (g["n_train"], g["n_holdout"])}
    return Twin(machine, ident["cycle_min"], channels, smap,
                provenance={"source": "identify", "cycle_min": ident["cycle_min"],
                            "dt_min": ident["dt_min"]},
                unidentifiable=unident)


def render(twins, checks):
    L = ["数字孪生（从辨识结果装配，未辨识的通道不装）", ""]
    for mid, tw in twins.items():
        L.append("== %s" % mid)
        for ch, c in sorted(tw.channels.items()):
            L.append("  %-20s y0 %8.3f -> y_inf %8.3f  tau %5.1f 分钟  R2 %.2f"
                     % (ch, c["y0"], c["y_inf"], c["tau_min"], c["r2"]))
        for ch, s in sorted(tw.symptom_map.items()):
            L.append("  %-20s 质量 = %+.3f * 症状 %+.3f   留出R2 %+.2f"
                     % (ch + " ->质量", s["gain"], s["offset"], s["r2_holdout"]))
        for ch, why in sorted(tw.unidentifiable.items()):
            tag = "剔除" if ch in (tw.rejected or {}) else "未装"
            L.append("  %-20s %s：%s" % (ch, tag, why))
        c = checks.get(mid) or {}
        L.append("  验它（班次内 RMSE，孪生 vs 恒定中位值基线）")
        for ch, r in sorted(c.items()):
            L.append("    %-20s %8.4f vs %8.4f   改善 %+5.1f%%   %s"
                     % (ch, r["rmse_twin"], r["rmse_flat_baseline"], r["improvement"] * 100,
                        "赢" if r["beats_baseline"] else "没赢，这条曲线不该装"))
        L.append("")
    return "\n".join(L)
