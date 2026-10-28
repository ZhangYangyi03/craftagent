# -*- coding: utf-8 -*-
"""twin.py 的验收：装进孪生的每一条曲线都必须赢过"整班恒定"这个最笨的基线。

这一层最关键的两条不是"预测得准"，而是：
  1. 正在漂移的机器，班次级孪生必须**自己认输**并把通道剔掉（而不是装进去）；
  2. 认不出的参数不许被预测，报错要带原因。

第 1 条是这一整节管子的核心结论：一台在漂的机器需要的不是一条平稳曲线，是修。
"""
import math
import os
import random
import unittest

from craftagent import align, changepoint, ident, synth, twin


def _shift_rows(channel, tau, gain, n_shifts=5, cycle=480, offsets=None, noise=0.02, seed=5):
    """造一份"班次内形状相同、每个班次水平不同"的日志。offset 是每班的偏移，
    故意给成一条上升序列，模拟正在漂的机器：叠形状前必须先减水平。"""
    rnd = random.Random(seed)
    offsets = offsets or [i * 1.5 for i in range(n_shifts)]
    rows = []
    for sh in range(n_shifts):
        for ph in range(cycle):
            v = 10.0 + offsets[sh] + gain * (1.0 - math.exp(-ph / tau)) + rnd.gauss(0, noise)
            rows.append({"ts": None, "id": "MX", "sig": {channel: v}})
    return rows


class TestPlantedShape(unittest.TestCase):
    def test_deleveling_makes_a_drifting_machine_identifiable(self):
        rows = _shift_rows("ch", tau=25.0, gain=2.0, noise=0.01)
        prof, n_shifts, base = ident.phase_profile(rows, "ch", cycle_min=480)
        self.assertEqual(n_shifts, 5)
        r = ident.fit_first_order(prof[:240], dt_min=1.0, cycle_min=480)
        self.assertTrue(r["identifiable"], r)
        self.assertLess(abs(r["tau_min"] - 25.0) / 25.0, 0.10, r["tau_min"])

    def test_partial_shifts_do_not_pollute_the_shape(self):
        """真实日志的末尾班次常常是断的。让后三个班次各缺一半相位，看形状还认不认得出。

        这条是替代一条被删掉的反证：我原本想证明"不减水平就会算错 tau"，
        但在数据完整、噪声很小时，减水平再取中位、最后把水平中位加回来，
        恰好互相抵消，两条路数值相同——那个反证不成立，删掉。
        缺点的情况才分开：不减水平时，某个相位只由某几个班次提供，
        水平就被混进形状里了。"""
        rows = _shift_rows("ch", tau=25.0, gain=2.0, noise=0.005, n_shifts=6)
        # 断日志的形态是"这一行没有值"，不是"这一行不存在"：行还在，索引不动。
        # 第一版我把行直接删掉，结果后面的班次被重新编号，班次边界全错——
        # 这个坑和日志读取器的行为是一致的，删行等于改时间轴。
        trimmed = []
        for i, r in enumerate(rows):
            if (i // 480) >= 3 and (i % 480) >= 240:
                trimmed.append({"ts": r["ts"], "id": r["id"], "sig": {}})
            else:
                trimmed.append(r)
        prof, n_shifts, base = ident.phase_profile(trimmed, "ch", cycle_min=480)
        self.assertEqual(n_shifts, 6)
        r = ident.fit_first_order(prof[:240], dt_min=1.0, cycle_min=480)
        self.assertTrue(r["identifiable"], r)
        self.assertLess(abs(r["tau_min"] - 25.0) / 25.0, 0.10, r["tau_min"])
        self.assertGreater(base, 10.0)


class TestOnSynthData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        synth.write(d)
        rows_all = align.load_machine_log(os.path.join(d, "machine_log.csv"))
        cmm, _ = align.load_cmm(os.path.join(d, "cmm_report.csv"))
        cls.rows_by, al_by, cp = {}, {}, {}
        for mid in sorted({r["id"] for r in rows_all}):
            rows = [r for r in rows_all if r["id"] == mid]
            cls.rows_by[mid] = rows
            al, _ = align.align(rows, [c for c in cmm if c["id"] == mid])
            al_by[mid] = al
            cps = changepoint.find_changes([a["quality"] for a in al], min_size=60)
            if cps:
                cp[mid] = cps[0][0]
        cls.ident = ident.identify(cls.rows_by, al_by, cycle_min=480, cp_by_machine=cp)
        cls.twins = {}
        for mid in cls.rows_by:
            tw = twin.from_identify(cls.ident, mid)
            tw.prune(cls.rows_by[mid])
            cls.twins[mid] = tw

    def test_healthy_machine_twin_beats_the_flat_baseline(self):
        tw = self.twins["M02"]
        self.assertGreaterEqual(len(tw.channels), 4, tw.channels.keys())
        for ch, r in tw.validation.items():
            self.assertGreater(r["improvement"], 0.10, (ch, r))

    def test_drifting_machine_twin_refuses_instead_of_pretending(self):
        """M01 在漂：班次级孪生必须把它的通道剔掉，并说清原因。
        如果哪天它开始在 M01 上"赢"，先怀疑基线和去水平那一步是不是又坏了。"""
        tw = self.twins["M01"]
        self.assertNotIn("temp_c", tw.channels)
        self.assertIn("temp_c", tw.rejected)
        self.assertIn("漂移", tw.rejected["temp_c"] + "这台机器的这个通道不是班次内平稳的")
        self.assertIn("temp_c", tw.unidentifiable)

    def test_unidentifiable_channel_cannot_be_predicted(self):
        tw = self.twins["M02"]
        with self.assertRaises(KeyError) as cm:
            tw.predict_channel("ambient_rh", 30)
        self.assertIn("ambient_rh", str(cm.exception))

    def test_kv_is_refused_with_a_reason(self):
        ok, why = self.twins["M02"].can_predict("把跟随误差降到 5um，圆角误差会掉到多少")
        self.assertFalse(ok)
        self.assertIn("1 kHz", why)

    def test_quality_prediction_needs_a_holdout_pass(self):
        tw = self.twins["M01"]
        self.assertIn("vibration_mm_s", tw.symptom_map)
        self.assertGreater(tw.symptom_map["vibration_mm_s"]["r2_holdout"], 0.5)
        # M02 上过不了的映射不许装
        self.assertEqual(self.twins["M02"].symptom_map, {})

    def test_predicted_roundness_at_current_vibration_matches_report(self):
        """同一个数，两条路算出来要一致：孪生的质量映射 vs 报告里的中位数。
        对不上说明装配时把增益或偏置接错了。"""
        tw = self.twins["M01"]
        q = tw.predict_quality("vibration_mm_s", 5.06)
        self.assertLess(abs(q - 21.06), 0.5, q)


if __name__ == "__main__":
    unittest.main()
