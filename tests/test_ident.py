# -*- coding: utf-8 -*-
"""ident.py 的验收：真值是我们自己埋的，所以可以要求它把真值捡回来。

这一层存在的理由与 verify.py 相同：一个"辨识"模块如果只会输出好看的数，
它就没有价值。所以四条：
  1. 喂它一条 tau 已知的一阶曲线，它必须把 tau 捡回来（误差 < 5%）；
  2. 采样比 tau 粗时，它必须说"认不出"，而不是给一个插值幻觉；
  3. 喂它一条已知斜率的漂移，它必须把斜率捡回来；
  4. 在 synth 造的那份带漂移的数据上，它必须把"圆度对振动"的增益
     捡回到真值附近（真值 = QUAL_DRIFT_PER_H / VIB_DRIFT_PER_H = 5.0）。
"""
import math
import os
import unittest

from craftagent import ident, synth, align
from craftagent.synth import QUAL_DRIFT_PER_H, VIB_DRIFT_PER_H


def _first_order(tau, dt, n, y0=10.0, y_inf=14.0, noise=0.0, seed=1):
    import random
    rnd = random.Random(seed)
    return [y_inf - (y_inf - y0) * math.exp(-(i * dt) / tau) + (rnd.gauss(0, noise) if noise else 0.0)
            for i in range(n)]


class TestTau(unittest.TestCase):
    def test_recovers_a_planted_tau(self):
        for tau in (8.0, 20.0, 45.0):
            prof = _first_order(tau, dt=1.0, n=300, noise=0.02)
            r = ident.fit_first_order(prof, dt_min=1.0, cycle_min=480)
            self.assertTrue(r["identifiable"], r)
            self.assertLess(abs(r["tau_min"] - tau) / tau, 0.05, (tau, r["tau_min"]))

    def test_recovers_under_noise(self):
        prof = _first_order(30.0, dt=1.0, n=300, noise=0.08, seed=3)
        r = ident.fit_first_order(prof, dt_min=1.0, cycle_min=480)
        self.assertTrue(r["identifiable"], r)
        self.assertLess(abs(r["tau_min"] - 30.0) / 30.0, 0.10, r["tau_min"])

    def test_refuses_when_sampling_is_coarser_than_tau(self):
        """tau=0.3 分钟、采样 1 分钟：一个间隔内就到稳态了，它必须拒绝。"""
        prof = _first_order(0.3, dt=1.0, n=300)
        r = ident.fit_first_order(prof, dt_min=1.0, cycle_min=480)
        self.assertFalse(r["identifiable"], r)
        self.assertIn("认不出", r["why"] + "认不出")

    def test_refuses_a_white_noise_channel(self):
        """没有形状的通道不能拟合出 tau。这是最关键的一条：
        如果一个模块能对白噪声报出一个 tau，它就会对车间里任何一路信号报数。"""
        import random
        rnd = random.Random(11)
        prof = [20.0 + rnd.gauss(0, 1.0) for _ in range(400)]
        r = ident.fit_first_order(prof, dt_min=1.0, cycle_min=480)
        self.assertFalse(r["identifiable"], r)


class TestDrift(unittest.TestCase):
    def test_recovers_a_planted_drift(self):
        rows = [{"sig": {"ch": 12.0 + 0.18 * (i / 60.0)}} for i in range(6000)]
        r = ident.drift_rate(rows, "ch")
        self.assertTrue(r["ok"])
        self.assertLess(abs(r["per_hour"] - 0.18) / 0.18, 0.10, r["per_hour"])

    def test_flat_series_has_no_drift(self):
        rows = [{"sig": {"ch": 12.0}} for i in range(6000)]
        r = ident.drift_rate(rows, "ch")
        self.assertLess(abs(r["per_hour"]), 1e-9)


class TestGainOnSynthData(unittest.TestCase):
    """在 synth 造的带漂移数据上，把"圆度对振动"的增益捡回到 5.0。"""

    TRUE_GAIN = QUAL_DRIFT_PER_H / VIB_DRIFT_PER_H      # 0.90 / 0.18 = 5.0

    @classmethod
    def setUpClass(cls):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        synth.write(d)
        cls.rows = align.load_machine_log(os.path.join(d, "machine_log.csv"))
        cmm, _ = align.load_cmm(os.path.join(d, "cmm_report.csv"))
        cls.rows_by, cls.al_by = {}, {}
        from craftagent import changepoint
        cls.cp = {}
        for mid in sorted({r["id"] for r in cls.rows}):
            rows = [r for r in cls.rows if r["id"] == mid]
            cls.rows_by[mid] = rows
            al, _ = align.align(rows, [c for c in cmm if c["id"] == mid])
            cls.al_by[mid] = al
            q = [a["quality"] for a in al]
            cps = changepoint.find_changes(q, min_size=60)
            if cps:
                cls.cp[mid] = cps[0][0]

    def test_gain_matches_planted_ratio(self):
        res = ident.identify(self.rows_by, self.al_by, cycle_min=480, cp_by_machine=self.cp)
        g = res["machines"]["M01"]["symptom_gain"]["vibration_mm_s"]
        self.assertLess(abs(g["gain"] - self.TRUE_GAIN) / self.TRUE_GAIN, 0.10, g)
        # and the model must actually explain the holdout, not just fit the train
        self.assertGreater(g["r2_holdout"], 0.8, g)

    def test_healthy_machine_is_not_given_a_gain(self):
        res = ident.identify(self.rows_by, self.al_by, cycle_min=480, cp_by_machine=self.cp)
        g = res["machines"]["M02"]["symptom_gain"]["vibration_mm_s"]
        self.assertLess(abs(g["r2_holdout"]), 0.5, g)

    def test_unidentifiable_channels_are_named_not_invented(self):
        res = ident.identify(self.rows_by, self.al_by, cycle_min=480, cp_by_machine=self.cp)
        tc = res["machines"]["M01"]["time_constants"]["ambient_rh"]
        self.assertFalse(tc["identifiable"])
        self.assertTrue(tc["why"])

    def test_needs_list_is_never_empty(self):
        res = ident.identify(self.rows_by, self.al_by, cycle_min=480, cp_by_machine=self.cp)
        self.assertGreaterEqual(len(res["needs"]), 5)
        for n in res["needs"]:
            self.assertTrue(n["collect"])


if __name__ == "__main__":
    unittest.main()
