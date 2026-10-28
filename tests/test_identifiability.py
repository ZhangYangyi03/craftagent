# -*- coding: utf-8 -*-
"""identifiability 的验收：这些"不能"的结论本身必须是最硬的。

一个说"逆不出来"的模块最危险：它可以用"这不可辨识"来解释自己的任何失败。
所以这里的测试全部指向可证伪的方向 —— 每条"不可辨识"都要能给出**至少两个
不同参数、同样的观测**（构造出的反例），每条"可辨识"都要能报出误差多大。
"""
import math
import unittest

from craftagent import identifiability as I


class TestNotIdentifiable(unittest.TestCase):
    def test_peak_alone_gives_a_whole_curve_of_k_and_zeta(self):
        """峰值固定时，四组 (k, zeta) 给出同一个峰值 —— 那就是"解不是点"。"""
        rows = I.peak_tradeoff(5.0e4, 0.025)
        peaks = [r["peak"] for r in rows]
        for p in peaks:
            self.assertAlmostEqual(p / peaks[0], 1.0, places=12)
        ks = [r["k_n_mm"] for r in rows]
        self.assertGreater(max(ks) / min(ks), 7.0)      # 同样观测，k 差 8 倍
        # 而 a_min 几乎不变：这就是"看起来一切正常"的来源
        a = [I.a_min(r["k_n_mm"], r["zeta"], 700.0, 4) for r in rows]
        self.assertLess(max(a) / min(a), 1.10)

    def test_a_min_only_constrains_the_product(self):
        """a_min 近似正比于 k*zeta：k 翻倍 zeta 减半，a_min 不变。
        所以"我用 a_min 对上了"这句话，只证明了乘积对。"""
        a1 = I.a_min(5.0e4, 0.02, 700.0, 4)
        a2 = I.a_min(1.0e5, 0.01, 700.0, 4)
        self.assertLess(abs(a2 / a1 - 1.0), 0.02)

    def test_kz_needs_a_second_observable_and_says_which(self):
        r = I.kz_identifiable_from_peak_alone()
        self.assertFalse(r["identifiable"])
        self.assertIn("半功率带宽", r["extra_observable_needed"])

    def test_one_force_has_infinitely_many_kt_ap_pairs(self):
        """F = Kt*ap*h：同一个 F 可以由多组 (Kt, ap) 产生。
        这是我上一轮说的"力逆不出 Kt"的可执行版本。"""
        F, h = 99.0, 0.0354
        pairs = [(kt, F / (kt * h)) for kt in (500.0, 700.0, 1200.0)]
        for kt, ap in pairs:
            self.assertAlmostEqual(kt * ap * h, F, places=6)
        # Kt 变大则 ap 变小：方向别写反（我第一版就写反过）
        self.assertLess(pairs[-1][1] / pairs[0][1], 0.5)

    def test_penalty_stiffness_converges_instead_of_being_right(self):
        """罚函数刚度越大，解越靠近真解：它是数值手段，不是物理量。
        所以"逆出 Kp"没有意义 —— 它该被推到无穷。"""
        r = I.penalty_sensitivity(1.0e5, 100.0)
        errs = [row["rel_err"] for row in r["rows"]]
        for a, b in zip(errs, errs[1:]):
            self.assertGreater(a, b)               # 单调改善
        self.assertGreater(errs[0], 0.5)           # 小 Kp 时错得离谱
        self.assertLess(errs[-1], 1e-5)            # 大 Kp 时基本正确
        # 100 N / 1e5 N/mm = 1e-3 mm = 1 um（不是 10 um）
        self.assertAlmostEqual(r["true_u_um"], 1.0, places=6)


class TestIdentifiable(unittest.TestCase):
    def test_kienzle_is_identifiable_and_the_span_decides_how_well(self):
        """可辨识的一例：取对数后线性，跨度越宽越准。窄区间实测 3.2%，宽区间 0.5%。"""
        narrow = I.kienzle_identifiable([0.02 + 0.005 * i for i in range(11)])
        wide = I.kienzle_identifiable([0.005 * 2 ** i for i in range(11)])
        self.assertTrue(narrow["ok"] and wide["ok"])
        self.assertLess(wide["kc1_rel_sd"], narrow["kc1_rel_sd"])
        self.assertGreater(wide["h_span_decades"], 2.5)
        self.assertLess(narrow["h_span_decades"], 1.0)
        self.assertLess(wide["kc1_rel_sd"], 0.01)     # 宽区间下 <1%

    def test_two_identical_thicknesses_carry_no_slope_information(self):
        r = I.kienzle_identifiable([0.05, 0.05, 0.05])
        self.assertFalse(r["ok"])

    def test_too_few_points_is_refused(self):
        self.assertFalse(I.kienzle_identifiable([0.02, 0.06])["ok"])

    def test_runout_vanishes_from_the_total_and_shows_per_tooth(self):
        """跳动在总和里抵消（净影响 0），在单齿上是 runout/fz = 20%。"""
        r = I.mean_vs_per_tooth(700.0, 2.0, 0.05, 4, runout_um=10.0)
        self.assertEqual(r["rel_effect_on_total_mean"], 0.0)
        self.assertAlmostEqual(r["per_tooth_rel_spread"], 0.2, places=9)


class TestVerdictTable(unittest.TestCase):
    def test_every_row_states_why(self):
        rows = I.verdict_table()
        self.assertGreaterEqual(len(rows), 6)
        for r in rows:
            self.assertTrue(r["why"].strip(), r["const"])
            self.assertIn(r["ok"], (True, False))

    def test_the_hard_cases_are_marked_not_identifiable(self):
        bad = {r["const"]: r["ok"] for r in I.verdict_table()}
        self.assertFalse(bad["k 与 zeta（只看峰值）"])
        self.assertFalse(bad["Kt 与 ap（只看一个力）"])
        self.assertFalse(bad["罚函数接触刚度 Kp"])


if __name__ == "__main__":
    unittest.main()
