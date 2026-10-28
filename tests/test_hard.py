# -*- coding: utf-8 -*-
"""标准与高难度参数的验收：每一条都要能自己被推翻。

这一层最容易滑向"看起来很专业"。所以每条测试都在做同一件事：
要么用一个独立的式子把同一个数再算一遍（两条路对上才算数），
要么用一个已知的极端情形把公式钉住（L 变 2 倍，挠度必须变 8 倍）。
"""
import math
import unittest

from craftagent import hard, standards


class TestStandards(unittest.TestCase):
    def test_it7_at_50mm_matches_the_classic_value(self):
        """IT7 在 30-50 段是 25 um，这是最常被引用的一格。"""
        r = standards.it_tolerance_um(50, 7)
        self.assertAlmostEqual(r["um"], 25.0, delta=1.0)
        self.assertEqual(r["handbook_um"], 25)

    def test_formula_matches_handbook_above_3mm(self):
        """6 mm 以上，公式与手册最大差 1 um。差得多就说明公式或参照值有一个错。"""
        worst = 0.0
        for (hi, g), ref in standards.HANDBOOK_REF.items():
            if hi <= 3:
                continue
            r = standards.it_tolerance_um(hi, g)
            worst = max(worst, abs(r["delta_um"]))
        self.assertLessEqual(worst, 1.0, worst)

    def test_first_step_is_declared_a_table_region_not_extrapolated(self):
        """<=3 mm 段公式最多差 5 um，所以它必须自己说"请查表"。"""
        r = standards.it_tolerance_um(2.0, 10)
        self.assertFalse(r["formula_applies"])
        self.assertIn("查表", r["warning"])
        self.assertGreaterEqual(abs(r["delta_um"]), 3.0)

    def test_grade_scaling_is_exact(self):
        """IT8/IT7 = 25/16，与尺寸无关。这是"用公式重算"能被验算的地方。"""
        for size in (10, 30, 100):
            a = standards.it_tolerance_um(size, 8)["raw_um"]
            b = standards.it_tolerance_um(size, 7)["raw_um"]
            self.assertAlmostEqual(a / b, 25.0 / 16.0, places=9)

    def test_preferred_numbers_start_from_the_series(self):
        r10 = standards.preferred_numbers(1, 10, "R10")
        self.assertAlmostEqual(r10[0], 1.0, places=3)
        self.assertAlmostEqual(r10[3], 10 ** 0.3, places=3)


class TestStability(unittest.TestCase):
    """颤振叶瓣是这一层最值钱的一条，所以验收也最硬：解析解与网格解必须对上。"""

    def test_grid_minimum_matches_the_analytic_unconditional_depth(self):
        stab = hard.stability_lobes(fn_hz=600.0, k_n_mm=5.0e4, zeta=0.025,
                                    kt_n_mm2=700.0, teeth=4)
        analytic = hard.unconditional_depth_mm(5.0e4, 0.025, 700.0, 4)
        self.assertLess(abs(stab["worst"]["a_lim_mm"] / analytic - 1.0), 0.05,
                        (stab["worst"], analytic))

    def test_unconditional_depth_is_linear_in_stiffness(self):
        a1 = hard.unconditional_depth_mm(5.0e4, 0.025, 700.0, 4)
        a2 = hard.unconditional_depth_mm(1.0e5, 0.025, 700.0, 4)
        self.assertAlmostEqual(a2 / a1, 2.0, places=9)

    def test_more_damping_is_more_stable(self):
        a1 = hard.unconditional_depth_mm(5.0e4, 0.02, 700.0, 4)
        a2 = hard.unconditional_depth_mm(5.0e4, 0.05, 700.0, 4)
        self.assertGreater(a2, a1)

    def test_a_shallow_cut_needs_no_speed_choice(self):
        """ap 低于无条件稳定切深时，任何转速都该放行。反之就要挑转速。"""
        stab = hard.stability_lobes(fn_hz=600.0, k_n_mm=5.0e4, zeta=0.025,
                                    kt_n_mm2=700.0, teeth=4)
        shallow = hard.avoid_bands(stab, 0.5)
        deep = hard.avoid_bands(stab, 3.0)
        self.assertEqual(shallow["avoid"], [])
        self.assertGreater(len(deep["avoid"]), 0)
        self.assertGreater(deep["bad_fraction"], shallow["bad_fraction"])

    def test_lobes_are_declared_one_dimensional_and_optimistic(self):
        stab = hard.stability_lobes()
        self.assertIn("偏乐观", stab["note"])
        self.assertEqual(stab["grades"]["zeta"], "must-measure")


class TestMechanics(unittest.TestCase):
    def test_deflection_scales_as_length_cubed(self):
        a = hard.tool_deflection_mm(100.0, 40.0, 10.0)["deflection_mm"]
        b = hard.tool_deflection_mm(100.0, 80.0, 10.0)["deflection_mm"]
        self.assertAlmostEqual(b / a, 8.0, places=6)

    def test_deflection_scales_as_diameter_to_the_fourth(self):
        a = hard.tool_deflection_mm(100.0, 40.0, 10.0)["deflection_mm"]
        b = hard.tool_deflection_mm(100.0, 40.0, 20.0)["deflection_mm"]
        self.assertAlmostEqual(b / a, 1.0 / 16.0, places=6)

    def test_deflection_number_worth_knowing(self):
        """D10 伸出 40 mm、100 N 侧向力：挠度在十几 um 量级。
        这个量级才是"精工"真正的约束 —— IT7 在 50 mm 段是 25 um。"""
        d = hard.tool_deflection_mm(100.0, 40.0, 10.0)["deflection_mm"] * 1000.0
        self.assertTrue(5.0 < d < 30.0, d)

    def test_thermal_growth_is_the_thing_that_eats_the_tolerance(self):
        """500 mm 钢件温差 1 K 长 5.75 um；IT7 在 50 mm 段是 25 um。

        最初我写的是"4 K 就吃掉公差"，实测 4 K = 23 um，还差 2 um 没超。
        5 K = 28.8 um 才超。留成这个测试，是因为下一次有人想当然地说"4 K"
        的时候，它会当场指出实际是 4~5 K 之间。"""
        g = hard.thermal_growth_um(500.0, 1.0)["growth_um"]
        self.assertAlmostEqual(g, 5.75, places=6)
        tol = standards.it_tolerance_um(50, 7)["um"]
        self.assertLess(4 * g, tol)          # 4 K 还没超
        self.assertGreater(5 * g, tol)       # 5 K 超了

    def test_residual_height_exact_vs_approximation(self):
        """近似式 s^2/(8R) 在 s<<R 时对，误差随 s/R 单调涨，s->2R 时比值趋近 2。

        原来这里写"3 倍"，实测最大值是 2 倍（s=2R：精确 0.5R 以上，近似 0.5R）。
        测试按核实过的行为写，不按好听的说法写。"""
        small = hard.residual_height_um(0.2, 5.0)
        self.assertLess(abs(small["height_um"] / small["approx_um"] - 1.0), 0.01)
        mid = hard.residual_height_um(4.0, 5.0)      # s = 0.8 R
        self.assertGreater(mid["height_um"] / mid["approx_um"], 1.0)
        edge = hard.residual_height_um(9.5, 5.0)     # s = 1.9 R
        self.assertGreater(edge["height_um"] / edge["approx_um"], 1.4)
        self.assertLess(edge["height_um"] / edge["approx_um"], 2.05)

    def test_stepover_for_ra_round_trips(self):
        s = hard.stepover_for_ra(5.0, 1.6)["stepover_mm"]
        back = hard.residual_height_um(s, 5.0)["height_um"]
        self.assertAlmostEqual(back, 1.6, places=6)

    def test_min_fz_is_a_lower_bound_and_scales_with_edge_radius(self):
        a = hard.min_fz_mm(8.0, "steel")
        b = hard.min_fz_mm(16.0, "steel")
        self.assertAlmostEqual(b["fz_min_mm"] / a["fz_min_mm"], 2.0, places=9)
        self.assertEqual(a["grade"], "must-measure")


class TestTaylor(unittest.TestCase):
    def test_two_points_recover_a_planted_pair(self):
        C, n = 400.0, 0.25
        pts = [(vc, (C / vc) ** (1.0 / n)) for vc in (120.0, 180.0)]
        r = hard.fit_taylor(pts)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["n"], n, places=6)
        self.assertAlmostEqual(r["C"], C, places=4)

    def test_one_point_is_refused(self):
        self.assertFalse(hard.fit_taylor([(120.0, 60.0)])["ok"])

    def test_life_falls_with_speed(self):
        a = hard.taylor_life(200, 0.25, 100)["life_min"]
        b = hard.taylor_life(200, 0.25, 200)["life_min"]
        self.assertGreater(a, b)


if __name__ == "__main__":
    unittest.main()
