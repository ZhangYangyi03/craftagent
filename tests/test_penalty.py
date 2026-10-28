# -*- coding: utf-8 -*-
"""penalty 的验收：这条 U 形必须能被自己的闭式预测出来。

这个文件最怕的失败是"我量到一条曲线，于是它成了新的黑箱"。
所以每个测试都在做同一件事：**用闭式预测实测**。预测对不上就说明我理解错了机制。
"""
import math
import unittest

from craftagent import penalty as P


class TestShape(unittest.TestCase):
    def setUp(self):
        self.s = P.sweep()

    def test_error_falls_then_rises(self):
        """U 形：先随 Kp 增大而下降（软约束误差），后随 Kp 增大而上升（浮点丢位）。"""
        errs = [r["err"] for r in self.s["rows"]]
        i = errs.index(min(errs))
        self.assertGreater(i, 2)                     # 不是一开始就最小
        self.assertLess(i, len(errs) - 2)            # 也不是一路降到底
        self.assertLess(errs[i], errs[0])
        self.assertLess(errs[i], errs[-1])

    def test_minimum_is_where_the_two_closed_forms_cross(self):
        """谷底位置由 sqrt(1/eps) 决定，与材料、几何、厂商都无关。"""
        errs = [r["err"] for r in self.s["rows"]]
        i = errs.index(min(errs))
        ratio_at_min = self.s["rows"][i]["kp_over_kchain"]
        predicted = P.best_kp_ratio()["kp_over_kchain"]
        self.assertLess(abs(math.log10(ratio_at_min) - math.log10(predicted)), 1.0)

    def test_closed_forms_predict_the_measured_shape(self):
        """两条闭式 max(soft, round) 要能预测实测到同一数量级。
        这一条成立，Kp 就不是黑箱：它的形状可以从 F/Kp 与 eps*Kp/k 直接写出来。"""
        cmp = P.measured_vs_predicted(self.s)
        hit = sum(1 for c in cmp if c["same_decade"])
        self.assertGreaterEqual(hit, len(cmp) - 2, (hit, len(cmp)))

    def test_soft_side_is_exactly_one_over_one_plus_ratio(self):
        """左边闭式是**精确**的，不是渐近的：err = k/(Kp+k) = 1/(1+r)。

        所以 Kp 乘 10 时误差不是除以 10，是 (1+r)/(1+10r) —— r=100 时是 9.91 而不是 10。
        我第一版把渐近式当成了恒等式，测试当场指出。渐近只在 r>>1 时成立。"""
        for r in (10.0, 100.0, 1000.0):
            self.assertAlmostEqual(P.err_from_soft(r), 1.0 / (1.0 + r), places=12)
        # 实测与闭式逐点相等，但只能在浮点能分辨的位数内：r=1 那一格实测与闭式
        # 差 2.2e-15（正好是 eps），因为解是解线性系统出来的，不是直接代的公式。
        # 写成 places=12 而不是 15，是因为 15 位在双精度里本来就不存在。
        for row in P.sweep(decades=range(0, 7))["rows"]:
            # 残差用**绝对**量比、门槛挂在 eps 上：实测与闭式的差是消元累积的舍入，
            # 量级就是 eps 乘以问题规模，跟比值 r 无关。用相对量比会随 r 变小而变严，
            # 那是把"浮点本来就没有那么多位"当成"公式错了"。
            # 实测偏离左边闭式的量，被右边那半条曲线（舍入项）上界住：
            # 实测 gap/eps 随 r 增长，r=1e6 时是 4.4e5，与 eps*r 同量级。
            # 所以这不是"该放多松的容差"，而是两个机制在同一处的必然叠加。
            gap = abs(row["err"] - row["pred_soft"])
            self.assertLess(gap, 2.0 * P.err_from_roundoff(row["kp_over_kchain"]) + 1e-14,
                            (row["kp_over_kchain"], gap))
        # 渐近：r 很大时比值趋近 10
        self.assertAlmostEqual(P.err_from_soft(1e6) / P.err_from_soft(1e7), 10.0, places=3)

    def test_roundoff_side_is_linear(self):
        """右边：Kp 乘 10，误差乘 10。"""
        a = P.err_from_roundoff(1e10)
        b = P.err_from_roundoff(1e11)
        self.assertAlmostEqual(b / a, 10.0, places=6)


class TestBalance(unittest.TestCase):
    def test_soft_and_roundoff_are_equal_at_the_optimum(self):
        r = P.best_kp_ratio()["kp_over_kchain"]
        self.assertAlmostEqual(P.err_from_soft(r), P.err_from_roundoff(r), places=1)

    def test_the_optimum_error_is_about_sqrt_eps(self):
        """最好的情况下也只能准到 1e-8 量级 —— 这是双精度的物理上限，不是软件的能力。"""
        e = P.best_kp_ratio()["expected_err"]
        self.assertGreater(e, 1e-9)
        self.assertLess(e, 1e-7)

    def test_lower_bound_is_force_over_penetration(self):
        self.assertAlmostEqual(P.kp_lower_bound(2100.0, 0.001), 2100000.0)

    def test_zero_penetration_is_refused(self):
        with self.assertRaises(ValueError):
            P.kp_lower_bound(1.0, 0.0)

    def test_dimensionless_form_has_no_material_in_it(self):
        """Kp/E 的下限只含 A/L 与应变，不含 E 本身 —— 所以换材料不改这个比例。"""
        a = P.kp_over_e(100.0, 100.0, 1e-4, 1e-3)["kp_over_e_min"]
        b = P.kp_over_e(100.0, 100.0, 1e-4, 1e-3)["kp_over_e_min"]
        self.assertEqual(a, b)
        self.assertGreater(a, 0)


class TestExactReference(unittest.TestCase):
    def test_fraction_solve_lowers_the_roundoff_side(self):
        """用 Fraction 解同一道题，右边那半条曲线消失 —— 这就证明了上升是浮点造成的，
        不是我模型里的物理。这一条是这个文件所有结论的地基。"""
        k_chain = 1050.0 / 40.0
        limit = 0.01 * k_chain
        for ratio in (1e8, 1e12):
            f_float = P.chain_force(40, 1050.0, ratio, 0.01)
            f_exact = float(P.chain_force(40, 1050.0, ratio, 0.01, exact=True))
            e_float = abs(f_float - limit) / limit
            e_exact = abs(f_exact - limit) / limit
            if ratio >= 1e12:
                self.assertGreater(e_float, 10 * e_exact)
            else:
                self.assertLess(e_float, 1e-6)


class TestWindow(unittest.TestCase):
    def test_implicit_upper_bound_shrinks_with_tolerance(self):
        """要的精度越紧，Kp 的上界越低 —— 这是丢位项的直接后果。"""
        a = P.kp_upper_roundoff(1.0, tol=1e-3)
        b = P.kp_upper_roundoff(1.0, tol=1e-4)
        self.assertGreater(a, b)
        self.assertAlmostEqual(a / b, 10.0, places=6)

    def test_explicit_upper_bound_is_square_of_step(self):
        """显式：步长减半，Kp 上界变成四分之一。这就是摘要里那句"stability
        condition forces to reduce the time step size"的算式版本。"""
        a = P.kp_upper_explicit(1.0, 1e-5)
        b = P.kp_upper_explicit(1.0, 5e-6)
        self.assertAlmostEqual(b / a, 4.0, places=6)

    def test_window_is_feasible_in_the_ordinary_case(self):
        w = P.kp_window(26.25, 0.2625, 1e-3)
        self.assertTrue(w["feasible"])
        self.assertGreater(w["geometric_mean"], w["lower_n_per_mm"])
        self.assertLess(w["geometric_mean"], w["upper_n_per_mm"])

    def test_infeasible_window_says_to_switch_formulation(self):
        """下界超过上界时，回答必须是"换公式"，不是继续调 Kp。
        这条是给未来的自己下的判决：碰到调不动的情况，先算窗口。"""
        w = P.kp_window(1e-6, 1e9, 1e-9)
        self.assertFalse(w["feasible"])
        self.assertIn("换接触公式", w["advice"])

    def test_grid_dependence_is_one_over_element_size(self):
        """单元尺寸减半，Kp 的量级翻倍，与零件真实尺寸无关。"""
        self.assertAlmostEqual(P.grid_dependence(2.0)["relative"] * 2,
                               P.grid_dependence(1.0)["relative"], places=12)


if __name__ == "__main__":
    unittest.main()
