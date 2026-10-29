# -*- coding: utf-8 -*-
"""非线性的验收：几何非线性对弹性线，材料非线性对双线性硬化闭式解。

这个文件要防的失败是"加了几步循环的线性求解器"。那种东西会收敛、
会给出光滑的曲线、看起来完全正常，但它解的是错的方程。所以：

  弹性线      端部纯力偶 -> 半径 R=EI/M 的圆弧，任意大挠度都成立。
              turn=pi/2（转 90 度）时线性理论差 23%，所以这个算例能
              区分真非线性 and 假非线性；turn=0.5 时只差 2%，区分不了。
  客观性      刚体转动下内力必须为零（1e-9 相对量级），否则单元是
              "跟着坐标轴走"的，一转动就产生假应变。
  收敛阶      残差每步降到上一步的平方量级。切线错就退化成一阶，
              这一条比"收敛了没有"严格得多 —— 错的切线也会收敛，
              只是收敛到另一个方程的解。注意差分切线的噪声下限会让
              最后几步的观测阶掉下来，所以判据取**出现过**的最高阶。
  双线性杆    两段直线，拐点 sigma_y，第二段斜率 EH/(E+H)。闭式。
"""
import math
import unittest

from craftagent import nonlinear as N


class TestElastica(unittest.TestCase):
    def test_tip_displacement_against_circular_arc(self):
        """端部纯力偶：精确解是圆弧，挠度 R(1-cos)，端部水平位移
        R sin(L/R) - L。比的是**位移**不是位置 —— 拿位置去比会差 157%，
        而挠度和转角都对，只看挠度这个错永远不会暴露。"""
        r = N.elastica_end_moment(n=20, turn=0.5, steps=6)
        self.assertTrue(r["ok"])
        self.assertLess(r["v_rel_err"], 2e-3)
        self.assertLess(r["u_rel_err"], 5e-3)

    def test_refines_towards_the_exact_arc(self):
        errs = [N.elastica_end_moment(n=n, turn=0.5)["v_rel_err"]
                for n in (10, 20, 40)]
        self.assertGreater(errs[0], errs[1])
        self.assertGreater(errs[1], errs[2])
        self.assertLess(errs[-1], 2e-4)

    def test_quarter_turn_is_where_linear_theory_breaks(self):
        """转 90 度：线性理论差 23%。这条同时证明这个算例有区分力。"""
        r = N.elastica_end_moment(n=20, turn=math.pi / 2, steps=8)
        self.assertTrue(r["ok"])
        self.assertGreater(r["linear_rel_err"], 0.20)
        self.assertLess(r["v_rel_err"], 2e-3)

    def test_inextensible(self):
        """纯力偶不产生轴力，变形后折线长度必须还是 L。"""
        r = N.elastica_end_moment(n=20, turn=math.pi / 2, steps=8)
        self.assertLess(r["length_err"], 1e-6)


class TestObjectivity(unittest.TestCase):
    def test_no_internal_force_under_rigid_rotation(self):
        r = N.objectivity()
        self.assertTrue(r["ok"])
        self.assertLess(r["rel"], 1e-9)


class TestNewton(unittest.TestCase):
    def test_residual_drops_quadratically(self):
        """切线对 -> 二阶；切线错 -> 一阶。取观测到的最高阶，因为差分
        切线的噪声下限（约 1e-11 相对）会把最后几步压成一阶。"""
        r = N.elastica_end_moment(n=20, turn=math.pi / 2, steps=8)
        p = N.quadratic(r["history"])
        self.assertGreaterEqual(len(p), 3)
        self.assertGreaterEqual(max(p), 1.7)

    def test_nonconvergence_is_reported_not_hidden(self):
        """迭代上限给 1 步时必须说没收敛，不能返回一个"看起来收敛"的解。"""
        m = N.corotational_beam(n=8)
        f = [0.0] * (3 * len(m["nodes"]))
        tip = len(m["nodes"]) - 1
        f[3 * tip + 2] = m["EI"] / 50.0
        r = N.newton(m, f, {0: 0.0, 1: 0.0, 2: 0.0}, steps=1, maxit=1)
        self.assertFalse(r["ok"])
        self.assertTrue(r.get("reason"))

    def test_tighter_than_the_noise_floor_is_not_called_converged(self):
        """tol 比差分切线的噪声下限还紧时，不能声称收敛。它可能报
        "未收敛"，也可能报 stagnated —— 两者都算如实，假装 ok 不行。"""
        m = N.corotational_beam(n=20)
        f = [0.0] * (3 * len(m["nodes"]))
        tip = len(m["nodes"]) - 1
        f[3 * tip + 2] = m["EI"] / 200.0
        r = N.newton(m, f, {0: 0.0, 1: 0.0, 2: 0.0}, steps=1, maxit=20,
                     tol=1e-16)
        self.assertFalse(r["ok"])
        self.assertTrue(r.get("stagnated") or r.get("reason"))


class TestPlastic(unittest.TestCase):
    def test_bilinear_hardening_closed_form(self):
        r = N.plastic_bar_check()
        self.assertTrue(r["ok"])
        self.assertLess(r["max_rel_err"], 1e-12)

    def test_two_slopes_are_E_and_EH_over_E_plus_H(self):
        r = N.plastic_bar_check(E=210000.0, sy=250.0, H=2000.0)
        self.assertAlmostEqual(r["slope_pre"] / 210000.0, 1.0, places=6)
        want = 210000.0 * 2000.0 / (210000.0 + 2000.0)
        self.assertAlmostEqual(r["slope_post"] / want, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
