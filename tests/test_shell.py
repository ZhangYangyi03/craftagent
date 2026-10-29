# -*- coding: utf-8 -*-
"""shell.py 的验收：五条，参照全在代码之外。

  简支方板   0.00406 q a^4 / D（Navier 解）
  固支方板   0.00126 q a^4 / D
  膜单轴     位移精确等于 sigma x / E 与 -nu sigma y / E
  零空间     24 个自由度里必须恰好 6 个刚体模态
  薄极限     t/a 从 0.1 到 0.001 归一化挠度不能塌 —— 抓剪切锁死

第四条和第五条是这一层最容易被糊过去的地方：锁死的单元在中等厚度下
看起来完全正常，只有把 t 调薄才会塌，而"看起来正常"正是最危险的。
"""
import unittest

from craftagent import shell as S


class TestMembrane(unittest.TestCase):
    def test_uniaxial_tension_is_exact(self):
        """常应变场必须被精确复现 —— 这不是"接近"，是一位不差。"""
        r = S.membrane_tension(n=4)
        self.assertTrue(r["converged"], r)
        self.assertLess(r["u_rel"], 1e-10, r)
        self.assertLess(r["v_rel"], 1e-10, r)

    def test_poisson_contraction_must_not_be_suppressed(self):
        """横向收缩若被边界条件挡住，v 会正好少一半。

        第一版把 v = 0 加在整条 y=0 边上，实测 v 只有解析值的一半，
        而 u 完全正确 —— 只看 u 的话这个错误永远不会暴露。
        """
        r = S.membrane_tension(n=4)
        self.assertGreater(abs(r["v"]), 0.0)
        self.assertLess(r["v_rel"], 1e-6, r)


class TestBending(unittest.TestCase):
    def test_simply_supported_plate_matches_navier(self):
        r = S.simply_supported(n=12)
        self.assertTrue(r["converged"], r)
        self.assertLess(abs(r["ratio"] - 1.0), 0.02, r)

    def test_clamped_plate_matches_plate_theory(self):
        r = S.clamped(n=12)
        self.assertTrue(r["converged"], r)
        self.assertLess(abs(r["ratio"] - 1.0), 0.02, r)

    def test_clamped_is_about_three_times_stiffer_than_simply_supported(self):
        """两个解必须**不同**。曾经把简支写成固支（多固定了法向转角），
        两个数一模一样，而各自对着自己的理论值都能挑不出毛病。"""
        a = S.simply_supported(n=8)["w"]
        b = S.clamped(n=8)["w"]
        self.assertGreater(a / b, 2.5, (a, b))
        self.assertLess(a / b, 4.0, (a, b))

    def test_refinement_converges_from_below(self):
        rs = [S.simply_supported(n=n)["ratio"] for n in (4, 8, 12)]
        self.assertLess(rs[0], rs[1], rs)
        self.assertLess(rs[1], rs[2] + 1e-3, rs)
        self.assertLess(abs(rs[-1] - 1.0), 0.01, rs)


class TestRigidModes(unittest.TestCase):
    def test_nullity_is_exactly_six(self):
        """平面壳的 24 个自由度里刚体运动恰好 6 个。

        多一个说明钻头自由度没被罚住（矩阵奇异，解里会出现自由的
        theta_z 场）；少一个说明某个刚体模态被罚出了能量 —— 后者在
        整体转动时会凭空产生应力。
        """
        r = S.rigid_modes()
        self.assertEqual(r["nullity"], 6, r)

    def test_drill_penalty_does_not_kill_the_rigid_rotation(self):
        """罚 theta_z - omega_z，不是罚 theta_z。

        罚 theta_z 本身会把绕 z 的刚体转动变成有能量的模态，零空间掉到 5。
        """
        r = S.rigid_modes()
        self.assertEqual(r["size"] - r["rank"], 6, r)


class TestShearLocking(unittest.TestCase):
    def test_thin_limit_does_not_collapse(self):
        """t/a 从 0.1 降到 0.001，归一化挠度必须稳在 1 附近。

        锁死的单元在这里会掉到接近零：剪应变项在 t -> 0 时占主导，
        位移插值无法让剪应变趋于零，于是单元拒绝弯曲。
        """
        rows = S.thin_limit(n=6)
        for r in rows:
            self.assertTrue(r["converged"], r)
            self.assertGreater(r["ratio"], 0.9, r)
            self.assertLess(r["ratio"], 1.2, r)

    def test_thick_plate_is_slightly_more_flexible_than_thin_plate_theory(self):
        """t/a = 0.1 时剪切变形有贡献，挠度比薄板理论**大**一点。

        方向必须对：Mindlin 板比 Kirchhoff 板软。反过来就是符号错了。
        """
        rows = S.thin_limit(n=6)
        self.assertGreater(rows[0]["ratio"], rows[1]["ratio"], rows)


if __name__ == "__main__":
    unittest.main()
