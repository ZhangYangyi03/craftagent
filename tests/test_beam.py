# -*- coding: utf-8 -*-
"""beam 的验收：四条解析解，一条"细化不变"。

欧拉-伯努利梁在**点载荷下的节点位移是解析精确的**，所以这里的容差是 1e-10
而不是 1e-2。一个只能用"误差 5%"来验收的梁单元，通常是错的。

"细化不变"那一条最有信息量：分 1 / 2 / 4 个单元结果必须一位不差。变了说明
坐标变换或装配有错，而错的表现通常只是"稍微软了一点"，看起来像网格粗细的
正常差别 —— 那是最容易被接受下来的一种错。
"""
import math
import unittest

from craftagent import beam as B


class TestAnalytic(unittest.TestCase):
    def test_cantilever_tip_matches_PL3_over_3EI(self):
        r = B.cantilever()
        self.assertAlmostEqual(r["tip"], r["theory"], places=10)
        self.assertAlmostEqual(r["theta"], r["theta_theory"], places=10)

    def test_axial_matches_PL_over_EA(self):
        r = B.axial(3)
        self.assertAlmostEqual(r["tip"], r["theory"], places=10)

    def test_simply_supported_matches_PL3_over_48EI(self):
        r = B.simply_supported(8)
        self.assertAlmostEqual(r["mid"], r["theory"], places=10)

    def test_mesh_refinement_does_not_change_the_answer(self):
        """1 / 2 / 4 个单元的端部挠度必须相同。"""
        vals = [B.cantilever(n=n)["tip"] for n in (1, 2, 4)]
        self.assertAlmostEqual(vals[0], vals[1], places=12)
        self.assertAlmostEqual(vals[1], vals[2], places=12)


class TestStructure(unittest.TestCase):
    def test_one_element_has_exactly_three_rigid_modes(self):
        r = B.rigid_modes()
        self.assertEqual(r["nullity"], 3, r)

    def test_bending_dofs_are_one_two_four_five_not_one_to_four(self):
        """弯曲块铺到 (1,2,4,5)，不是 (1,2,3,4)。

        按 1,2,3,4 铺过去，t2 那一行会是全零（等于把第二个节点铰接），
        而刚度矩阵仍然对称 —— 所以只能直接查那一行。
        """
        k = B.element_stiffness_local(210000.0, 100.0, 1000.0, 100.0)
        self.assertNotEqual(k[5][5], 0.0)
        self.assertAlmostEqual(k[5][5], 4 * 210000.0 * 1000.0 / 100.0, places=6)
        # u2 与弯曲不耦合
        self.assertEqual(k[3][4], 0.0)
        self.assertEqual(k[3][5], 0.0)

    def test_inclined_member_splits_the_load_into_axial_and_bending(self):
        """45 度杆：竖向力分解成轴向压 + 横向弯，两个理论值分别对。

        角度搞反（c 与 s 互换）时 45 度的结果只差一个因子，看起来像网格效应；
        而 0 度和 90 度两种极端下它恰好还是对的 —— 所以只测极端值会漏掉它。
        """
        r = B.inclined()
        self.assertAlmostEqual(r["along"], r["along_theory"], places=10)
        self.assertAlmostEqual(r["across"], r["across_theory"], places=10)

    def test_zero_length_element_is_refused(self):
        with self.assertRaises(ValueError):
            B.element_stiffness_global((0.0, 0.0), (0.0, 0.0), 210000.0, 100.0, 1000.0)


class TestBoundary(unittest.TestCase):
    def test_shear_deformation_matters_for_short_members(self):
        """L/h = 10 时剪切挠度占 1.6%，L/h = 2 时占 19.5%。

        这个数存在的意义：让人自己判断什么时候该换 Timoshenko 单元，
        而不是在文档里写一句"细长杆适用"。
        """
        long_ = B.shear_note(L=100.0, h=10.0)
        short = B.shear_note(L=20.0, h=10.0)
        self.assertLess(long_["ratio"], 0.02)
        self.assertGreater(short["ratio"], 0.15)
        self.assertGreater(short["ratio"] / long_["ratio"], 10.0)


if __name__ == "__main__":
    unittest.main()
