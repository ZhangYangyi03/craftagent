# -*- coding: utf-8 -*-
"""offset 的验收：三条外部参照，加一条"必须报错"的用例。

这个文件最怕的失败是"偏出来一条看着很顺的轮廓"。等距曲线的错法几乎都长得
像对的：方向反了会给出一个更小的件、斜接会把角拉长、删环删错会少一整块，
而三种情况的轮廓都是闭合、光滑、无自交的。

所以每一条都对着**不是它自己**的东西：
  凸件面积对 Steiner 公式（凸集的数学事实），
  任意件面积对网格采样（另一套完全不同的算法），
  每一点到原轮廓的距离对 |dist|（等距曲线的定义）。

第 8 条是反过来的：窄槽向内偏置到断成两截时，**正确答案是报失败**。
一条硬凑出来的单环轮廓会通过"闭合、无自交、距离对"三项检查，
只有网格数能看出它少了一半。这条测试盯着那个"报出来"的行为本身。
"""
import math
import unittest

from craftagent import offset as O

RECT = [(0.0, 0.0), (60.0, 0.0), (60.0, 84.0), (0.0, 84.0)]
L_SHAPE = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0),
           (20.0, 20.0), (20.0, 40.0), (0.0, 40.0)]
# 40x20 的板，顶上开一条 4 mm 宽、16 mm 深的槽
SLOT = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (22.0, 20.0),
        (22.0, 4.0), (18.0, 4.0), (18.0, 20.0), (0.0, 20.0)]


class TestAgainstMath(unittest.TestCase):
    def test_outward_offset_of_a_rectangle_matches_steiner(self):
        """凸集外偏 r 的面积是 A + P*r + pi*r^2。这是数学事实，不是我的实现。

        只有圆角连接满足它 —— 斜接会多出四个方角，面积偏大（见下一条）。
        """
        r = 5.0
        res = O.offset_polygon(RECT, r)
        want = O.area(RECT) + O.perimeter(RECT) * r + math.pi * r * r
        self.assertLess(abs(O.area(res["pts"]) - want) / want, 1e-4, res["pts"][:4])
        self.assertTrue(res["ok"], res)

    def test_miter_is_not_an_equidistant_curve(self):
        """斜接顶点到轮廓的距离是 7.07 而不是 5 —— 它本来就不是等距曲线。

        这条测试存在的理由：`cam.offset_contour` 一直在做斜接，而报告里
        只说"近似"。把差多少写进测试，那个"近似"就有了数字。
        """
        res = O.offset_polygon(RECT, 5.0, join="miter")
        self.assertAlmostEqual(O.area(res["pts"]), 70.0 * 94.0, places=6)
        self.assertAlmostEqual(res["max_dist_err"], 5.0 * (math.sqrt(2) - 1), places=6)
        self.assertTrue(res["simple"])

    def test_inward_offset_of_a_rectangle_is_exact(self):
        """内偏没有圆弧，四个角是四条偏移线的交点，面积必须一位不差。"""
        res = O.offset_polygon(RECT, -5.0)
        self.assertAlmostEqual(O.area(res["pts"]), 50.0 * 74.0, places=9)
        self.assertLess(res["max_dist_err"], 1e-12)


class TestDefinition(unittest.TestCase):
    def test_every_offset_point_is_exactly_dist_from_the_boundary(self):
        """等距曲线的定义。这一条不过，后面所有面积数都没有意义。"""
        for src, d in ((RECT, 5.0), (RECT, -5.0), (L_SHAPE, 4.0), (L_SHAPE, -3.0)):
            res = O.offset_polygon(src, d)
            worst = max(abs(O.distance_to_boundary(p, src) - abs(d))
                        for p in res["pts"])
            self.assertLess(worst, 1e-9, (src, d, worst))

    def test_winding_does_not_change_the_result(self):
        """顺时针输入与逆时针输入必须给出同一个件。

        绕向搞反的表现是"整条轮廓往里偏"——件小一圈，程序全合法。
        """
        a = O.offset_polygon(RECT, 5.0)["pts"]
        b = O.offset_polygon(list(reversed(RECT)), 5.0)["pts"]
        self.assertAlmostEqual(O.area(a), O.area(b), places=6)
        # 70*94 是斜接的面积，减去四个角"方角换圆角"的差 (4-pi)*r^2。
        # 容差 0.5 mm^2 是圆弧用弦代替带来的（弦高 0.005，四角合计 0.1）。
        self.assertAlmostEqual(O.area(a), 70.0 * 94.0 - (4 - math.pi) * 25.0, delta=0.5)

    def test_output_never_self_intersects(self):
        """自交的表现是"轮廓多一小段回头路"，而程序仍然合法。"""
        for src in (RECT, L_SHAPE, SLOT):
            for d in (2.0, -2.0, 6.0):
                res = O.offset_polygon(src, d)
                self.assertTrue(res["simple"], (src, d))
                self.assertIsNone(O.first_self_intersection(res["pts"]))


class TestAgainstGrid(unittest.TestCase):
    def test_L_shape_matches_the_grid_oracle(self):
        """网格不知道什么是偏移线，它只数点到边的距离。1% 以内算一致。"""
        for d in (4.0, -3.0):
            v = O.verify(L_SHAPE, d, n=300)
            self.assertTrue(v["ok"], v)
            self.assertLess(v["rel_err"], 0.01, v)

    def test_a_narrow_slot_that_closes_is_still_one_contour(self):
        """槽宽 4、向外偏 3：槽被填平，但整块还是连的，单环表示得了。"""
        v = O.verify(SLOT, 3.0, n=300)
        self.assertTrue(v["ok"], v)
        self.assertTrue(v["simple"], v)
        self.assertEqual(v["loops_removed"], 1, v)   # 确实裁过一次

    def test_a_narrow_slot_that_splits_must_be_reported_not_faked(self):
        """槽宽 4、向内偏 3：腐蚀后左右两块互不相连，**单环表示不了**。

        这里断言的不是"结果对"，是"结果被报成不对"。硬凑一条环会通过
        闭合、无自交、距离三项检查，只有网格数看得出来它少了一半 ——
        所以正确行为是 ok=False，且 rel_err 大到藏不住。
        """
        v = O.verify(SLOT, -3.0, n=300)
        self.assertFalse(v["ok"], v)
        self.assertGreater(v["rel_err"], 0.2, v)
        self.assertTrue(v["suspect"], v)
        self.assertEqual(v["reason"], "loop_removed")


class TestArc(unittest.TestCase):
    def test_arc_sampling_follows_chord_height_not_a_fixed_count(self):
        """弦高容差放宽，点数必须变少；收紧，点数变多。半径大时更少。

        固定段数的实现会在这三条里至少错一条。
        """
        big = O.offset_polygon(RECT, 20.0, arc_tol=0.05)
        fine = O.offset_polygon(RECT, 20.0, arc_tol=0.001)
        self.assertLess(len(big["pts"]), len(fine["pts"]))
        # 半径 20 的圆弧，弦高 0.005 -> 每段约 0.63 rad -> 四角共约 40 点
        self.assertGreater(len(fine["pts"]), 20)

    def test_circle_offset_matches_the_analytic_radius(self):
        """正 64 边形外偏 4：面积必须逼近 pi*(30+4)^2。"""
        c = [(30.0 * math.cos(2 * math.pi * k / 64),
              30.0 * math.sin(2 * math.pi * k / 64)) for k in range(64)]
        res = O.offset_polygon(c, 4.0)
        want = math.pi * 34.0 ** 2
        self.assertLess(abs(O.area(res["pts"]) - want) / want, 0.01)
        self.assertLess(res["max_dist_err"], 1e-9)


class TestDegenerate(unittest.TestCase):
    def test_zero_offset_returns_the_input(self):
        res = O.offset_polygon(RECT, 0.0)
        self.assertEqual(res["kind"], "degenerate")
        self.assertFalse(res["ok"])

    def test_duplicate_and_collinear_points_are_dropped(self):
        """共线点留着会让远处的交点把轮廓甩出一个尖刺，而距离检查抓不住。"""
        noisy = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0), (60.0, 84.0), (0.0, 84.0)]
        res = O.offset_polygon(noisy, 5.0)
        self.assertEqual(len(O.clean(noisy)), 4)
        self.assertLess(abs(O.area(res["pts"]) - O.area(O.offset_polygon(RECT, 5.0)["pts"])),
                        1e-9)


if __name__ == "__main__":
    unittest.main()
