# -*- coding: utf-8 -*-
"""CSG 的验收：体积对解析值，流形性单独一条，不许混。

这一层最容易骗自己的地方是**只看体积**。BSP 的布尔是体积精确的：
两个轴对齐盒体的交并差误差是 0.0，带孔件对棱柱解析体积也精确到 1e-16。
但结果网格上有 T 型接点（裂边），下游拿去做有限元会出问题。

所以判据拆成两条，而且它们分别报：
    体积   布尔做对了没有
    裂边   结果能不能给下游用
第二条在这条路线上永远不过（随棱数线性增长），这是路线的边界，
不是可以调掉的参数 —— 要无裂边就走 OCC（booleans.py）。
"""
import math
import unittest

from craftagent import csg


class TestBoxOps(unittest.TestCase):
    def test_union_volume(self):
        a = csg.box(0, 10, 0, 10, 0, 10)
        b = csg.box(5, 15, 5, 15, 5, 15)
        self.assertAlmostEqual(csg.volume(csg.union(a, b)), 1875.0, places=9)

    def test_intersect_volume(self):
        a = csg.box(0, 10, 0, 10, 0, 10)
        b = csg.box(5, 15, 5, 15, 5, 15)
        self.assertAlmostEqual(csg.volume(csg.intersect(a, b)), 125.0, places=9)

    def test_subtract_volume(self):
        a = csg.box(0, 10, 0, 10, 0, 10)
        b = csg.box(5, 15, 5, 15, 5, 15)
        self.assertAlmostEqual(csg.volume(csg.subtract(a, b)), 875.0, places=9)

    def test_volume_of_an_untouched_box(self):
        self.assertAlmostEqual(csg.volume(csg.box(0, 3, 0, 4, 0, 5)), 60.0,
                               places=9)

    def test_area_of_an_untouched_box(self):
        self.assertAlmostEqual(csg.area(csg.box(0, 3, 0, 4, 0, 5)),
                               2 * (12 + 15 + 20), places=9)


class TestDrilledPlate(unittest.TestCase):
    def test_volume_against_the_prism_closed_form(self):
        """带孔板的体积对**棱柱**（不是圆柱）的解析体积。拿圆柱的去比
        会差 1-2%，那是离散化误差，不是布尔做错了。"""
        plate = csg.box(0, 40, 0, 20, 0, 6)
        for n in (12, 48):
            d = csg.subtract(plate, csg.prism(20, 10, 5, -1, 7, n=n))
            want = 40 * 20 * 6 - 0.5 * n * 25 * math.sin(2 * math.pi / n) * 6
            self.assertAlmostEqual(csg.volume(d) / want, 1.0, places=12)

    def test_it_is_not_manifold_and_says_so(self):
        """这一条断言的是"如实报告"，不是"它是流形"。哪天修好了，
        这条会红 —— 那时候把它改成 assertTrue。"""
        r = csg.check()
        self.assertFalse(r["drilled_manifold"])
        for n, v in r["drilled"].items():
            self.assertFalse(v["manifold"])
            self.assertGreater(v["edges_bad"], 0)

    def test_crack_edges_grow_with_the_facet_count(self):
        """裂边随棱数增长，说明它是 T 型接点（几何逼近的必然产物），
        不是某个特定的巧合。"""
        r = csg.check()
        self.assertGreater(r["drilled"][48]["edges_bad"],
                           r["drilled"][12]["edges_bad"])


class TestCheckSummary(unittest.TestCase):
    def test_box_ops_are_exact(self):
        r = csg.check()
        self.assertTrue(r["ok"])
        for row in r["rows"]:
            self.assertLess(row["rel_err"], 1e-12)

    def test_the_two_verdicts_are_separate(self):
        r = csg.check()
        self.assertTrue(r["drilled_volume_ok"])
        self.assertFalse(r["drilled_manifold"])


if __name__ == "__main__":
    unittest.main()
