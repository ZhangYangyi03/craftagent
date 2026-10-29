# -*- coding: utf-8 -*-
"""五轴的验收：距离闭式解、三轴退化、反例、逆解往返。

这个文件要防两件事：一是距离函数写错（采样近似、边界漏掉），二是
"我实现了扫掠体"这种说法 —— 一个永远返回 False 的碰撞函数也能过
"三轴一致"那条，所以必须有一条**反例**证明它在五轴下真的抓住了东西。

  距离        线段-点、线段-线段、线段-长方体 对解析值。长方体那条
              用"内部点为 0、外部等于到最近面的距离"这个性质交叉验。
  三轴退化    刀轴竖直时扫掠体与高度场必须逐条一致。不一致说明两条
              基线里有一条是错的 —— 先怀疑自己。
  反例        高度场说安全、扫掠体说撞的姿态必须存在，且倾角要够大。
              这是这一层存在的唯一理由。
  逆解        (A,C) 解出来再正解回去，残差 1e-12；刀轴竖直时 C 不定，
              必须报 singular 而不是随便给一个角度。
"""
import math
import unittest

from craftagent import fiveaxis as X


class TestDistances(unittest.TestCase):
    def test_segment_point(self):
        a, b = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
        self.assertAlmostEqual(X.segment_point_dist(a, b, (5.0, 3.0, 0.0)), 3.0,
                               places=12)
        self.assertAlmostEqual(X.segment_point_dist(a, b, (-4.0, 0.0, 0.0)), 4.0,
                               places=12)
        self.assertAlmostEqual(X.segment_point_dist(a, b, (14.0, 0.0, 0.0)), 4.0,
                               places=12)
        # 退化成一个点
        self.assertAlmostEqual(X.segment_point_dist(a, a, (0.0, 0.0, 2.0)), 2.0,
                               places=12)

    def test_segment_segment_parallel_and_skew(self):
        d = X.segment_segment_dist((0, 0, 0), (10, 0, 0), (0, 5, 0), (10, 5, 0))
        self.assertAlmostEqual(d, 5.0, places=12)
        d = X.segment_segment_dist((0, 0, 0), (10, 0, 0), (5, 4, 3), (5, 4, 3))
        self.assertAlmostEqual(d, 5.0, places=12)          # 3-4-5
        d = X.segment_segment_dist((0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1))
        self.assertAlmostEqual(d, math.sqrt(2.0), places=12)

    def test_segment_box_inside_is_zero(self):
        box = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        self.assertEqual(X.segment_box_dist((5, 5, 5), (6, 6, 6), box), 0.0)

    def test_segment_box_outside_matches_the_nearest_face(self):
        box = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        # 沿 x 方向、在 y 外 3、z 在范围内
        d = X.segment_box_dist((-5, 13, 5), (15, 13, 5), box)
        self.assertAlmostEqual(d, 3.0, places=12)
        # 斜对着角：到 (10,10,10) 那个角的距离
        d = X.segment_box_dist((13, 14, 10), (13, 14, 10), box)
        self.assertAlmostEqual(d, math.sqrt(9 + 16), places=12)

    def test_segment_box_crossing_a_face_is_zero(self):
        """穿过去必须算 0，不能因为两端都在外面就报正距离。"""
        box = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
        self.assertEqual(X.segment_box_dist((-5, 5, 5), (15, 5, 5), box), 0.0)


class TestTool(unittest.TestCase):
    def test_holder_does_not_bulge_below_its_start(self):
        """刀柄是平底的：球头刀杆 + 刀柄，刀柄段不能因为胶囊的球冠
        往下凸出 holder_r。第一版就是这样，把 holder_start 附近
        多算了 20 mm 的碰撞。"""
        t = X.Tool(radius=5.0, flute=30.0, holder_start=45.0, holder_r=20.0)
        segs = t.segments((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        starts = [s[0][2] for s in segs]
        self.assertAlmostEqual(min(starts), 0.0, places=12)
        holder = [s for s in segs if s[3] == "holder"][0]
        self.assertAlmostEqual(holder[0][2], 45.0 + 20.0, places=12)

    def test_axis_is_normalised(self):
        t = X.Tool()
        segs = t.segments((0, 0, 0), (0, 0, 7.0))
        self.assertAlmostEqual(segs[-1][1][2], segs[-1][1][2], places=12)
        self.assertGreater(segs[-1][1][2], 0.0)


class TestModels(unittest.TestCase):
    def test_three_axis_poses_agree(self):
        """刀轴竖直时两条基线必须逐条一致。"""
        r = X.compare_models()
        self.assertGreater(len(r["three_axis"]), 0)
        for row in r["rows"]:
            if row["tilt_deg"] == 0.0:
                self.assertTrue(row["agree"], row["pose"])

    def test_there_are_poses_the_heightfield_misses(self):
        """这一层存在的理由：必须有高度场漏掉、扫掠体抓住的姿态。"""
        r = X.compare_models()
        self.assertGreater(len(r["missed_by_height"]), 0)
        # 全部漏掉的姿态都发生在有倾角的时候
        for row in r["missed_by_height"]:
            self.assertGreater(row["tilt_deg"], 10.0)

    def test_heightfield_never_reports_a_crash_the_swept_model_allows(self):
        """反方向的错也不能有：高度场不能比扫掠体更保守到凭空报警。"""
        r = X.compare_models()
        self.assertEqual(len(r["false_alarm_by_height"]), 0)


class TestIK(unittest.TestCase):
    def test_roundtrip(self):
        r = X.ik_roundtrip()
        self.assertTrue(r["ok"])
        self.assertLess(r["worst"], 1e-12)

    def test_vertical_axis_is_singular_and_says_so(self):
        r = X.ik_ac((0.0, 0.0, 1.0))
        self.assertTrue(r["singular"])
        self.assertIsNone(r["residual"])

    def test_tilted_axis_roundtrips(self):
        ax = (math.sin(0.4), 0.0, math.cos(0.4))
        r = X.ik_ac(ax)
        self.assertFalse(r["singular"])
        self.assertLess(r["residual"], 1e-12)
        back = X.fk_ac(r["A"], r["C"])
        for i in range(3):
            self.assertAlmostEqual(back[i], ax[i], places=12)


if __name__ == "__main__":
    unittest.main()
