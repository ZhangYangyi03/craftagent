# -*- coding: utf-8 -*-
"""bvh 的验收：剪枝必须不改变答案，加速必须随规模增长。

这个文件最怕的成功是"它变快了"。剪枝剪错会让返回的距离偏大，
而偏大的距离看起来完全正常 —— 一个更远的点也是点。所以每一条都在对拍。
"""
import random
import unittest

from craftagent import bvh


def _pts(n, seed=7, span=(100.0, 100.0, 10.0)):
    rnd = random.Random(seed)
    return [(rnd.random() * span[0], rnd.random() * span[1], rnd.random() * span[2])
            for _ in range(n)]


class TestAgreesWithBruteforce(unittest.TestCase):
    def test_nearest_matches_bruteforce_exactly(self):
        """剪枝唯一可接受的后果是少算，不是算错。距离必须**逐位**相等。"""
        pts = _pts(3000)
        root = bvh.build(pts)
        rnd = random.Random(11)
        for _ in range(60):
            q = (rnd.random() * 100, rnd.random() * 100, rnd.random() * 10)
            d, i = bvh.nearest(pts, root, q)
            d2, i2 = bvh.nearest_bruteforce(pts, q)
            self.assertEqual(i, i2)
            self.assertAlmostEqual(d, d2, places=12)

    def test_nearest_matches_when_query_is_far_outside_the_box(self):
        """查询点在包围盒外时，剪枝最容易过早放弃。这一条专门打那里。"""
        pts = _pts(2000)
        root = bvh.build(pts)
        for q in ((-5000.0, -5000.0, -5000.0), (5000.0, 5000.0, 5000.0),
                  (50.0, 50.0, 1e6)):
            d, i = bvh.nearest(pts, root, q)
            d2, _ = bvh.nearest_bruteforce(pts, q)
            self.assertAlmostEqual(d, d2, places=12)

    def test_radius_search_matches_bruteforce(self):
        pts = _pts(2000)
        root = bvh.build(pts)
        rnd = random.Random(13)
        for _ in range(20):
            q = (rnd.random() * 100, rnd.random() * 100, rnd.random() * 10)
            got = sorted(bvh.radius_search(pts, root, q, 8.0))
            want = sorted(i for i, p in enumerate(pts)
                          if (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2 <= 64.0)
            self.assertEqual(got, want)

    def test_duplicate_points_still_return_one_of_them(self):
        """重合点：最近距离必须是 0，下标是哪一个是无所谓的 —— 但距离不能不是 0。"""
        pts = [(1.0, 2.0, 3.0)] * 50 + [(9.0, 9.0, 9.0)]
        root = bvh.build(pts)
        d, _ = bvh.nearest(pts, root, (1.0, 2.0, 3.0))
        self.assertEqual(d, 0.0)


class TestScaling(unittest.TestCase):
    def test_visited_nodes_grow_slowly_with_n(self):
        """BVH 的意义是访问节点数近似对数增长，而不是随 n 线性增长。
        这一条是"它真的在剪枝"的证据：规模扩大 10 倍，访问数不该扩大 10 倍。"""
        small = _pts(2000, seed=3)
        big = _pts(20000, seed=3)
        rs, rb = bvh.build(small), bvh.build(big)
        q = (50.0, 50.0, 5.0)
        _, _, ss = bvh.nearest(small, rs, q, with_stats=True)
        _, _, sb = bvh.nearest(big, rb, q, with_stats=True)
        self.assertLess(sb["visited"], 3 * ss["visited"])

    def test_empty_point_set_is_refused(self):
        """空点集必须报错，不能返回一个"距离无穷远"的答案 ——
        无穷远在调用那一侧看起来和"没找到"一模一样，会被当成安全。"""
        with self.assertRaises(ValueError):
            bvh.build([])


if __name__ == "__main__":
    unittest.main()
