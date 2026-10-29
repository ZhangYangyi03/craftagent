# -*- coding: utf-8 -*-
"""voxel / tetmesh / 四面体单元 的验收。

这个文件要防的失败是"网格看起来像那么回事"。体素化 + 回贴 + Kuhn 拆分
这条链上，每一步都能产出一个视觉上完全正常的四面体网格，而它在求解器里
是错的。所以每一条判据都对着**网格之外**的东西：

  体积        对解析体积（球 4/3 pi r^3），且必须按 O(h^2) 收敛
  欧拉示性数  V-E+F-T：实心球是 1，带一个通孔的板是 0
  流形         每个内部面恰好被两个四面体共用（Kuhn 拆分是否一致的判据）
  分片试验     对解析线性场
  刚体转动     对"应变必须为零"这个物理事实 —— 这一条专门抓 Jacobian 转置，
              分片试验抓不到（对称应变转置后不变）
  悬臂梁       对梁理论，且四面体与六面体两条路径必须互相吻合
"""
import math
import unittest

from craftagent import fem3d as F
from craftagent import tetmesh as T
from craftagent import voxel as V


class TestIntervals(unittest.TestCase):
    def test_boolean_ops_on_intervals(self):
        a = V.Box(0, 2, 0, 2, 0, 1)
        b = V.Box(0, 2, 0, 2, 0.5, 2)
        self.assertEqual(V.Union(a, b).intervals(1, 1), [(0, 2)])
        self.assertEqual(V.Intersect(a, b).intervals(1, 1), [(0.5, 1)])
        self.assertEqual(V.Diff(a, b).intervals(1, 1), [(0, 0.5)])

    def test_boolean_outside_the_x_y_footprint_is_empty(self):
        a = V.Box(0, 2, 0, 2, 0, 1)
        b = V.Box(10, 12, 10, 12, 0, 1)
        self.assertEqual(V.Union(a, b).intervals(5, 5), [])
        self.assertEqual(V.Intersect(a, b).intervals(5, 5), [])

    def test_volume_ref_matches_closed_forms(self):
        """中点求积是"没有闭式体积的件"的参照，它自己得先被验。"""
        for sh in (V.Box(0, 10, 0, 8, 0, 6), V.Sphere(0, 0, 0, 5),
                   V.CylinderZ(0, 0, 4, 0, 7),
                   V.PrismZ([(0, 0), (10, 0), (10, 6), (5, 9), (0, 6)], 0, 4)):
            ref = V.volume_ref(sh, n=60)
            self.assertLess(abs(ref - sh.volume()) / sh.volume(), 5e-3, sh)


class TestVoxelize(unittest.TestCase):
    def test_thin_wall_is_reported_as_lost_not_silently_dropped(self):
        """比 h 还薄的壁会整条落空。必须数出来，不能让它悄悄消失。"""
        plate = V.Box(0, 20, 0, 20, 0, 0.4)
        vm = V.voxelize(plate, 2.0)
        self.assertEqual(vm["cells"], 0)
        self.assertGreater(vm["lost_volume"], 0.9 * plate.volume())

    def test_thick_wall_is_kept(self):
        plate = V.Box(0, 20, 0, 20, 0, 6)
        vm = V.voxelize(plate, 2.0)
        self.assertGreater(vm["cells"], 0)
        self.assertEqual(vm["lost_volume"], 0.0)


class TestTetMeshTopology(unittest.TestCase):
    def test_box_volume_is_exact_and_euler_is_one(self):
        r = T.check(V.Box(0, 10, 0, 10, 0, 10), h=2.5)
        self.assertTrue(r["ok"], r)
        self.assertAlmostEqual(r["volume"], 1000.0, places=9)
        self.assertEqual(r["euler"], 1)
        self.assertEqual(r["negative"], 0)

    def test_every_interior_face_is_shared_by_exactly_two_tets(self):
        """Kuhn 拆分的一致性判据。

        相邻两格用同一条对角线方向，公共面上的三角化才一样。不一样的话
        内部面会被用 1 次或 3 次 —— 网格在缝上是断的，而 PCG 照样收敛。
        """
        m = T.build(V.Box(0, 10, 0, 10, 0, 10), 2.5)
        topo = T.topology(m)
        self.assertEqual(topo["faces_nonmanifold"], 0)
        self.assertEqual(topo["faces_interior"] * 2 + topo["faces_boundary"],
                         topo["F"] * 1 + topo["faces_interior"])
        # 边界面上每条边只属于一个面
        self.assertEqual(topo["faces_boundary"], 6 * 4 * 4 * 2)

    def test_sphere_volume_converges_at_second_order(self):
        s = V.Sphere(0, 0, 0, 5)
        errs = [T.check(s, h=h)["volume_err"] for h in (2.5, 1.25, 0.625)]
        for e in errs:
            self.assertLess(e, 0.12, errs)
        # 误差必须单调下降，且相邻两级的比值接近 4（O(h^2)）
        self.assertGreater(errs[0], errs[1])
        self.assertGreater(errs[1], errs[2])
        ratio = errs[1] / errs[2]
        self.assertTrue(2.0 < ratio < 8.0, (errs, ratio))

    def test_projection_keeps_volume_close_to_exact(self):
        """回贴的意义：台阶面换成真实表面，体积不能因此塌掉。

        第一版只把"在外侧"的节点往里推，内侧节点不动，表面整体内缩，
        球体积被削掉 29.5%。这一条守的就是那个 bug。
        """
        s = V.Sphere(0, 0, 0, 5)
        a = T.build(s, 0.8333, project=True)
        b = T.build(s, 0.8333, project=False)
        ea = abs(T.volume(a) - s.volume()) / s.volume()
        eb = abs(T.volume(b) - s.volume()) / s.volume()
        self.assertLess(ea, 0.02, (ea, eb))
        self.assertLess(ea, eb, (ea, eb))

    def test_plate_with_hole_has_genus_one(self):
        """带一个通孔：欧拉示性数从 1 掉到 0。掉不下去说明孔被填了。"""
        plate = V.Diff(V.Box(0, 40, 0, 20, 0, 6), V.CylinderZ(20, 10, 5, 0, 6))
        for h in (2.0, 1.0):
            r = T.check(plate, h=h)
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["euler"], 0, r)
            self.assertEqual(r["negative"], 0, r)
            ref = V.volume_ref(plate, n=80)
            self.assertLess(abs(r["volume"] - ref) / ref, 0.01, (h, r["volume"], ref))

    def test_compact_drops_nodes_no_element_uses(self):
        m = T.build(V.Box(0, 1, 0, 1, 0, 1), 0.5, project=False)
        used = set()
        for t in m["tets"]:
            used.update(t)
        self.assertEqual(len(used), len(m["nodes"]))
        self.assertEqual(max(used), len(m["nodes"]) - 1)
        self.assertGreater(m["dropped_nodes"], 0)

    def test_tet10_midpoints_match_the_shape_function_order(self):
        """中点编号与 fem3d.tet10_stiffness 的形函数顺序必须一致。

        不一致时刚度矩阵仍然对称正定，解出来仍然是个位移场 —— 只是它是
        另一个问题的解。
        """
        m10 = T.tet10(T.build(V.Box(0, 1, 0, 1, 0, 1), 0.5, project=False))
        for t in m10["tets"]:
            for a, b, mid in ((0, 1, 4), (1, 2, 5), (2, 0, 6),
                              (0, 3, 7), (1, 3, 8), (2, 3, 9)):
                want = tuple(0.5 * (m10["nodes"][t[a]][i] + m10["nodes"][t[b]][i])
                             for i in range(3))
                got = m10["nodes"][t[mid]]
                self.assertLess(max(abs(want[i] - got[i]) for i in range(3)), 1e-12)


class TestTetElements(unittest.TestCase):
    def test_patch_test_is_exact_for_both_orders(self):
        for order in (1, 2):
            r = F.patch_test_tet(h=0.5, order=order)
            self.assertTrue(r["ok"], r)
            self.assertLess(r["max_err"], 1e-12, r)

    def test_rigid_rotation_produces_no_strain(self):
        """这一条专门抓 Jacobian 转置。

        分片试验用对称应变，转置后对称部分不变，抓不到；刚体转动的位移
        梯度反对称，转置等于取负，一混就露。
        """
        for order in (1, 2):
            r = F.rotation_test_tet(order)
            self.assertTrue(r["ok"], r)

    def test_nullity_is_six_for_both_orders(self):
        for order in (1, 2):
            r = F.rigid_body_nullity_tet(order)
            self.assertEqual(r["nullity"], 6, r)

    def test_tet10_matches_beam_theory_and_hex8(self):
        r = F.tet_vs_hex(h=4.0)
        self.assertTrue(r["converged"], r)
        self.assertLess(abs(r["tet10"] - 1.0), 0.15, r)
        self.assertLess(r["hex_vs_tet10"], 0.20, r)

    def test_tet4_locks_and_that_is_reported_not_hidden(self):
        """常应变单元在纯弯下锁死。比值明显小于 1，且比 TET10 更差。

        写成"必须接近 1"会逼着人去调参数，那是把单元做错让测试变绿。
        """
        r = F.tet_vs_hex(h=4.0)
        self.assertLess(r["tet4"], 0.75, r)
        self.assertLess(r["tet4"], r["tet10"], r)

    def test_tet10_improves_under_refinement(self):
        coarse = F.cantilever_tet(h=5.0, order=2)["ratio"]
        fine = F.cantilever_tet(h=2.5, order=2)["ratio"]
        self.assertLess(abs(fine - 1.0), abs(coarse - 1.0) + 0.05, (coarse, fine))


class TestVoxelBlindSpot(unittest.TestCase):
    """`lost_volume` 只看得见沿 z 方向薄的壁。这一条把这个盲区钉住。

    壁薄在 x 或 y 上时整列采不到，`iv` 为空，什么都不累加 —— 于是
    `lost_volume` 是 0，而材料确实丢了。要判断网格完不完整，看的是
    `volume_error`（净误差，可正可负）。
    """

    def _thin_wall(self):
        return V.Union(V.Box(0, 40, 0, 20, 0, 6),
                       V.Box(0, 40, 9.6, 10.4, 0, 12))

    def test_lost_volume_is_blind_to_a_wall_thin_in_y(self):
        sh = self._thin_wall()
        for h in (2.0, 1.0):
            m = V.voxelize(sh, h)
            self.assertEqual(m["lost_volume"], 0.0, h)
            e = V.volume_error(sh, m, n=60)
            self.assertLess(e["rel"], -0.02, (h, e))

    def test_net_error_can_be_positive_when_a_centre_lands_inside(self):
        """h=0.5 时格心落进壁里，整格算成实心 —— 净误差变正。
        所以这个数必须带符号，取绝对值会把两种错混成一种。"""
        sh = self._thin_wall()
        m = V.voxelize(sh, 0.5)
        e = V.volume_error(sh, m, n=60)
        self.assertGreater(e["rel"], 0.0)

    def test_lost_volume_does_see_a_wall_thin_in_z(self):
        """反过来的对照：z 方向薄的壁能被数到，说明 lost_volume 本身没错，
        只是覆盖面窄。"""
        sh = V.Box(0, 10, 0, 10, 0, 0.4)
        m = V.voxelize(sh, 2.0)
        self.assertGreater(m["lost_volume"], 0.0)


if __name__ == "__main__":
    unittest.main()
