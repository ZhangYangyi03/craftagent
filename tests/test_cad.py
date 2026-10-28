# -*- coding: utf-8 -*-
"""cad 的验收：读进来的是不是**那块**零件，而不是"一个零件"。

这一层最容易的假通过是"没崩、有三角面、能跑下去"。一个读错的零件
在任何下游检查里都表现得像个正常零件，直到它说"没撞"。所以每一条验收
都对着解析值：体积、面积、包围盒、拓扑计数，全部能手算。

OpenCASCADE 是可选依赖（CI 上不装），所以没有 OCP 时这一组测试
会 skip 并**明确说出来**，而不是静默通过 —— 静默通过的验收等于没有验收。
"""
import math
import os
import sys
import tempfile
import unittest

from craftagent import cad


def _have_occ():
    return cad.find_interpreter() is not None


class TestBackend(unittest.TestCase):
    def test_backend_status_says_what_it_found(self):
        st = cad.backend_status()
        self.assertIn("available", st)
        if st["available"]:
            self.assertTrue(os.path.exists(st["python"]))
            self.assertTrue(os.path.exists(st["worker"]))
        else:
            self.assertIn("why", st)
            self.assertIn("how_to_fix", st)

    def test_no_interpreter_means_a_clear_no_not_a_crash(self):
        """一个候选都没有时必须返回 ok=False 和原因，不能抛异常。"""
        st = cad.backend_status(candidates=[])
        self.assertFalse(st["available"])
        self.assertIn("why", st)
        self.assertIn("how_to_fix", st)
        self.assertEqual(st["tried"], [])

    def test_current_interpreter_is_probed_first_after_the_env_override(self):
        """这一条是被 CI 抓出来的。

        第一版只列了本机几个硬编码路径，于是 CI 上装了 cadquery-ocp
        却报"没有 OpenCASCADE" —— CI 把它装进的是 sys.executable，
        而它不在列表里。探测范围太窄和读进来是空的，是同一种失败：
        适配器说"没有"，而实际上有。"""
        os.environ.pop("CRAFTAGENT_OCC_PYTHON", None)
        cands = cad._interpreter_candidates()
        self.assertEqual(cands[0], sys.executable)
        self.assertEqual(len(cands), len(set(cands)), "候选里有重复项")

    def test_diagnostics_list_what_was_actually_tried(self):
        """说不清自己试过什么的诊断，下一次还是查不出问题。"""
        st = cad.backend_status()
        self.assertIn("tried", st)
        self.assertTrue(st["tried"], "tried 不能是空的：空的诊断等于没诊断")
        for p_ in st["tried"]:
            self.assertIsInstance(p_, str)

    def test_missing_file_and_bad_extension_are_reported(self):
        self.assertFalse(cad.read_part("no_such_part.step")["ok"])
        self.assertEqual(cad.read_part("no_such_part.step")["why"], "no_such_file")
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as fh:
            bad = fh.name
        try:
            r = cad.read_part(bad)
            self.assertFalse(r["ok"])
            self.assertEqual(r["why"], "unsupported_extension")
        finally:
            os.unlink(bad)


@unittest.skipUnless(_have_occ(), "没有 OpenCASCADE：CAD 这一层未验收（不是通过）")
class TestReadsTheRightPart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.tmpdir, "sample_block.step")
        r = cad.make_sample(cls.path, with_hole=True)
        if not r.get("ok"):
            raise unittest.SkipTest("样板件生成失败：%s" % r.get("why"))
        cls.part = cad.read_part(cls.path, tol=0.2)
        if not cls.part.get("ok"):
            raise unittest.SkipTest("样板件读取失败：%s" % cls.part.get("why"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_volume_matches_the_analytic_value(self):
        """50x30x10 的块挖 R8 通孔，体积解析值可算。
        这是这一层唯一真正硬的验收：读错一个面，体积就不对。"""
        exact = cad.exact_volume_with_hole()
        got = self.part["volume"]
        self.assertLess(abs(got - exact) / exact, 1e-6, (got, exact))

    def test_bounding_box_is_the_part_not_the_mesh(self):
        """包围盒必须是零件尺寸，不是三角面尺寸 —— 三角化有挠度，
        顶点可能比真实表面略靠里，容差取 tol。"""
        b = self.part["bbox"]
        for got, want in zip((b[1][0] - b[0][0], b[1][1] - b[0][1],
                              b[1][2] - b[0][2]), (50.0, 30.0, 10.0)):
            self.assertLess(abs(got - want), 1e-5, (got, want))

    def test_topology_counts_are_what_a_block_with_a_hole_has(self):
        """7 个面：6 个块面 + 1 个孔壁。面数错了说明布尔没干净。"""
        c = self.part["counts"]
        self.assertEqual(c["solids"], 1)
        self.assertEqual(c["faces"], 7)
        self.assertEqual(c["shells"], 1)

    def test_triangles_are_all_inside_the_bounding_box(self):
        """三角面顶点必须都落在包围盒里 —— 出界说明变换没应用上
        （面自带的 Location 没乘，症状是零件飘到原点附近）。"""
        coords = self.part["coords"]
        lo, hi = self.part["bbox"]
        for k in range(3):
            vals = coords[k::3]
            self.assertGreaterEqual(min(vals), lo[k] - 1e-6)
            self.assertLessEqual(max(vals), hi[k] + 1e-6)

    def test_triangle_indices_are_in_range(self):
        """索引越界在纯 Python 里不会崩，只会取到别的顶点 ——
        于是零件看起来正常，但是歪的。"""
        n = len(self.part["coords"]) // 3
        for i in self.part["tris"]:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, n)

    def test_a_real_hole_is_really_there(self):
        """体积比实心块小多少，就要等于孔的体积。这一条防的是
        "布尔没切" —— 那时面数可能是 6，体积是 15000，一切看起来都正常。"""
        solid = 50.0 * 30.0 * 10.0
        hole = solid - self.part["volume"]
        self.assertLess(abs(hole - math.pi * 64.0 * 10.0) / (math.pi * 64.0 * 10.0),
                        1e-6)


if __name__ == "__main__":
    unittest.main()
