# -*- coding: utf-8 -*-
"""OCC 布尔的验收：有 OCP 就对解析体积，没有就断言"如实说没有"。

这台机器没有 OCP，所以本地跑的是后半部分。CI 的 cad job 装了
cadquery-ocp，那里跑前半部分 —— 两边都有断言，没有一边是空的。

判据在 OCC 之外：两个 10 立方体偏移 5，并 1875 / 交 125 / 差 875。
结果还要过 BRepCheck_Analyzer 并且至少剩一个实体 —— 空壳是 B-rep 布尔
最危险的失败方式，下游会把它读成"没有材料"或"没有干涉"。
"""
import unittest

from craftagent import booleans


_HAVE_OCC = booleans.status()["available"]


class TestAbsenceIsReported(unittest.TestCase):
    def test_status_explains_itself_either_way(self):
        st = booleans.status()
        if st["available"]:
            self.assertTrue(st.get("python"))
        else:
            self.assertTrue(st.get("why"))
            self.assertTrue(st.get("how_to_fix"))
            self.assertTrue(st.get("tried"))

    def test_boolean_does_not_return_an_empty_result_when_ocp_is_missing(self):
        """缺 OCP 时必须**明确失败**，不能返回体积 0 的结果 ——
        下游会把 0 读成"没有干涉"，那是安全方向上的假阴性。"""
        if _HAVE_OCC:
            self.skipTest("本机有 OCP，这条测的是缺席路径")
        r = booleans.boolean("cut", "a.step", "b.step")
        self.assertFalse(r["ok"])
        self.assertEqual(r["why"], "no_opencascade")
        self.assertFalse(r["verified_here"])

    def test_bad_op_is_rejected_before_anything_else(self):
        with self.assertRaises(ValueError):
            booleans.boolean("subtract", "a.step", "b.step")


@unittest.skipUnless(_HAVE_OCC, "需要装 cadquery-ocp（CI 的 cad job 会装）")
class TestWithOCC(unittest.TestCase):
    def test_three_ops_against_closed_form(self):
        r = booleans.selftest()
        self.assertTrue(r["verified_here"], r)
        for name in ("fuse", "cut", "common"):
            row = r["ops"][name]
            self.assertTrue(row["ok"], (name, row))
            self.assertLess(row["rel"], 1e-9, (name, row))
            self.assertTrue(row["valid"], (name, row))
            self.assertGreaterEqual(row["solids"], 1, (name, row))


if __name__ == "__main__":
    unittest.main()
