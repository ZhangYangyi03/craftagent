# -*- coding: utf-8 -*-
"""supply 的验收：路由表本身要能被审。

这里测的不是数值，是纪律。这个仓里最容易滑坡的地方是：一个说不清来源的数
被写进表里，然后因为"看起来专业"而留下。所以这里把三条规矩测出来。
"""
import unittest

from craftagent import supply


class TestSupplyRoute(unittest.TestCase):
    def test_every_item_declares_a_route(self):
        for r in supply.route_table():
            self.assertIn(r["route"], ("recompute", "reference", "measure", "-"),
                          r["item"])

    def test_every_item_says_what_backs_it(self):
        """没有"凭什么"的条目不许存在 —— 这和 entity-graph 拒绝无源边是同一条规矩。"""
        for r in supply.route_table():
            self.assertTrue(r["how"].strip(), r["item"])
            self.assertTrue(r["output"].strip(), r["item"])

    def test_recompute_items_name_an_actual_expression_or_standard(self):
        """标成 recompute 的，必须给出式子或标准号；否则它不是可重算，是自称可重算。"""
        for r in supply.by_route("recompute"):
            h = r["how"]
            self.assertTrue(any(k in h for k in ("=", "ISO", "Altintas")),
                            (r["item"], h))

    def test_the_two_hardest_items_are_not_claimed_as_recomputable(self):
        """fn/k/zeta 和 Taylor 常数必须落在 measure 档。这两样被算出来就是编的。"""
        measure = " ".join(r["item"] for r in supply.by_route("measure"))
        self.assertIn("fn / k / zeta", measure)
        self.assertIn("Taylor", measure)

    def test_historical_simulation_is_routed_to_measure_not_copy(self):
        """历史仿真结果不许以"抄来"的形式进库：它是别人求解器的输出。"""
        r = [x for x in supply.route_table() if "历史仿真结果" in x["item"]]
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["route"], "measure")

    def test_measure_plan_refuses_to_fill_in_a_default(self):
        """第三档每一条都要有一句"拒绝"：不测就填默认值，是这一档最常见的失败。"""
        for k, v in supply.MEASURE_PLAN.items():
            self.assertTrue(v["refuse"].strip(), k)
            self.assertGreaterEqual(len(v["points"]), 1, k)


if __name__ == "__main__":
    unittest.main()
