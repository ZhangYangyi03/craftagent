# -*- coding: utf-8 -*-
"""fem2d 的验收：三道公认的门，加一条把一维结论搬到二维。

这个文件最怕的失败是"算出来一个像样的数"。有限元的错法几乎都长得很像
对的结果：沙漏模态会给出光滑位移场、错的雅可比会给出线性场、装配错一个
自由度会让挠度差 30% 而不报错。所以每一条都对着**不是它自己**的参照：
  分片试验对解析线性场、
  零空间对维数 3（数学事实，不是算出来的）、
  悬臂梁对梁理论（且在细化网格下必须收敛过去）。
"""
import math
import unittest

from craftagent import fem2d as F


class TestElement(unittest.TestCase):
    def test_patch_test_reproduces_a_linear_field(self):
        """常应变场必须被精确复现。这一条不过，单元公式就是错的。"""
        r = F.patch_test(4, 4)
        self.assertTrue(r["ok"], r)
        self.assertLess(r["rel_err"], 1e-12)

    def test_patch_test_holds_on_a_non_square_mesh(self):
        """矩形网格上的线性场也是线性 —— 通不过说明雅可比只在正方形上对。"""
        r = F.patch_test(5, 3, lx=3.0, ly=1.0)
        self.assertLess(r["rel_err"], 1e-12, r)

    def test_q4_has_exactly_three_rigid_body_modes(self):
        """两个平移加一个转动。多一个是沙漏（能量漏掉但位移光滑），
        少一个是被锁住。两个方向都是错的，所以用等号。"""
        r = F.rigid_body_nullity()
        self.assertEqual(r["nullity"], 3, r)

    def test_jacobian_refuses_a_flipped_element(self):
        """节点顺序反了必须报错，不能算出一个负体积的单元 ——
        负雅可比会让刚度矩阵变成负定，解出来是个能看的数，但符号是反的。"""
        D = F.D_plane_strain(1000.0, 0.3)
        bad = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]   # 顺时针
        with self.assertRaises(ValueError):
            F.element_stiffness(bad, D)


class TestAssembly(unittest.TestCase):
    def test_cantilever_converges_to_beam_theory(self):
        """细化时必须单调靠近梁理论。这一条验的是装配与求解，
        因为单元已经被分片试验验过了。"""
        ratios = [F.cantilever_tip(nx, ny)["ratio"]
                  for nx, ny in ((10, 2), (20, 4), (40, 8))]
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], ratios[2])
        self.assertGreater(ratios[-1], 0.95)

    def test_singular_system_is_refused_with_zero_load(self):
        """约束不够时求解必须报错。

        无载荷这一版是最阴的一版：我第一版阈值写成绝对 1e-300，它解出了一串
        0.0 而没有报错 —— 而零位移场正是无载荷时的正确答案，所以没有任何
        东西会因此变红。阈值改成相对量之后这条才立得住。"""
        nodes, elems = F.rect_mesh(2, 2)
        D = F.D_plane_strain(1000.0, 0.3)
        K = F.assemble(nodes, elems, D)
        with self.assertRaises(ZeroDivisionError):
            F.solve(K, [0.0] * (2 * len(nodes)))

    def test_singular_system_is_refused_with_load(self):
        """带载荷时奇异系统会解出一串 1e16 量级的数 —— 那比 0 更像答案。
        它必须同样报错。"""
        nodes, elems = F.rect_mesh(2, 2)
        D = F.D_plane_strain(1000.0, 0.3)
        K = F.assemble(nodes, elems, D)
        f = [1.0] * (2 * len(nodes))
        with self.assertRaises(ZeroDivisionError):
            F.solve(K, f)

    def test_dirichlet_moves_the_load_to_the_right_side(self):
        """给定位移的移项要对：约束位移不为零时，右端项要减去 K*u_fixed。
        错了会得到一个"看起来合理但差一截"的解，而分片试验恰好能抓住它 ——
        所以这里直接复用分片试验做交叉验证。"""
        # patch_test 里约束位移全部非零，能通过就说明移项对
        r = F.patch_test(3, 3)
        self.assertLess(r["rel_err"], 1e-12)


class TestPenaltyInTwoDimensions(unittest.TestCase):
    def test_the_u_shape_survives_in_2d(self):
        """一维量到的 U 形是不是一维的产物？这一条是那个问题的答案。

        二维平面应变、真实装配、真实求解，谷底仍在 Kp/k_chain ~ 1e8，
        两侧各按 k/Kp 与 eps*Kp/k 上升。所以那条结论不是一维模型的假象。"""
        r = F.penalty_contact_2d(nx=20, ny=1,
                                 ratios=[1e2, 1e4, 1e6, 1e8, 1e10, 1e12])
        errs = [row["rel_err_vs_stablest"] for row in r["rows"]]
        i = errs.index(min(errs))
        self.assertGreater(i, 1)
        self.assertLess(i, len(errs) - 2)
        self.assertLess(errs[i], 1e-6)
        self.assertGreater(errs[-1], 1e-5)          # 右边确实掉头上升
        self.assertGreater(errs[0], 1e-3)           # 左边确实太软
        # 谷底位置在 1e6..1e10 之间
        self.assertTrue(1e6 <= r["reference_ratio"] <= 1e10, r["reference_ratio"])

    def test_beyond_the_precision_edge_it_refuses_instead_of_lying(self):
        """Kp/k=1e15 已经越过双精度的尽头（条件数 ~1e17）。这时必须报奇异，
        而不是返回一个看起来很平静的数。讲得出来比给一个数好。"""
        with self.assertRaises(ZeroDivisionError):
            F.penalty_contact_2d(nx=10, ny=1, ratios=[1e15])

    def test_the_result_says_what_it_does_not_claim(self):
        """这个函数没有精确解可比（二维精确解太贵），所以它必须自己说出来。
        一个不写边界的"验证通过"比没有验证更坏。"""
        r = F.penalty_contact_2d(nx=10, ny=1,
                                 ratios=[1e4, 1e6, 1e8])
        self.assertIn("不是无穷", r["caveat"])


if __name__ == "__main__":
    unittest.main()
