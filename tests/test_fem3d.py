# -*- coding: utf-8 -*-
"""fem3d 与 sparse 的验收：四道门，每道都对着不是它们自己的东西。

这个文件最怕的失败是"三维位移场看起来光滑"。有限元的错法在三维里比二维
更隐蔽：自由度多，任何一个错都会被平均掉，最后给出一个像样的挠度曲线。

所以：
  分片试验   对解析线性场（三线性单元能精确表示它）
  零空间     对维数 6（三维刚体运动的数学事实）
  悬臂梁     对梁理论，且必须**随细化收敛**（hex8 会锁死，从下面逼近）
  稀疏/稠密  同一个方程组两种解法，解必须一致（抓装配时行列写反）

最后一条单独存在，是因为行列写反的矩阵仍然对称正定，共轭梯度照样收敛 ——
它收敛到的是另一个问题的解，而那个解也光滑。
"""
import math
import unittest

from craftagent import fem3d as F
from craftagent import sparse as S


class TestSparse(unittest.TestCase):
    def test_pcg_solves_the_same_system_as_dense_elimination(self):
        r = F.sparse_vs_dense(4, 2, 2)
        self.assertTrue(r["sparse_converged"], r)
        self.assertLess(r["rel_diff"], 1e-9, r)

    def test_pcg_reports_non_convergence_instead_of_a_pretty_vector(self):
        """迭代次数用光时必须说没收敛，不能把最后一步当答案返回。

        悄悄返回的样子是"位移场光滑、量级合理"，而它离真解可能差几倍。
        """
        A = S.zeros(3)
        for i in range(3):
            S.add(A, i, i, 1.0)
            S.add(A, i, (i + 1) % 3, -1.0)      # 每行和为零 -> 奇异
        r = S.pcg(A, [1.0, 0.0, 0.0], tol=1e-14, maxit=20)
        self.assertFalse(r["converged"], r)

    def test_dirichlet_moves_the_column_terms_to_the_rhs(self):
        """只清行不清列，非约束自由度会被约束位移的反力污染。

        结果仍然满足约束、仍然光滑，所以从位移场上看不出来 —— 只能靠
        一个手算得出的 2x2 例子抓。
        """
        A = S.zeros(2)
        S.add(A, 0, 0, 2.0)
        S.add(A, 0, 1, -1.0)
        S.add(A, 1, 0, -1.0)
        S.add(A, 1, 1, 2.0)
        b = S.apply_dirichlet(A, [0.0, 0.0], {0: 3.0})
        self.assertEqual(b[0], 3.0)
        self.assertAlmostEqual(b[1], 3.0, places=12)   # 0 - (-1)*3
        self.assertEqual(A[0], {0: 1.0})
        self.assertNotIn(0, A[1])


class TestElement(unittest.TestCase):
    def test_patch_test_reproduces_a_linear_field(self):
        """常应变场必须被精确复现。这一条不过，B 矩阵或雅可比就是错的。"""
        r = F.patch_test(3, 3, 3)
        self.assertTrue(r["ok"], r)
        self.assertLess(r["rel_err"], 1e-12, r)

    def test_patch_test_holds_on_a_non_cubic_box(self):
        """非立方体的雅可比不是对角阵 —— 只在正方体上对说明雅可比写死了。"""
        r = F.patch_test(4, 2, 3)
        self.assertTrue(r["ok"], r)

    def test_hex8_has_exactly_six_rigid_body_modes(self):
        """三平移 + 三转动。多一个是沙漏，少一个是被锁住，两个方向都错。"""
        r = F.rigid_body_nullity()
        self.assertEqual(r["nullity"], 6, r)

    def test_a_flipped_element_is_refused(self):
        """节点顺序反了必须报错。负雅可比让刚度矩阵负定，解出来是个能看的数。"""
        coords = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
                  (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
        with self.assertRaises(ValueError):
            F.element_stiffness(coords, F.D_isotropic(1000.0, 0.3))

    def test_shear_modulus_is_not_divided_by_two(self):
        """G = E/(2(1+nu))。写成 E/(1+nu) 的表现是拉伸对、弯曲软一倍。"""
        D = F.D_isotropic(210000.0, 0.3)
        G = 210000.0 / (2 * 1.3)
        self.assertAlmostEqual(D[3][3], G, places=9)
        self.assertAlmostEqual(D[4][4], G, places=9)
        self.assertAlmostEqual(D[5][5], G, places=9)


class TestCantilever(unittest.TestCase):
    def test_tip_deflection_converges_to_beam_theory(self):
        """hex8 在纯弯下锁死，比值从下面逼近 1。断言的是"单调 + 逼近"，
        不是"等于 1" —— 后者会逼着人去调参数把单元做错来让测试变绿。"""
        ratios = [F.cantilever(nx=nx)["ratio"] for nx in (8, 16, 24)]
        self.assertTrue(all(b > a for a, b in zip(ratios, ratios[1:])), ratios)
        self.assertGreater(ratios[-1], 0.90, ratios)
        self.assertLess(ratios[-1], 1.02, ratios)

    def test_locking_depends_on_slenderness(self):
        """同一套代码，短梁比细长梁收敛得好。这不是巧合，是锁定项 ~ (h/L)^2。"""
        slender = F.cantilever(nx=12, ny=4, nz=4, L=100.0, b=12.0, h=12.0)["ratio"]
        stubby = F.cantilever(nx=12, ny=4, nz=4, L=60.0, b=12.0, h=12.0)["ratio"]
        self.assertGreater(stubby, slender + 0.02, (slender, stubby))

    def test_solver_reports_convergence_with_the_residual(self):
        r = F.cantilever(nx=8)
        self.assertTrue(r["converged"], r)
        self.assertLess(r["rel_res"], 1e-10, r)


if __name__ == "__main__":
    unittest.main()
