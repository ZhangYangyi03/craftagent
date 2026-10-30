# -*- coding: utf-8 -*-
"""并行后端的验收：逐位一致、加速比如实、CUDA 缺席如实。

这一层要防的失败是"报了个好看的加速比"和"假装有 GPU"。

  逐位一致  并行不是近似。装配结果必须与串行**逐位**相同 —— 能做到
            是因为单元矩阵在子进程里算，累加留在父进程且按原顺序做。
            这里断言的是 `==`，不是 1e-15 以内。
  加速比    只要求"报出来了、而且是正的"。**不要求大于 1**：小问题
            并行更慢是真的，测试不能逼着代码撒谎。断言大于 1 会让
            这条测试在 CI 上随机红，然后有人把阈值调到 0.1。
  CUDA      本机没有卡时，`available` 必须是 False，而且必须给出原因。
            如果哪天有卡了，这条会走另一支 —— 那时数字才是真的。
"""
import unittest

from craftagent import fem3d
from craftagent import gpu


class TestStatus(unittest.TestCase):
    def test_cpu_count_is_positive(self):
        self.assertGreaterEqual(gpu.cpu_count(), 1)

    def test_status_names_the_start_method(self):
        st = gpu.backend_status()
        self.assertIn(st["process"]["start_method"], ("spawn", "fork", "forkserver"))

    def test_probe_cannot_kill_the_caller(self):
        """探测必须是隔进程的：这台机器上的 torch 一 import 就把进程 abort
        （OpenBLAS 分配失败），父进程里 __import__ 会让这条测试整个挂住，
        而失败信息里没有一个字提到 GPU。这一条断言的是"探测活着回来了"，
        并且如果某个包把子进程弄死了，结果里要指名道姓。"""
        c = gpu.cuda_probe()
        self.assertIn("available", c)
        self.assertTrue(c.get("tried"))
        for rec in c["tried"]:
            if rec.get("error") == "import_crashed":
                self.assertTrue(rec.get("crashed_on"),
                                "说是探测崩了，却没说崩在哪个包上")
                self.assertIn(rec["crashed_on"], ("cupy", "torch"))

    def test_cuda_absence_comes_with_a_reason(self):
        c = gpu.cuda_probe()
        if c["available"]:
            self.assertGreaterEqual(len(c["devices"]) if isinstance(c["devices"], list)
                                    else c["devices"], 1)
        else:
            self.assertTrue(c.get("why"))
            self.assertTrue(c.get("tried"))
            # 缺席时不能有 note 之外的任何数字承诺
            self.assertNotIn("speedup", c)


class TestBitwise(unittest.TestCase):
    def test_tet4_assembly_is_bitwise_identical(self):
        r = gpu.compare(h=4.0, order=1, workers=2, repeat=1)
        self.assertTrue(r["bitwise"])
        self.assertEqual(r["max_abs_diff"], 0.0)
        self.assertGreater(r["entries"], 0)

    def test_tet10_assembly_is_bitwise_identical(self):
        r = gpu.compare(h=4.0, order=2, workers=2, repeat=1)
        self.assertTrue(r["bitwise"])
        self.assertEqual(r["max_abs_diff"], 0.0)

    def test_parallel_matrix_still_solves_the_same_way(self):
        """装配对上了还不够：拿它解一次，位移必须和串行的**逐位**一样。
        PCG 是确定性的，输入逐位相同则输出逐位相同 —— 这条验的是
        "装配确实逐位相同"这个前提在整条链上还成立。"""
        D = fem3d.D_isotropic(210000.0, 0.3)
        mesh = gpu._beam(h=4.0)
        A_s, nodes, tets = gpu.assemble_serial(mesh, D, 1)
        A_p, _, _ = gpu.assemble_parallel(mesh, D, 1, workers=2)
        from craftagent import sparse
        n = 3 * len(nodes)
        fixed = {}
        for k in range(len(nodes)):
            if abs(nodes[k][0]) < 1e-12:
                fixed[3 * k] = fixed[3 * k + 1] = fixed[3 * k + 2] = 0.0
        b = [0.0] * n
        tip = max(range(len(nodes)), key=lambda k: nodes[k][0])
        b[3 * tip + 2] = -1.0
        bs = sparse.apply_dirichlet(A_s, b, fixed)
        bp = sparse.apply_dirichlet(A_p, b, fixed)
        self.assertEqual(bs, bp)
        xs = sparse.pcg(A_s, bs, tol=1e-12)
        xp = sparse.pcg(A_p, bp, tol=1e-12)
        self.assertTrue(xs["converged"])
        self.assertEqual(xs["x"], xp["x"])
        self.assertGreater(max(abs(v) for v in xs["x"]), 0.0)


class TestSpeedup(unittest.TestCase):
    def test_speedup_is_reported_and_finite(self):
        r = gpu.compare(h=4.0, order=1, workers=2, repeat=1)
        self.assertGreater(r["serial_s"], 0.0)
        self.assertGreater(r["speedup"], 0.0)
        self.assertTrue(r["speedup"] < 1e6)
        self.assertGreater(r["speedup_reuse"], 0.0)

    def test_speedup_may_be_below_one_and_that_is_fine(self):
        """小问题并行更慢 —— 断言的是"照报"，不是"必须大于 1"。"""
        r = gpu.compare(h=4.0, order=1, workers=2, repeat=1)
        self.assertIsInstance(r["speedup"], float)
        if r["speedup"] < 1.0:
            # 慢的原因必须能被解释：建池开销比计算本身大
            self.assertGreater(r["pool_setup_s"], 0.0)


class TestCheck(unittest.TestCase):
    def test_check_passes_without_claiming_a_gpu(self):
        r = gpu.check(full=False)
        self.assertTrue(r["ok"])
        if not r["status"]["cuda"]["available"]:
            self.assertTrue(r["status"]["cuda"]["why"])


if __name__ == "__main__":
    unittest.main()
