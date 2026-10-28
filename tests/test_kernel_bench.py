# -*- coding: utf-8 -*-
"""这套评分本身也要有验收 —— 否则"谁写对了内核"就变成了我的一个说法。

这一组测试不调用任何模型。它验的是**判分链路**：

  * 一份已知正确的实现（craftagent 自己的 fem2d 包成 C 的等价物，用纯 Python
    重算同一组比值）必须被判 sane 且 converges；
  * 已知错法（形函数导数少乘 1/4、整体缩放 4 倍、位移几乎为 0）必须被判错；
  * 答案没给模型，所以"把常数写死"这条路必须被堵住：常数写死会同时丢掉
    10x2 那一档的 0.7095 —— 一条测试专门钉这个。

第三条是这个题库真正的防线。没有它，一个只会背 0.9779 的实现能过所有检查。
"""
import os
import unittest

from craftagent import fem2d

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "out", "bench_llm")
TOL = dict(sane=(0.80, 1.05))


def beam_reference(L, H, total_y, E=210000.0, nu=0.3):
    """和 bench_llm/run_bench.py 里那个必须是同一个口径。两边一旦分叉，
    评分就变成了两套标准。"""
    Eps = E / (1.0 - nu * nu)
    I = H ** 3 / 12.0
    return total_y * L ** 3 / (3.0 * Eps * I)


def reference_ratios():
    """用 fem2d 真解一遍，得到"正确实现"在这三档网格上的比值。
    fem2d 自己已经被分片试验、零空间维数、梁理论三道门验过。"""
    out = []
    for nx, ny in ((10, 2), (20, 4), (40, 8)):
        c = fem2d.cantilever_tip(nx, ny, L=100.0, H=10.0)
        out.append(c["ratio"])
    return out


def judge(ratios):
    """评分口径。参数是一个实现给出的三个比值（10x2, 20x4, 40x8）。"""
    if len(ratios) != 3:
        return {"sane": False, "converges": None, "why": "需要三档网格"}
    sane = 0.80 <= ratios[-1] <= 1.05
    converges = all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:]))
    return {"sane": sane, "converges": converges,
            "exit_reason": "ok" if sane else "insane_number"}


class TestTheJudgeItself(unittest.TestCase):
    def test_the_reference_implementation_passes(self):
        r = reference_ratios()
        v = judge(r)
        self.assertTrue(v["sane"], r)
        self.assertTrue(v["converges"], r)
        # 三档都贴着梁理论收敛过去，这才是"对了"的样子
        self.assertAlmostEqual(r[0], 0.7095, places=3)
        self.assertAlmostEqual(r[-1], 0.9779, places=3)

    def test_a_four_times_too_stiff_kernel_is_rejected(self):
        """形函数导数少乘 1/4：B 大 2 倍、K 大 4 倍、位移小 4 倍。
        这是 GLM-5.3-Flash 实际交上来的那一版。它会编译、会跑、会收敛 ——
        比值单调靠近 0.25 —— 所以只靠"收敛"这一条会放过它。"""
        bad = [r / 4.0 for r in reference_ratios()]
        v = judge(bad)
        self.assertFalse(v["sane"], bad)
        self.assertTrue(v["converges"], "此错法单调收敛，所以收敛判据单独不够")

    def test_a_sixteen_times_too_stiff_kernel_is_rejected(self):
        """dN 大 4 倍的那一版（GLM-5.3-Flash r3，比值 0.0611）。"""
        v = judge([r / 16.0 for r in reference_ratios()])
        self.assertFalse(v["sane"])

    def test_a_collapsed_solution_is_rejected(self):
        """GLM-5.3 r3 交上来的 1.3e-08：系统在数值上退化了。
        它一样打印出一行 tip_uy=，一样有小数位。"""
        v = judge([1.3e-8, 2.0e-8, 3.0e-8])
        self.assertFalse(v["sane"])
        self.assertTrue(v["converges"], "位移趋零也是单调的")

    def test_hardcoding_the_answer_is_caught_by_the_coarse_mesh(self):
        """题库真正的防线：答案没给模型，所以它没法把 0.9779 写死。
        一个只会返回 0.9779 的实现，在 10x2 那一档会露出 0.7095 的缺口 ——
        除非它连三档都硬编码，而三档的 CLI 参数由调用方随机给。"""
        fake = [0.9779, 0.9779, 0.9779]
        v = judge(fake)
        # 它 sane（细网格蒙对了），但 coarse 那一档差了 27%
        self.assertTrue(v["sane"])
        self.assertGreater(abs(fake[0] - reference_ratios()[0]) / 0.7095, 0.2)

    def test_the_benchmark_and_this_test_share_one_reference(self):
        """两处口径分叉，评分就变成两套标准。这一条钉住它们相等。"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_bench", os.path.join(BENCH, "run_bench.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            self.skipTest("bench_llm 不在源码树里（它被 .gitignore 排除时）")
        self.assertAlmostEqual(mod.beam_reference(100, 10, 1000),
                               beam_reference(100, 10, 1000), places=9)
        # 而它必须与 fem2d 的梁理论同一个数
        self.assertAlmostEqual(beam_reference(100, 10, 1000), 17.3333333, places=6)


if __name__ == "__main__":
    unittest.main()
