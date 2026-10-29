# -*- coding: utf-8 -*-
"""实时闭环的验收：估计器、Lyapunov 参考值、截止期、阶跃、饱和。

四条判据各有各的外部参照，互不替代：

  估计器    后验/先验方差比 = 1/(1+snr) = 1-g_K。Riccati 给理论值，
            蒙特卡洛给实测值。注意 snr = P⁻/R 不是 Kalman 增益 g_K。
  闭环      **精确参考值是离散 Lyapunov 方程的解**，不是仿真。这里没有
            1/(1+g) 那种漂亮公式 —— 拿它去套输出方差，会在传感器脏、
            增益小的时候给出"闭环一定更好"的错误结论。实测那一片区域
            闭环是更差的，所以判据只要求"仿真对得上参考值"，不要求
            "闭环更好"。
  截止期    虚拟时钟下确定性：故意慢的拍必须**正好**被数成 miss，
            不能顺延、不能漏算、不能把实际周期写成标称周期。
  阶跃      实测收敛拍数对理想积分的下界；超调如实报。
  饱和      限幅生效时必须报 saturated，且残差等于闭式解
            amp/(1-p) - limit —— 闭环不能承诺它做不到的事。
"""
import math
import unittest

from craftagent import closedloop as C


class TestRiccati(unittest.TestCase):
    def test_random_walk_steady_state(self):
        """随机游走 (phi=1) 的稳态解满足 g² + qg - q = 0，q = Q/R。"""
        q = 0.25
        r = C.riccati(Q=q, R=1.0, phi=1.0)
        g = r["g_k"]
        self.assertAlmostEqual(g * g + q * g - q, 0.0, places=12)

    def test_posterior_over_prior_is_one_over_one_plus_snr(self):
        for Q, R, phi in ((1.0, 4.0, 0.98), (0.5, 2.0, 1.0), (2.0, 0.5, 0.9)):
            r = C.riccati(Q, R, phi)
            self.assertAlmostEqual(r["P_post"] / r["P_prior"], 1.0 / (1.0 + r["snr"]),
                                   places=14)
            self.assertAlmostEqual(r["ratio"], r["ratio_alt"], places=14)

    def test_zero_measurement_noise(self):
        r = C.riccati(Q=1.0, R=0.0)
        self.assertEqual(r["P_post"], 0.0)
        self.assertEqual(r["g_k"], 1.0)


class TestFilterGate(unittest.TestCase):
    def test_monte_carlo_matches_theory(self):
        r = C.filter_gate(N=200000)
        self.assertTrue(r["ok"])
        self.assertLess(abs(r["measured_ratio"] - r["theory_ratio"]) / r["theory_ratio"],
                        r["tol"])

    def test_prior_and_posterior_variances_are_both_right(self):
        """比例对上了也可能两条方差都错同一个倍数，所以分开比。"""
        r = C.filter_gate(N=200000)
        self.assertLess(abs(r["measured_prior"] / r["theory_prior"] - 1.0), 0.03)
        self.assertLess(abs(r["measured_post"] / r["theory_post"] - 1.0), 0.03)


class TestLyapunov(unittest.TestCase):
    def test_lyapunov_solver_on_a_known_case(self):
        """一阶 AR(1) 的稳态方差是 Q/(1-phi²)，用 Lyapunov 解一遍。"""
        phi, Q = 0.7, 2.0
        S = C._lyap([[phi]], [[Q]])
        self.assertAlmostEqual(S[0][0], Q / (1.0 - phi * phi), places=12)

    def test_lyapunov_solver_on_a_two_state_case(self):
        """x_{k+1} = p x_k + d_k, d 白噪声方差 Q：稳态方差 Q/(1-p²)。"""
        p, Q = 0.6, 1.0
        S = C._lyap([[p, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, Q]])
        self.assertAlmostEqual(S[0][0], Q / (1.0 - p * p), places=12)


class TestLoopGate(unittest.TestCase):
    def test_simulation_matches_the_exact_reference(self):
        r = C.loop_gate(ticks=200000)
        self.assertTrue(r["ok"])
        self.assertLess(r["rel"], r["tol"])

    def test_closed_loop_is_better_here_but_that_is_not_the_gate(self):
        r = C.loop_gate(ticks=200000)
        self.assertLess(r["ratio"], 1.0)


class TestCrossover(unittest.TestCase):
    def test_dirty_sensor_can_make_the_loop_worse(self):
        """这是这一层最重要的一条：闭环不是无条件更好。"""
        c = C.crossover()
        worst = max(b["ratio"] for b in c["by_noise"])
        self.assertGreater(worst, 1.0)

    def test_more_gain_helps_when_the_sensor_is_clean(self):
        c = C.crossover()
        ratios = [b["ratio"] for b in c["by_gain"]]
        self.assertLess(ratios[-1], ratios[0])


class TestDeadline(unittest.TestCase):
    def test_slow_ticks_are_counted_exactly(self):
        r = C.deadline_gate(ticks=40, slow=(7, 8, 9))
        self.assertTrue(r["ok"])
        self.assertEqual(r["missed"], 3)
        self.assertEqual(r["miss_idx"], [7, 8, 9])

    def test_achieved_period_reflects_the_overrun(self):
        """实际平均周期必须大于每拍的计算耗时 —— 不能把标称周期当实际。"""
        r = C.deadline_gate(ticks=40, slow=(7, 8, 9))
        self.assertGreater(r["achieved"], r["cost"] * 1.2)

    def test_no_slow_ticks_means_no_misses(self):
        r = C.deadline_gate(ticks=30, slow=())
        self.assertTrue(r["ok"])
        self.assertEqual(r["missed"], 0)


class TestStep(unittest.TestCase):
    def test_step_is_rejected_within_the_ideal_bound(self):
        r = C.step_gate(ticks=400)
        self.assertTrue(r["ok"])
        self.assertIsNotNone(r["settle"])

    def test_overshoot_is_reported_not_hidden(self):
        """量测低通 + 机床滞后会带来明显超调，这个数要报出来。"""
        r = C.step_gate(ticks=400)
        self.assertGreater(r["overshoot"], 1.0)


class TestSaturation(unittest.TestCase):
    def test_residual_equals_the_closed_form_floor(self):
        """补偿量被限幅时，稳态残差 = amp/(1-p) - limit。这是闭式的，
        所以能拿来验"闭环没做到的事有没有被报出来"。"""
        p, amp, limit = 0.6, 10.0, 10.0
        r = C.run(ticks=400, period=1.0, clock=C.VirtualClock(), phi=0.99,
                  sigma_w=0.0, sigma_v=0.0, p=p, g_c=0.3, beta=0.618,
                  limit=limit, step=(20, amp), seed=0)
        floor = amp / (1.0 - p) - limit
        self.assertGreater(r["saturated"], 0)
        self.assertAlmostEqual(r["err"][-1], floor, places=6)

    def test_rate_limit_is_counted(self):
        r = C.run(ticks=200, period=1.0, clock=C.VirtualClock(), phi=0.99,
                  sigma_w=0.0, sigma_v=0.0, p=0.6, g_c=0.3, beta=0.618,
                  rate=0.05, step=(20, 10.0), seed=0)
        self.assertGreater(r["rate_limited"], 0)


class TestClock(unittest.TestCase):
    def test_virtual_clock_only_advances_on_sleep(self):
        c = C.VirtualClock()
        self.assertEqual(c.now(), 0.0)
        c.sleep(0.25)
        self.assertAlmostEqual(c.now(), 0.25, places=15)
        c.sleep(-1.0)                       # 负的 sleep 不动时间
        self.assertAlmostEqual(c.now(), 0.25, places=15)

    def test_wall_clock_is_monotone(self):
        c = C.WallClock()
        a = c.now()
        c.sleep(0.001)
        b = c.now()
        self.assertGreaterEqual(b, a)


if __name__ == "__main__":
    unittest.main()
