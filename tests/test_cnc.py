# -*- coding: utf-8 -*-
"""G 代码 -> 内核 -> 门 -> 虚拟机床 -> 归因 这条链的测试。

每一条都针对一个**真的错过一次**的地方。测试写在这里是为了让那个错
不会悄悄回来，不是为了数通过了几条。
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from craftagent import gcode, plan, plant, gate_cnc, cnc


class TestParse(unittest.TestCase):

    def test_arc_geometry_is_checked_not_assumed(self):
        """圆弧终点不在它所声称的圆上时必须拒绝，并报出行号。

        这是 gate 的 gcode_syntax 规则唯一真正能抓到的东西：解析器读不懂的
        东西，门也拦不住，于是"它跑起来了"什么也不意味着。
        """
        with self.assertRaises(gcode.GcodeError) as cm:
            gcode.parse('G21 G90\nG1 X0 Y0 F500\nG2 X5 Y5 I0 J1')
        self.assertEqual(cm.exception.line, 3)

    def test_feed_is_mm_per_min_in_the_parser(self):
        """F 在 G 代码里是 mm/min。解析器原样保留，换算留给计划器。

        单位换错一次代价很大：900 mm/min 被当成 900 mm/s 之后，
        所有速度、时间、误差全错，而公式全对，很难看出来。
        """
        segs, _ = gcode.parse('G21 G90\nG1 X10 F900')
        self.assertAlmostEqual(segs[0].feed, 900.0, places=6)

    def test_g0_and_g1_are_distinguishable(self):
        """G0（快速定位）和 G1（切削进给）语义不同，必须分得开。

        压成同一种 'line' 之后，质量预测会把空行程也按规格评判，
        一个健康程序会因为"快速移动太粗"被判报废。
        """
        segs, _ = gcode.parse('G21 G90\nG0 X10\nG1 X20 F500')
        self.assertEqual(segs[0].s, 0)
        self.assertEqual(segs[1].s, 1)


class TestPlan(unittest.TestCase):

    def test_peak_never_exceeds_its_own_cap(self):
        """每段的峰值速度不得超过这段自己的上限。

        原来峰值只由加速度和段长算出，从没被 cap 夹住，
        于是 F900 的段跑出了 251 mm/s 的峰值。公式全对，结果超限。
        """
        segs, _ = gcode.parse('G21 G90\nG1 X0 Y0 F900\nG1 X30 F900\nG1 X0 Y30 F900')
        p = plan.plan(segs)
        for r in p['rows']:
            if r['feed']:
                self.assertLessEqual(r['v_peak'], r['feed'] / 60.0 + 1e-6)

    def test_corner_slows_down(self):
        """反向拐角必须降速，否则机床在拐角处抖。

        两个反向直线段之间的过弯速度应当远低于各段的名义进给：
        sqrt(a * dev) 里 dev 是允许的拐角偏差，与进给无关。
        """
        segs, _ = gcode.parse('G21 G90\nG1 X0 Y0 F900\nG1 X30 F900\nG1 X0 Y30 F900')
        p = plan.plan(segs)
        mid = p['rows'][1]
        self.assertLess(mid['v_exit'], mid['v_peak'] * 0.75)

    def test_no_zero_speed_stall(self):
        """插补器不允许在弧长没走完时速度退到 0 并原地打转。

        原来在某些段上 a 被算成 ~0、v 被求成 0，tick 永不推进——
        进程直接卡死，不是慢，是死循环。
        """
        segs, _ = gcode.parse('G21 G90\nG1 Z-1 F150\nG1 X30 F900')
        p = plan.plan(segs)
        for r in p['rows']:
            if r['length_mm'] > 1e-9:
                self.assertGreater(r['v_peak'], 0.0)
                self.assertGreater(r['t_s'], 0.0)


class TestPlant(unittest.TestCase):

    def test_accel_phase_error_matches_the_closed_form(self):
        """加速期间的跟随误差要用闭式解，不能用稳态值。

        e(t) = a*tau^2*(t/tau - 1 + exp(-t/tau))
        拿稳态的 a/kv^2 代替它会把误差算小几十倍（实测 24 倍），
        而算小的方向恰好是"看起来一切良好"的方向。
        """
        a, tau = 1500.0, 0.0008
        # 数值积分同一台一阶伺服，与闭式解比
        dt, T = 1e-7, 0.005
        xc = xa = vc = 0.0
        t = 0.0
        while t < T:
            vc += a * dt
            xc += vc * dt
            xa += (xc - xa) * (dt / tau)
            t += dt
        num = (xc - xa) * 1000.0
        ana = plant.err_accel_phase(a, T, tau) * 1000.0
        self.assertLess(abs(num - ana) / max(1e-9, ana), 0.02)

    def test_actual_accel_uses_d_acc_not_segment_time(self):
        """段内加速度必须从 d_acc 反算，不能拿整段时长当加速时长。

        一段 60 mm 的行程加速只占 0.07 mm，却被当成 2 秒，
        加速度被低估几十倍，质量被算好——方向上总是"没问题"，最坏的一种错。
        """
        segs, _ = gcode.parse('G21 G90\nG1 X0 Y0 F150\nG1 X60 F600')
        p = plan.plan(segs)
        r = [x for x in p['rows'] if x['length_mm'] > 1.0][0]
        a = plant.actual_accel(r)
        # 规划器按 a_max 规划，所以真实加速度应当接近 a_max，不是接近 0
        self.assertGreater(a, p['params']['a_max_mm_s2'] * 0.5)

    def test_quality_is_not_an_affine_copy_of_any_channel(self):
        """质量不能是某个通道的仿射变换。

        q = q_base + k*follow_err 时，跟随误差通道就是质量的精确线性像，
        它在归因里必然排第一——那不是发现了原因，是把我自己写的那行公式
        念了一遍。这是循环论证，正是归因要防的那件事。
        """
        segs, _ = gcode.parse('G21 G90\nG1 X0 Y0 F150\n' + '\n'.join(
            'G1 X%.1f Y%d F%d' % (60 if i % 2 == 0 else 0, 12 * i, 600)
            for i in range(1, 8)))
        p = plan.plan(segs)
        M = plant.machine_pass(p['rows'], p['params'])
        pts = [(m['follow_err_um'], m['roundness_um']) for m in M
               if m['roundness_um'] is not None and m['follow_err_um'] > 0]
        # 相关系数不该是 1：允许很强，不允许是恒等仿射
        n = len(pts)
        mx = sum(a for a, _ in pts) / n
        my = sum(b for _, b in pts) / n
        sxy = sum((a - mx) * (b - my) for a, b in pts)
        sxx = sum((a - mx) ** 2 for a, _ in pts)
        syy = sum((b - my) ** 2 for _, b in pts)
        r = sxy / math.sqrt(sxx * syy) if sxx * syy > 0 else 0.0
        self.assertLess(abs(r), 0.999)

    def test_rapid_moves_have_no_quality(self):
        """空行程没有圆度。填 0 会让它看起来很好，填实测值会让它看起来很糟。"""
        segs, _ = gcode.parse('G21 G90\nG0 X10 F2000\nG1 X20 F600')
        p = plan.plan(segs)
        M = plant.machine_pass(p['rows'], p['params'])
        rapid = [m for m in M if m['line'] == 2][0]
        self.assertIsNone(rapid['roundness_um'])


class TestGate(unittest.TestCase):

    def _good(self):
        return ('G21 G90\nG0 Z5 F2000\nM3 S8000\nG1 X0 Y0 F150\nG1 Z-1\n'
                'G1 X60 F600\nG1 Y12\nG1 X0\nM5')

    def test_legal_program_is_admitted(self):
        self.assertTrue(gate_cnc.check_text(self._good()).ok)

    def test_over_travel_names_the_axis_and_the_line(self):
        v = gate_cnc.check_text('G21 G90\nM3 S8000\nG1 X900 F500')
        self.assertFalse(v.ok)
        self.assertIn('axis_limit', v.reasons)
        self.assertIn('X=', v.detail['axis_limit'])

    def test_quality_predict_blocks_what_limits_do_not(self):
        """进给在额定之内、行程也在之内，但质量做不到，门必须拦。

        这一条是"事前拦截"相对"事后抽检"的全部价值所在：
        它是唯一能拦住"每个参数都合规、结果仍报废"的那一类。
        """
        fast = ('G21 G90\nM3 S8000\nG1 X0 Y0 F150\nG1 Z-1\n'
                + '\n'.join('G1 X%.1f Y%d F900' % (60 if i % 2 == 0 else 0, 12 * i)
                            for i in range(1, 9)))
        v = gate_cnc.check_text(fast)
        self.assertFalse(v.ok)
        self.assertIn('quality_predict', v.reasons)

    def test_rapid_is_not_judged_by_the_quality_spec(self):
        """健康程序里的 G0 不该被质量门拦下——它不切材料。"""
        self.assertTrue(gate_cnc.check_text(self._good()).ok)


class TestEndToEnd(unittest.TestCase):

    def test_change_point_lands_on_the_planted_line(self):
        """埋在第 4 次走刀的故障，变点必须落在附近 ±2 次走刀内。"""
        text, truth = cnc.demo_program(bad_line_at=4, bad_factor=2.6, persist=True)
        r = cnc.analyse(text, gate_on=False)
        self.assertTrue(r['change_points'])
        target = None
        for i, m in enumerate(r['passes']):
            if m['line'] == truth['bad_line']:
                target = i
        self.assertIsNotNone(target)
        best = min(r['change_points'], key=lambda c: abs(c['index'] - target))
        self.assertLessEqual(abs(best['index'] - target), 2)

    def test_feed_is_the_top_effect_when_feed_is_the_fault(self):
        """故障就是抬高进给，那么变点前后动得最多的通道必须是进给。

        用效应量而不是相关大小：故障是阶跃，拿相关去评判阶跃是在用错尺子，
        连续量必然赢，而那个"赢"与模型对不对无关。
        """
        text, _ = cnc.demo_program(bad_line_at=4, bad_factor=2.6, persist=True)
        r = cnc.analyse(text, gate_on=False)
        self.assertEqual(r['attribution']['by_effect'][0], 'feed_mm_min')

    def test_healthy_program_has_no_strong_change_point(self):
        """健康程序不该在稳态段里报出强变点。

        开机瞬态（首段 G0、下刀、首次切入）与稳态是两个问题，
        混在一张序列上，检测器给出的答案会同时属于两者。
        """
        text, _ = cnc.demo_program(bad_line_at=None)
        r = cnc.analyse(text, gate_on=False)
        strong = [c for c in r['change_points'] if c['score'] >= 3.0]
        self.assertEqual(strong, [])


class TestParity(unittest.TestCase):
    """C 内核与 Python 规划器必须逐段一致。

    同一份代码跑两遍永远同意，那测的是确定性，不是正确性。
    两份独立实现同意，才叫规划结果是对的。
    """

    def test_c_and_python_agree(self):
        from bench import pair
        r = pair.compare(pair.standard_programs()['zigzag_arcs'])
        if r.get('ok') is None:
            self.skipTest('cncrt.exe 未编译（先跑 bench/build_cncrt.py）')
        self.assertTrue(r['ok'], r.get('reason') or
                        'max|dv|=%s max|dt|=%s' % (r['max_dv_mm_s'], r['max_dt_s']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
