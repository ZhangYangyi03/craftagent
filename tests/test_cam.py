# -*- coding: utf-8 -*-
"""图纸 -> 参数 -> G 代码 -> 门 这条链的测试。

每一条都针对一个**真的错过一次**的地方，不是数通过了几条：

  1. 图纸里读不懂的东西必须被**说出来**，不能静默跳过。
     计划器看不见的轮廓，后面的门也拦不住，"它跑起来了"就什么也不意味着。
  2. 圆弧要按弦高容差采样，不是按固定角度——固定角度对大圆弧超差。
  3. G 代码注释必须用 ';' 不用 '#'。'#' 是参数变量前缀，解析器会拒绝。
     这一条是我自己第一次跑就撞上的。
  4. 每齿进给要按直径**分档**，不是按直径线性缩放。线性缩放让 D10 钢粗铣
     得到 fz=0.20 mm，而那是断刀的进给，且所有自查都说"合规"。
  5. 功率校核必须能把进给夹下来。它是这张表唯一能自我反驳的地方。
  6. 回退只对**降进给能解决**的理由重试。撞刀降进给不会绕开压板，
     拿它去重试只会白跑八轮然后把真正的原因埋掉。
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from craftagent import drawing, cam, feeds, gate_cnc, gcode


def _dxf():
    def poly(layer, pts):
        s = "0\nLWPOLYLINE\n8\n%s\n90\n%d\n70\n1\n" % (layer, len(pts))
        for x, y in pts:
            s += "10\n%.4f\n20\n%.4f\n" % (x, y)
        return s

    def circ(layer, cx, cy, r):
        return "0\nCIRCLE\n8\n%s\n10\n%.4f\n20\n%.4f\n40\n%.4f\n" % (layer, cx, cy, r)

    def line(layer, x0, y0, x1, y1):
        return ("0\nLINE\n8\n%s\n10\n%.4f\n20\n%.4f\n11\n%.4f\n21\n%.4f\n"
                % (layer, x0, y0, x1, y1))

    d = "0\nSECTION\n2\nENTITIES\n"
    d += poly("CUT_-3.0", [(0, 0), (60, 0), (60, 40), (0, 40)])
    d += poly("POCKET_-1.5", [(10, 10), (30, 10), (30, 25), (10, 25)])
    d += circ("HOLE_-6.0", 45, 20, 4)
    d += line("CUT_-3.0", 0, 45, 20, 45)
    d += poly("NOTES", [(70, 0), (80, 0), (80, 10), (70, 10)])
    d += "0\nENDSEC\n0\nEOF\n"
    return d


STOCK = (-10.0, -10.0, 70.0, 50.0, 0.0, 28.0)


class TestDrawing(unittest.TestCase):

    def setUp(self):
        self.dr = drawing.read_dxf(_dxf())

    def test_closed_contours_are_read(self):
        self.assertEqual(len(self.dr.contours), 3)
        areas = sorted(round(abs(c.area()), 1) for c in self.dr.contours)
        self.assertEqual(areas, [50.1, 300.0, 2400.0])

    def test_circle_area_is_close_to_pi_r_squared(self):
        """圆的面积要逼近 pi r^2——折线化的误差有，但不该有 10% 那么大。"""
        hole = [c for c in self.dr.contours if 'HOLE' in c.layer][0]
        want = math.pi * 16.0
        self.assertAlmostEqual(abs(hole.area()), want, delta=want * 0.01)

    def test_unsupported_entities_are_reported_not_skipped(self):
        """读不懂必须说出来。静默跳过会让"图纸读完了"变成一句假话。"""
        kinds = [e[0] for e in self.dr.unsupported]
        self.assertIn('LINE', kinds)
        self.assertTrue(any('开链' in w for _, _, w in self.dr.unsupported))

    def test_missing_depth_is_reported_not_defaulted_to_zero(self):
        """图层名里没有层深，必须报出来，不能猜 0。

        猜 0 的后果很具体：轮廓被当成"在表面上划一刀"，门放行，
        件是废的，而所有报表都是绿的。
        """
        layers = {l for l, _ in self.dr.unknown_depth}
        self.assertIn('NOTES', layers)
        self.assertFalse(any('NOTES' in c.layer for c in self.dr.contours))

    def test_layer_depth_parsing(self):
        self.assertEqual(drawing.depth_from_layer('CUT_-3.0'), -3.0)
        self.assertEqual(drawing.depth_from_layer('POCKET_-1.5'), -1.5)
        self.assertIsNone(drawing.depth_from_layer('NOTES'))

    def test_chord_tolerance_scales_with_radius(self):
        """同样的容差下，大半径要更多点，小半径要更少点。

        固定 5 度对 D8 孔是 73 点，对 D200 圆弧也是 73 点：
        前者浪费，后者超差。
        """
        small = drawing.chord_steps(4.0, math.pi, 0.01)
        large = drawing.chord_steps(100.0, math.pi, 0.01)
        self.assertGreater(large, small)


class TestFeeds(unittest.TestCase):

    def test_gcode_comment_prefix_is_semicolon(self):
        """生成的程序必须能被这个仓自己的解析器读。

        G 代码里 '#' 是参数变量前缀，不是注释；用 '#' 写注释会让解析器
        带行号拒绝。这个错我第一次跑就撞上了。
        """
        dr = drawing.read_dxf(_dxf())
        plan = cam.build(dr, 'P', feeds.Tool(6, 3, 'carbide'),
                         feeds.Machine(), stock=STOCK, plan_ok=False)
        self.assertNotIn('#', plan.text)
        gcode.parse(plan.text)          # 读不进去就会抛

    def test_feed_per_tooth_is_staged_not_linear_in_diameter(self):
        """每齿进给按直径分档，不是按直径线性缩放。

        线性缩放让 D10 钢粗铣得到 fz=0.20 mm——断刀的进给，
        而进给上限、功率校核全都说"合规"。
        """
        small = feeds.choose('P', 'rough', feeds.Tool(3, 3, 'carbide'),
                             feeds.Machine(), ap_mm=0.5)
        large = feeds.choose('P', 'rough', feeds.Tool(12, 4, 'carbide'),
                             feeds.Machine(), ap_mm=0.5)
        self.assertLess(small.fz_mm, large.fz_mm)
        # 直径比是 4 倍（D3 -> D12）。线性缩放会得到 4 倍每齿进给；
        # 分档放大率必须**明显小于**直径比，否则就等于按比例缩放了。
        # 阈值写在这里是为了让这条测试可被反驳：改回线性缩放它立刻红。
        diam_ratio = 12.0 / 3.0
        fz_ratio = large.fz_mm / small.fz_mm
        self.assertLess(fz_ratio, diam_ratio * 0.8,
                        '每齿进给放大了 %.2f 倍，直径比 %.1f：这就是线性缩放'
                        % (fz_ratio, diam_ratio))

    def test_spindle_is_clamped_to_machine_limit_and_says_so(self):
        ch = feeds.choose('N', 'rough', feeds.Tool(6, 3, 'carbide'),
                          feeds.Machine(spindle_max_rpm=12000), ap_mm=1.0)
        self.assertLessEqual(ch.rpm, 12000.0)
        if ch.rpm_raw > 12000.0:
            self.assertIn('rpm_high', ch.clamped)

    def test_power_limit_can_clamp_the_feed(self):
        """功率校核必须真的能把进给夹下来。

        它是这张表唯一能自我反驳的地方：查出来的参数若让机床超功率，
        那"表说可行"就是假的。拿一台很小的机床逼它生效。
        """
        tiny = feeds.Machine(power_kw=0.5, efficiency=0.8)
        ch = feeds.choose('P', 'rough', feeds.Tool(12, 4, 'carbide'), tiny,
                          ap_mm=8.0, ae_mm=12.0)
        self.assertIn('power', ch.clamped)
        self.assertLessEqual(ch.power_kw, tiny.power_kw * tiny.efficiency + 1e-9)
        self.assertLess(ch.feed_mm_min, ch.feed_raw)

    def test_power_formula_units(self):
        """Pc = ap*ae*vf*kc/60e6。单位换错一次会差 60 倍。"""
        pc = feeds.power_kw(1.0, 10.0, 600.0, 2000.0)
        self.assertAlmostEqual(pc, 1.0 * 10.0 * 10.0 * 2000.0 / 1e6, places=9)

    def test_unknown_material_is_refused_by_name(self):
        with self.assertRaises(KeyError) as cm:
            feeds.choose('Unobtainium', 'rough', feeds.Tool(), feeds.Machine())
        self.assertIn('Unobtainium', str(cm.exception))


class TestCamClosedLoop(unittest.TestCase):

    def test_generated_program_passes_the_gate_or_is_retried(self):
        dr = drawing.read_dxf(_dxf())
        plan, atts = cam.solve(dr, 'P', feeds.Tool(6, 3, 'carbide'),
                               feeds.Machine(), stock=STOCK, ap_per_pass=1.5)
        self.assertTrue(atts)
        self.assertTrue(atts[-1].verdict.ok, '回退之后仍被拒：%s' % atts[-1].reason)

    def test_retry_actually_lowers_the_feed(self):
        """回退必须真的降低进给，而不是换个说法再来一次。"""
        dr = drawing.read_dxf(_dxf())
        plan, atts = cam.solve(dr, 'N', feeds.Tool(6, 3, 'carbide'),
                               feeds.Machine(), stock=STOCK, ap_per_pass=1.5,
                               max_tries=8)
        if len(atts) > 1:
            self.assertLess(atts[-1].plan.choices[0].feed_mm_min,
                            atts[0].plan.choices[0].feed_mm_min)

    def test_collision_refusal_is_not_retried(self):
        """撞刀降进给不会绕开压板。把几何问题拿去重试，
        只会白跑八轮之后报"试过了都不行"，把真正的原因埋掉。
        """
        dr = drawing.read_dxf(_dxf())
        # 毛坯和压板夹住整块件：生成的刀路必然进压板
        bad_stock = (0.0, 0.0, 60.0, 40.0, 0.0, 28.0)
        plan, atts = cam.solve(dr, 'P', feeds.Tool(6, 3, 'carbide'),
                               feeds.Machine(), stock=bad_stock, ap_per_pass=1.5)
        reasons = set(atts[-1].verdict.reasons)
        if reasons and not set(reasons) <= {'quality_predict', 'accel_limit'}:
            # 只允许一轮：不可由降进给解决的理由不该重试
            self.assertEqual(len(atts), 1,
                             '几何类拒绝被重试了 %d 轮：%s' % (len(atts), reasons))

    def test_generated_program_cuts_to_the_drawn_depth(self):
        """生成的程序必须真的切到图纸要求的最深那一层。

        切不到就是漏加工，而漏加工在每一条指令单独看时都合规。
        """
        dr = drawing.read_dxf(_dxf())
        plan, _ = cam.solve(dr, 'P', feeds.Tool(6, 3, 'carbide'),
                            feeds.Machine(), stock=STOCK, ap_per_pass=1.5)
        segs, _ = gcode.parse(plan.text)
        zmin = min(min(s.start[2], s.end[2]) for s in segs)
        want = min(c.depth for c in dr.contours)
        self.assertAlmostEqual(zmin, want, places=6)

    def test_empty_drawing_is_refused_by_name(self):
        dr = drawing.Drawing()
        with self.assertRaises(cam.CamError) as cm:
            cam.build(dr, 'P', feeds.Tool(), feeds.Machine())
        self.assertIn('没有可切', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
