# -*- coding: utf-8 -*-
"""撞刀那四条规则的测试。每一条都针对一次**真的差点漏掉**的地方。

这个文件不是"再数一遍通过了多少条"。它是四件具体的事：

  1. 四条规则各自都要能拦住它该拦的那一条，且**是它拦的**，
     不是别的规则顺手拦下。这一条曾经是假的：撞刀故障的坐标落在
     行程之外，于是 axis_limit 先拦了，那四条一次也没被考过。
  2. 每条故障只碰一个字段。撞刀故障一开始没带 M3 S，就被 spindle
     规则先拦了——同一个错换了个方向又出现一遍。
  3. 抬刀不算穿料。每个程序结尾都有 G0 Z 直着出来，判错它的话
     门会拒绝世界上每一份程序。
  4. 圆弧要沿弧采样。只取弦或只取端点会漏掉鼓出来的那一段。
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from craftagent import collide, gcode, gate_cnc


class TestCollisionRules(unittest.TestCase):

    def setUp(self):
        self.fx = collide.Fixture()
        self.gate = gate_cnc.Gate(gate_cnc.Limits(), fixture=self.fx)

    def test_every_planted_collision_is_caught_by_its_own_rule(self):
        """四条撞刀规则各自拦住自己那一条，而且真的是它拦的。

        如果 axis_limit 或 spindle 先拦下，这条测试会失败——那正是它该做的：
        "被别的规则拦住了"和"这条规则在工作"是两件不同的事。
        """
        for name, text, want in collide.planted_faults(self.fx):
            v = self.gate.check_text(text)
            self.assertFalse(v.ok, '%s 被放行了' % name)
            self.assertIn(want, v.reasons,
                          '%s 由 %r 拦下，不是 %r' % (name, v.reasons, want))

    def test_legal_programs_are_not_refused(self):
        """误拒数是门的另一半。只报拦截率，一个全部拒绝的门能拿满分。"""
        for name, text in collide.legal_programs(self.fx):
            v = self.gate.check_text(text)
            self.assertTrue(v.ok, '%s 被误拒: %r %r'
                            % (name, v.reasons, v.detail))

    def test_switching_a_collision_rule_off_collapses_only_its_own_faults(self):
        """消融：关掉一条，对应那一条必须漏过去。漏不过去说明它没在起作用。"""
        for rule in gate_cnc.COLLIDE_RULES:
            others = [r for r in gate_cnc.ALL_RULES if r != rule]
            g2 = gate_cnc.Gate(gate_cnc.Limits(), fixture=self.fx, rules=others)
            for name, text, want in collide.planted_faults(self.fx):
                if want != rule:
                    continue
                v = g2.check_text(text)
                self.assertTrue(v.ok,
                                '关掉 %s 之后 %s 仍被 %r 拦下，那条规则没在起作用'
                                % (rule, name, v.reasons))

    def test_lifting_out_of_the_part_is_not_a_rapid_through_material(self):
        """G0 Z 直着抬出工件是合法的，而且是每个程序的最后一行。

        这一条是实测出来的：不加区分地把"快速移动经过料"判成撞刀，
        会让门拒绝世界上每一份正常程序，而门看起来还是很严。
        """
        st = self.fx.stock
        sz = self.fx.safe_z()
        text = ('G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X30 Y40\n'
                'G1 Z-2.0 F150\nG0 Z%.1f\nM5\n' % (sz, sz))
        v = self.gate.check_text(text)
        self.assertTrue(v.ok, '抬刀被当成穿料: %r' % (v.reasons,))

    def test_rapid_through_material_is_caught(self):
        """同一位置上，横向穿过料必须被拦——所以上一条不是"这条规则没用"。"""
        st = self.fx.stock
        sz = self.fx.safe_z()
        text = ('G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\n'
                'G1 Z-2.0 F150\nG0 X%.1f Y%.1f\nM5\n'
                % (sz, st.x0 + 3, st.y0 + 3, st.x1 - 3, st.y1 - 3))
        v = self.gate.check_text(text)
        self.assertIn('rapid_into_stock', v.reasons)

    def test_holder_into_stock_is_a_rule_of_its_own(self):
        """刀柄进料与刀尖撞压板是两件事，不能合并。

        我原来还写了一条"刀柄撞压板"，写完发现它是死代码：刀柄与刀尖同轴，
        且 holder_bottom = 刀尖 + 伸出量 恒在刀尖之上，所以刀柄进压板时
        刀尖必然更在里面。那条规则永远轮不到，删掉了。
        真正独立的只有"刀柄进料"：料是有限厚的块，刀尖切在里面、
        刀柄也扎进来，那是刀装得太短——与压板无关。
        """
        st, tool = self.fx.stock, self.fx.tool
        sz = self.fx.safe_z()
        mid = ((st.x0 + st.x1) / 2.0, (st.y0 + st.y1) / 2.0)
        z = st.top_z - (tool.stickout + 2.0)
        self.assertGreater(z, st.table_z, '这一刀应当还没到台面（否则拦它的是 table_crash）')
        text = 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\nM5\n' % (
            sz, mid[0], mid[1], z)
        v = self.gate.check_text(text)
        self.assertIn('holder_crash', v.reasons)
        self.assertIn('刀柄', v.detail['holder_crash'])

    def test_tip_crash_and_holder_crash_are_distinguishable(self):
        """同一个刀尖位置, 深到伸出量以下才叫刀柄撞。

        只报"撞了"没有用：修刀尖和修伸出量是两个完全不同的动作。
        """
        st, tool = self.fx.stock, self.fx.tool
        sz = self.fx.safe_z()
        mid = ((st.x0 + st.x1) / 2.0, (st.y0 + st.y1) / 2.0)
        # 浅切：刀柄还在料顶之上
        shallow = 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\nM5\n' % (
            sz, mid[0], mid[1], st.top_z - 2.0)
        self.assertNotIn('holder_crash', self.gate.check_text(shallow).reasons)
        # 深切：刀柄扎进料里
        deep = 'G21 G90\nM3 S8000\nG0 Z%.1f\nG0 X%.1f Y%.1f\nG1 Z%.1f F100\nM5\n' % (
            sz, mid[0], mid[1], st.top_z - (tool.stickout + 2.0))
        self.assertIn('holder_crash', self.gate.check_text(deep).reasons)

    def test_arc_is_sampled_along_the_curve_not_chorded(self):
        """圆弧采样必须沿弧走。取弦会漏掉鼓出去的那一段。"""
        seg = gcode.Seg(line=1, text='G2', kind='arc',
                        start=(10.0, 0.0, 0.0), end=(-10.0, 0.0, 0.0),
                        feed=100.0, length=0.0, centre=(0.0, 0.0),
                        ccw=True, radius=10.0, t_in=None, t_out=None,
                        s=1, spindle=None, rpm=None)
        pts = collide.sample_seg(seg, step=0.5)
        # 半圆弧的中点在 (0,10) 或 (0,-10)，弦上没有这个点
        apex_y = max(abs(p[1]) for p in pts)
        self.assertGreater(apex_y, 9.0,
                           '沿弧采样时最远点应当在半径处，实测 %.3f' % apex_y)
        for p in pts:
            self.assertAlmostEqual(math.hypot(p[0], p[1]), 10.0, places=6)

    def test_machine_z_can_clear_the_fixture(self):
        """机床的 Z 行程必须够退到夹具之上。

        默认 Z 上限原来是 10 mm，而压板顶面在 12 mm、安全 Z 在 14 mm：
        这台机床按默认参数**退不出夹具**，任何一次正确的抬刀都会被
        axis_limit 拒绝。这两个默认值互相矛盾，而且矛盾的方向正好是
        "看起来门很严"。
        """
        lim = gate_cnc.Limits()
        (_, _), (_, _), (zlo, zhi) = lim.envelope
        self.assertGreater(zhi, self.fx.safe_z(),
                           'Z 上限 %.1f 够不到安全 Z %.1f'
                           % (zhi, self.fx.safe_z()))
        self.assertLess(zlo, self.fx.stock.table_z)


if __name__ == '__main__':
    unittest.main()
