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
        """读不懂必须说出来。静默跳过会让"图纸读完了"变成一句假话。

        这条测试第一版断言的是"LINE 进 unsupported"，因为当时 LINE 单独
        一条切不出件、也不拼环。拼环做出来之后 LINE 不再是"读不懂"，
        而是"攒起来用"——所以断言必须跟着改，否则它会一边绿着、
        一边描述一个已经不存在的行为。

        现在它盯的是**真的读不懂**那一类：块引用（INSERT）需要块表加变换栈，
        这个仓没做，所以它必须被点名。
        """
        parts = ["0", "SECTION", "2", "ENTITIES",
                 "0", "INSERT", "8", "CUT_-3.0", "2", "BLOCK_A",
                 "10", "0", "20", "0",
                 "0", "ENDSEC", "0", "EOF"]
        dr2 = drawing.read_dxf(chr(10).join(parts))
        self.assertTrue(any(e[0] == 'INSERT' for e in dr2.unsupported),
                        'INSERT 没被报出来：%s' % (dr2.unsupported,))

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


def _line(layer, x0, y0, x1, y1):
    return ("0\nLINE\n8\n%s\n10\n%.4f\n20\n%.4f\n11\n%.4f\n21\n%.4f\n"
            % (layer, x0, y0, x1, y1))


def _dxf_lines_only(edges, extra=""):
    d = "0\nSECTION\n2\nENTITIES\n"
    for e in edges:
        d += _line("CUT_-3.0", *e)
    d += extra
    d += "0\nENDSEC\n0\nEOF\n"
    return d


class TestStitching(unittest.TestCase):
    """真实图纸里绝大多数轮廓是一堆 LINE 首尾相接，不是 LWPOLYLINE。

    不拼环就等于"这张图纸没有可切的轮廓"——而那通常是假的。
    """

    RECT = [(0, 0, 60, 0), (60, 0, 60, 40), (60, 40, 0, 40), (0, 40, 0, 0)]

    def test_four_lines_in_order_become_one_contour(self):
        dr = drawing.read_dxf(_dxf_lines_only(self.RECT))
        self.assertEqual(len(dr.contours), 1)
        self.assertAlmostEqual(abs(dr.contours[0].area()), 2400.0, places=6)
        self.assertEqual(len(dr.contours[0].pts), 4)

    def test_shuffled_lines_still_stitch(self):
        """顺序打乱必须还能拼。

        图纸导出时实体的先后顺序是任意的，一个"必须按顺序给"的拼接器
        在实际图纸上一次也用不上。
        """
        import random
        rnd = random.Random(11)
        for _ in range(20):
            edges = list(self.RECT)
            rnd.shuffle(edges)
            dr = drawing.read_dxf(_dxf_lines_only(edges))
            self.assertEqual(len(dr.contours), 1,
                             '打乱后没拼成：%s' % (edges,))
            self.assertAlmostEqual(abs(dr.contours[0].area()), 2400.0, places=6)

    def test_reversed_edges_still_stitch(self):
        """边的方向不一致必须还能拼。

        DXF 不保证相邻两段的走向一致，真实图纸里正反混着来很常见。
        """
        edges = [(0, 0, 60, 0), (60, 0, 60, 40), (0, 40, 60, 40), (0, 0, 0, 40)]
        dr = drawing.read_dxf(_dxf_lines_only(edges))
        self.assertEqual(len(dr.contours), 1)
        self.assertAlmostEqual(abs(dr.contours[0].area()), 2400.0, places=6)

    def test_a_real_open_chain_is_reported_with_its_gap(self):
        """拼不上的必须报出来，并说出缺口多大。

        "拼不上"和"这里差 72 mm"是两条不同的信息：前者让人以为图纸坏了，
        后者能让人直接去图纸里找那 72 mm。
        """
        dr = drawing.read_dxf(_dxf_lines_only([(0, 0, 60, 0), (60, 0, 60, 40)]))
        self.assertEqual(len(dr.contours), 0)
        st = dr.stitch.as_dict()
        self.assertEqual(len(st['open_chains']), 1)
        self.assertGreater(st['open_chains'][0]['gap_mm'], 1.0)

    def test_junction_is_counted_not_hidden(self):
        """三条边共用一个端点时必须**数出来**，而不是悄悄挑一条。

        一个悄悄挑一条的拼接器会生出一条看起来完全正常的轮廓，而它是错的。
        报告里必须有一行说"这条轮廓经过了 N 个分叉，每个都是猜的"。
        """
        square = list(self.RECT)
        spur = [(60, 0, 90, 0), (90, 0, 90, 30), (90, 30, 60, 0)]
        dr = drawing.read_dxf(_dxf_lines_only(square + spur))
        self.assertEqual(len(dr.contours), 1)
        self.assertEqual(dr.contours[0].assumed_junctions, 1)
        st = dr.stitch.as_dict()
        self.assertEqual(len(st['junctions']), 1)
        self.assertEqual(st['junctions'][0]['edges'], 4)

    def test_stitch_does_not_cross_layers(self):
        """不同图层是不同的层深，跨层拼接会悄悄把半条轮廓的深度改掉。"""
        d = "0\nSECTION\n2\nENTITIES\n"
        d += _line("CUT_-3.0", 0, 0, 60, 0)
        d += _line("CUT_-3.0", 60, 0, 60, 40)
        d += _line("POCKET_-1.5", 60, 40, 0, 40)
        d += _line("POCKET_-1.5", 0, 40, 0, 0)
        d += "0\nENDSEC\n0\nEOF\n"
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 0, '跨图层拼了环：%s' % (dr.contours,))

    def test_mixed_polyline_and_lines_both_survive(self):
        d = "0\nSECTION\n2\nENTITIES\n"
        d += ("0\nLWPOLYLINE\n8\nPOCKET_-1.5\n90\n4\n70\n1\n"
              "10\n10\n20\n10\n10\n30\n20\n10\n10\n30\n20\n25\n10\n10\n20\n25\n")
        for e in self.RECT:
            d += _line("CUT_-3.0", *e)
        d += "0\nENDSEC\n0\nEOF\n"
        dr = drawing.read_dxf(d)
        areas = sorted(round(abs(c.area()), 1) for c in dr.contours)
        self.assertEqual(areas, [300.0, 2400.0])
        self.assertEqual(dr.unsupported, [])

    def test_truncated_file_is_not_reported_as_empty(self):
        """段没有 ENDSEC 结尾时也要收下。

        真实图纸会被截断（U 盘拷一半、邮件附件断掉）。截断的文件里
        ENTITIES 仍然可读，而"一个段都没闭合所以整张图是空的"
        是一句比事实更严重的谎。
        """
        d = "0\nSECTION\n2\nENTITIES\n"
        d += _line("CUT_-3.0", 0, 0, 10, 0)
        d += _line("CUT_-3.0", 10, 0, 10, 10)
        d += _line("CUT_-3.0", 10, 10, 0, 0)
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        self.assertAlmostEqual(abs(dr.contours[0].area()), 50.0, places=6)

    def test_layer_without_depth_is_not_stitched(self):
        """深度不明就不拼：拼出一条深度是错的轮廓，比拼不出来更糟。"""
        d = "0\nSECTION\n2\nENTITIES\n"
        for e in self.RECT:
            d += _line("NOTES", *e)
        d += "0\nENDSEC\n0\nEOF\n"
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 0)
        self.assertTrue(dr.stitch.as_dict()['open_chains'])


def _nl(*items):
    return "\n".join(str(x) for p in items for x in p) + "\n"


def _block_dxf(block_name, block_items, insert_items, extra_ents=()):
    d = _nl((0, "SECTION"), (2, "BLOCKS"))
    d += _nl(*block_items)
    d += _nl((0, "ENDBLK"))
    d += _nl((0, "ENDSEC"))
    d += _nl((0, "SECTION"), (2, "ENTITIES"))
    d += _nl(*insert_items)
    if extra_ents:
        d += _nl(*extra_ents)
    d += _nl((0, "ENDSEC"), (0, "EOF"))
    return d


SQUARE_IN_BLOCK = [
    (0, "BLOCK"), (8, "0"), (2, "SQ"), (70, 0), (10, 0), (20, 0),
    (0, "LWPOLYLINE"), (8, "POCKET_-1.5"), (90, 4), (70, 1),
    (10, 0), (20, 0), (10, 10), (20, 0), (10, 10), (20, 6), (10, 0), (20, 6),
]


class TestBlocks(unittest.TestCase):
    """块引用（INSERT）展开。

    真实图纸里重复出现的特征——型腔、孔组、倒角——几乎一定是一个块
    + 多次引用。不展开就等于把这些特征全部丢掉，而丢掉之后图纸看上去
    仍然"读完了"，只是件少做了一大半。
    """

    def test_insert_is_expanded_with_translation(self):
        d = _block_dxf("SQ", SQUARE_IN_BLOCK,
                       [(0, "INSERT"), (8, "POCKET_-1.5"), (2, "SQ"),
                        (10, 30), (20, 20)])
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        bx = tuple(round(v, 3) for v in dr.contours[0].bounds())
        self.assertEqual(bx, (30.0, 20.0, 40.0, 26.0))
        self.assertAlmostEqual(abs(dr.contours[0].area()), 60.0, places=6)
        self.assertEqual(dr.unsupported, [])

    def test_insert_rotation_is_applied(self):
        """旋转必须真的转。

        转错的后果不是"报错"，是"件做在错的位置上"——而每条指令都合法。
        这里用 90 度，因为 10x6 的矩形转完必须变成 6x10（宽高互换），
        一个把旋转忽略掉的实现会让宽高仍然保持 10 和 6。
        """
        d = _block_dxf("SQ", SQUARE_IN_BLOCK,
                       [(0, "INSERT"), (8, "POCKET_-1.5"), (2, "SQ"),
                        (10, 20), (20, 20), (50, 90)])
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        x0, y0, x1, y1 = dr.contours[0].bounds()
        self.assertAlmostEqual(x1 - x0, 6.0, places=6)
        self.assertAlmostEqual(y1 - y0, 10.0, places=6)

    def test_insert_scale_is_applied(self):
        d = _block_dxf("SQ", SQUARE_IN_BLOCK,
                       [(0, "INSERT"), (8, "POCKET_-2.5"), (2, "SQ"),
                        (10, 0), (20, 0), (41, 2), (42, 2)])
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        self.assertAlmostEqual(abs(dr.contours[0].area()), 240.0, places=6)

    def test_block_base_point_is_subtracted(self):
        """块基点不是原点时，引用必须减去基点。

        基点用错的表现是整块偏移一个固定量——量一下就知道，但只有量才知道。
        """
        items = list(SQUARE_IN_BLOCK)
        # 把基点从 (0,0) 改到 (10,6)，也就是矩形自己的右上角。
        # SQUARE_IN_BLOCK 里组码 (10,0),(20,0) 在索引 4、5，紧跟在 (70,0) 后面。
        self.assertEqual(items[3], (70, 0))
        items[4] = (10, 10)
        items[5] = (20, 6)
        d = _block_dxf("SQ", items,
                       [(0, "INSERT"), (8, "POCKET_-1.5"), (2, "SQ"),
                        (10, 100), (20, 100)])
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        x0, y0, x1, y1 = dr.contours[0].bounds()
        # 基点 (10,6) 落到 (100,100)，所以矩形是 (90,94)-(100,100)
        self.assertAlmostEqual(x0, 90.0, places=6)
        self.assertAlmostEqual(y0, 94.0, places=6)

    def test_missing_block_is_reported_not_silently_dropped(self):
        d = _block_dxf("SQ", SQUARE_IN_BLOCK,
                       [(0, "INSERT"), (8, "POCKET_-1.5"), (2, "NOPE"),
                        (10, 0), (20, 0)])
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 0)
        self.assertTrue(any('没有的块' in w for _, _, w in dr.unsupported))

    def test_circular_block_reference_terminates(self):
        """A 引用 B、B 引用 A 是合法 DXF，但照直展开会无限递归。

        无限递归的表现是"卡住"——最难看出原因的那种失败，
        而且它的表现和"这台机器很慢"一模一样。
        """
        d = _nl((0, "SECTION"), (2, "BLOCKS"))
        d += _nl((0, "BLOCK"), (8, "0"), (2, "A"), (70, 0), (10, 0), (20, 0))
        d += _nl((0, "INSERT"), (8, "0"), (2, "B"), (10, 0), (20, 0))
        d += _nl((0, "ENDBLK"))
        d += _nl((0, "BLOCK"), (8, "0"), (2, "B"), (70, 0), (10, 0), (20, 0))
        d += _nl((0, "INSERT"), (8, "0"), (2, "A"), (10, 0), (20, 0))
        d += _nl((0, "ENDBLK"))
        d += _nl((0, "ENDSEC"))
        d += _nl((0, "SECTION"), (2, "ENTITIES"))
        d += _nl((0, "INSERT"), (8, "CUT_-1"), (2, "A"), (10, 0), (20, 0))
        d += _nl((0, "ENDSEC"), (0, "EOF"))
        dr = drawing.read_dxf(d)
        self.assertTrue(any('循环' in w for _, _, w in dr.unsupported))

    def test_annotation_entities_are_skipped_quietly(self):
        """标注/文字不参与切削，跳过是对的，而且要不出声。

        把它们报成"读不懂"，每张真实图纸都会带几十条噪声，
        噪声里的真问题会被淹掉。
        """
        d = _block_dxf("SQ", SQUARE_IN_BLOCK,
                       [(0, "INSERT"), (8, "POCKET_-1.5"), (2, "SQ"),
                        (10, 0), (20, 0)],
                       extra_ents=[(0, "TEXT"), (8, "NOTES"), (1, "hello"),
                                   (0, "DIMENSION"), (8, "NOTES")])
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        self.assertEqual(dr.unsupported, [])


class TestPathGeneration(unittest.TestCase):
    """刀心偏置与下刀方式。

    这两件事的错法都是**沉默**的：偏置算小了件就小一圈，
    直插下刀会把刀顶崩——两者都不产生任何一行非法指令。
    所以它们必须有测试按数字盯住。
    """

    RECT = [(0, 0), (60, 0), (60, 40), (0, 40)]

    def test_offset_of_a_rectangle_is_exact(self):
        """矩形偏 5 mm 必须正好是 (-5,-5)(65,-5)(65,45)(-5,45)。

        第一版这里给的是 (-4.564,-4.564)：它把余弦夹了下限 0.2，
        于是直角的 miter 系数算成 1.291 而不是 1.414。
        件会小 0.44 mm，每次都在，量一下才知道。
        """
        got = cam.offset_contour(self.RECT, 5.0)
        want = [(-5.0, -5.0), (65.0, -5.0), (65.0, 45.0), (-5.0, 45.0)]
        for g, w in zip(got, want):
            self.assertAlmostEqual(g[0], w[0], places=6)
            self.assertAlmostEqual(g[1], w[1], places=6)

    def test_offset_direction_follows_winding(self):
        """绕向反过来时，偏移也必须反过来。

        一个靠"假设逆时针"写的实现，在顺时针轮廓上会往件里面偏一个刀半径。
        """
        ccw = cam.offset_contour(self.RECT, 5.0)
        cw = cam.offset_contour(list(reversed(self.RECT)), 5.0)
        self.assertGreater(abs(cam._signed_area(ccw)), abs(cam._signed_area(self.RECT)))
        self.assertGreater(abs(cam._signed_area(cw)), abs(cam._signed_area(self.RECT)))

    def test_ramp_entry_actually_descends_while_moving(self):
        """斜插的每一行必须同时有 xy 位移和 z 下降。

        只有 z 没有 xy 那还是直插；有 xy 但 z 一步到位也还是直插。
        这条测试盯的是"下刀过程真的斜着"。
        """
        pts = self.RECT
        steps = cam._ramp_in(pts, 0, 2.0, 5.0, 300.0)
        self.assertTrue(steps, '斜插没有产生任何一步')
        self.assertAlmostEqual(steps[-1][2], -2.0, places=6)
        prev = (pts[0][0], pts[0][1], 0.0)
        for x, y, z, f in steps:
            moved = abs(x - prev[0]) + abs(y - prev[1])
            dropped = abs(z - prev[2])
            self.assertGreater(moved, 1e-9, '这一步只有 z，还是直插')
            self.assertGreater(dropped, 1e-9, '这一步没有下降')
            self.assertLess(z, 1e-12)
            prev = (x, y, z)

    def test_ramp_angle_zero_means_plunge(self):
        self.assertEqual(cam._ramp_in(self.RECT, 0, 2.0, 0.0, 300.0), [])

    def test_build_records_the_offset_it_applied(self):
        dr = drawing.read_dxf(_dxf_lines_only(
            [(0, 0, 60, 0), (60, 0, 60, 40), (60, 40, 0, 40), (0, 40, 0, 0)]))
        plan = cam.build(dr, material='P', tool=feeds.Tool(10.0, 3, 'carbide'),
                         op='rough', ap_per_pass=1.5, offset='tool_radius')
        rec = plan.meta['offsets'][0]
        self.assertAlmostEqual(rec['dist'], 5.0, places=6)
        self.assertEqual(rec['kind'], 'point_normal_approx')
        self.assertIn('G1 X-5 Y-5', plan.text)
        self.assertIn('G1 X65 Y45', plan.text)

    def test_finish_allowance_shrinks_the_offset(self):
        dr = drawing.read_dxf(_dxf_lines_only(
            [(0, 0, 60, 0), (60, 0, 60, 40), (60, 40, 0, 40), (0, 40, 0, 0)]))
        plan = cam.build(dr, material='P', tool=feeds.Tool(10.0, 3, 'carbide'),
                         op='rough', ap_per_pass=1.5,
                         offset='tool_radius_minus_allow', finish_allow=0.3)
        self.assertAlmostEqual(plan.meta['offsets'][0]['dist'], 4.7, places=6)
        self.assertIn('G1 X-4.7 Y-4.7', plan.text)

    def test_allowance_larger_than_radius_is_refused(self):
        """留量不小于刀半径时刀心落在件里面，必须拒绝而不是照写。"""
        dr = drawing.read_dxf(_dxf_lines_only(
            [(0, 0, 60, 0), (60, 0, 60, 40), (60, 40, 0, 40), (0, 40, 0, 0)]))
        with self.assertRaises(cam.CamError):
            cam.build(dr, material='P', tool=feeds.Tool(10.0, 3, 'carbide'),
                      op='rough', offset='tool_radius_minus_allow',
                      finish_allow=5.0)

    def test_unknown_offset_mode_is_refused(self):
        dr = drawing.read_dxf(_dxf_lines_only(
            [(0, 0, 60, 0), (60, 0, 60, 40), (60, 40, 0, 40), (0, 40, 0, 0)]))
        with self.assertRaises(cam.CamError):
            cam.build(dr, material='P', op='rough', offset='comp_left')


class TestContourRole(unittest.TestCase):
    """外轮廓 vs 型腔：刀心偏置的方向由**角色**定，不由绕向定。

    偏错方向是过切（切成废件），而且每一条指令都合法、每一项检查都通过。
    这是这条链上唯一一个"错得完全看不出来"的地方，所以它必须被按数字盯住。
    """

    OUTER = [(0, 0), (60, 0), (60, 40), (0, 40)]
    POCKET = [(10, 10), (30, 10), (30, 25), (10, 25)]

    def _drawing(self, reverse_pocket=False, reverse_outer=False):
        keys = [("CUT_-3.0", self.OUTER), ("POCKET_-1.5", self.POCKET)]
        d = "0\nSECTION\n2\nENTITIES\n"
        for layer, pts in keys:
            p = list(pts)
            if (layer.startswith("POCKET") and reverse_pocket) or \
               (layer.startswith("CUT") and reverse_outer):
                p.reverse()
            d += "0\nLWPOLYLINE\n8\n%s\n90\n%d\n70\n1\n" % (layer, len(p))
            for x, y in p:
                d += "10\n%.4f\n20\n%.4f\n" % (x, y)
        d += "0\nENDSEC\n0\nEOF\n"
        return drawing.read_dxf(d)

    def test_outer_and_pocket_get_their_roles(self):
        dr = self._drawing()
        roles = {c.layer: c.role for c in dr.contours}
        self.assertEqual(roles['CUT_-3.0'], 'outer')
        self.assertEqual(roles['POCKET_-1.5'], 'inner')

    def test_role_does_not_depend_on_winding(self):
        """绕向反过来，角色不能变。

        用绕向定"哪边是里面"是一个很自然的错：它在逆时针的图纸上永远对。
        """
        for rp in (False, True):
            for ro in (False, True):
                dr = self._drawing(reverse_pocket=rp, reverse_outer=ro)
                roles = {c.layer: c.role for c in dr.contours}
                self.assertEqual(roles['CUT_-3.0'], 'outer', (rp, ro))
                self.assertEqual(roles['POCKET_-1.5'], 'inner', (rp, ro))

    def test_pocket_path_moves_inward_and_outer_moves_outward(self):
        """真正盯住的东西：同一次生成里，两条轮廓的刀心各朝各的方向。

        外轮廓 0..60 偏 5 mm 后应是 -5..65（变大）；
        型腔 10..30 偏 5 mm 后应是 15..25（变小）。
        两条都变大或都变小，就是方向搞错了。
        """
        dr = self._drawing()
        tool = feeds.Tool(10.0, 3, 'carbide')
        plan = cam.build(dr, 'P', tool, feeds.Machine(), op='rough',
                         ap_per_pass=1.0, offset='tool_radius', plan_ok=False)
        xs, ys = [], []
        for line in plan.text.splitlines():
            if line.startswith('G1 X'):
                words = line.split()
                for w in words:
                    if w.startswith('X'):
                        xs.append(float(w[1:]))
                    elif w.startswith('Y'):
                        ys.append(float(w[1:]))
        self.assertIn(-5.0, [round(v, 3) for v in xs], '外轮廓没有往外偏')
        self.assertIn(65.0, [round(v, 3) for v in xs], '外轮廓没有往外偏')
        self.assertIn(15.0, [round(v, 3) for v in xs], '型腔没有往里偏')
        self.assertIn(25.0, [round(v, 3) for v in xs], '型腔没有往里偏')

    def test_offset_record_says_inward(self):
        dr = self._drawing()
        plan = cam.build(dr, 'P', feeds.Tool(10.0, 3, 'carbide'),
                         feeds.Machine(), op='rough', ap_per_pass=1.0,
                         offset='tool_radius', plan_ok=False)
        recs = {r['layer']: r for r in plan.meta['offsets']}
        self.assertFalse(recs['CUT_-3.0']['inward'])
        self.assertTrue(recs['POCKET_-1.5']['inward'])


class TestSharedStock(unittest.TestCase):
    """生成器和门必须看同一块料。

    各用一块料的具体后果不是"报错"，是"同一条程序两个答案"：
    solve 里放行了，CLI 再检查一次拒绝了，而两次都对。
    """

    def test_default_stock_is_one_implementation(self):
        d = ("0\nSECTION\n2\nENTITIES\n"
             "0\nLWPOLYLINE\n8\nCUT_-3.0\n90\n4\n70\n1\n"
             "10\n0\n20\n0\n10\n60\n20\n0\n10\n60\n20\n40\n10\n0\n20\n40\n"
             "0\nENDSEC\n0\nEOF\n")
        dr = drawing.read_dxf(d)
        tool = feeds.Tool(10.0, 3, 'carbide')
        got = cam.default_stock(dr, tool, 1.5)
        bx0, by0, bx1, by1 = dr.bounds()
        pad = tool.diameter_mm + 10.0
        self.assertAlmostEqual(got[0], bx0 - pad)
        self.assertAlmostEqual(got[2], bx1 + pad)

    def test_solve_and_build_agree_on_stock(self):
        d = ("0\nSECTION\n2\nENTITIES\n"
             "0\nLWPOLYLINE\n8\nCUT_-3.0\n90\n4\n70\n1\n"
             "10\n0\n20\n0\n10\n60\n20\n0\n10\n60\n20\n40\n10\n0\n20\n40\n"
             "0\nENDSEC\n0\nEOF\n")
        dr = drawing.read_dxf(d)
        tool = feeds.Tool(10.0, 3, 'carbide')
        plan, atts = cam.solve(dr, 'P', tool, feeds.Machine(), op='rough',
                               ap_per_pass=1.0)
        self.assertTrue(atts and atts[-1].verdict.ok,
                        'solve 自己放行了，但门在别处拒绝：%s'
                        % (atts[-1].reason if atts else 'no attempts',))
        built = cam.build(dr, 'P', tool, feeds.Machine(), op='rough',
                          ap_per_pass=1.0, plan_ok=False)
        self.assertEqual(tuple(round(v, 6) for v in plan.meta['stock']),
                         tuple(round(v, 6) for v in built.meta['stock']))


class TestPocketClearing(unittest.TestCase):
    """型腔清腔：只沿边界走一圈会在中间留一根柱子。

    那根柱子的麻烦在于它**不报错**：每条指令合法、门放行，
    因为门只看撞不撞夹具、超不超限，它不知道"这块料应该被掏空"。
    """

    def test_pocket_gets_inner_rings(self):
        rings = cam.pocket_clear_paths([(0, 0), (40, 0), (40, 40), (0, 40)], 6.0)
        self.assertGreaterEqual(len(rings), 3, '40x40 的型腔用 D6 刀至少要 3 圈')
        # 由内向外：第一圈的跨度必须比最后一圈小
        def span(rg):
            xs = [q[0] for q in rg]
            return max(xs) - min(xs)
        self.assertLess(span(rings[0]), span(rings[-1]))

    def test_every_ring_stays_inside_the_pocket(self):
        """清腔的每一圈都必须在型腔里面。

        偏置是近似的，近似的东西必须被检查：自交跑出去的那一小段
        会切进型腔壁，而那是不可修的废。
        """
        pocket = [(10, 10), (70, 10), (70, 50), (10, 50)]
        for rg in cam.pocket_clear_paths(pocket, 10.0):
            for q in rg:
                self.assertTrue(cam._point_in_poly(q, pocket),
                                '清腔环跑出型腔了：%r' % (q,))

    def test_ring_never_gets_within_tool_radius_of_the_wall(self):
        """刀心离型腔壁必须至少一个刀半径，否则就是过切。"""
        pocket = [(10, 10), (70, 10), (70, 50), (10, 50)]
        r = 5.0
        for rg in cam.pocket_clear_paths(pocket, 2 * r):
            for q in rg:
                d = cam._poly_dist_to_segments(q, pocket)
                self.assertGreaterEqual(d, r - 1e-6,
                                        '刀心离壁只有 %.4f，小于半径 %.1f' % (d, r))

    def test_narrow_pocket_terminates_and_clears(self):
        """窄条型腔不能因为"短边小"就判成没料。

        第一版拿包围盒短边当停止判据，20x15 的型腔报 0 圈——
        而那条窄槽其实还要清。
        """
        rings = cam.pocket_clear_paths([(10, 10), (30, 10), (30, 25), (10, 25)], 10.0)
        self.assertGreaterEqual(len(rings), 1, '20x15 用 D10 至少还要清一圈')

    def test_clearing_terminates(self):
        """必须收敛。第一版这里跑出 400 圈完全重合的轨迹——

        而每一圈都"合法"，只是刀在同一个地方空转四百次。
        """
        rings = cam.pocket_clear_paths([(0, 0), (200, 0), (200, 200), (0, 200)], 10.0)
        self.assertLess(len(rings), 60, '清腔圈数异常：%d' % len(rings))

    def test_build_clears_inner_contours(self):
        d = ("0\nSECTION\n2\nENTITIES\n"
             "0\nLWPOLYLINE\n8\nCUT_-3.0\n90\n4\n70\n1\n"
             "10\n0\n20\n0\n10\n80\n20\n0\n10\n80\n20\n50\n10\n0\n20\n50\n"
             "0\nLWPOLYLINE\n8\nPOCKET_-1.5\n90\n4\n70\n1\n"
             "10\n15\n20\n15\n10\n65\n20\n15\n10\n65\n20\n40\n10\n15\n20\n40\n"
             "0\nENDSEC\n0\nEOF\n")
        dr = drawing.read_dxf(d)
        plan = cam.build(dr, 'P', feeds.Tool(10.0, 3, 'carbide'), feeds.Machine(),
                         op='rough', ap_per_pass=1.5, offset='tool_radius',
                         ramp_deg=0, plan_ok=False)
        pk = plan.meta['pockets']
        self.assertEqual(len(pk), 1)
        self.assertGreaterEqual(pk[0]['clear_rings'], 1,
                                '型腔没被清腔：%r' % (pk[0],))

    def test_outer_contour_is_not_treated_as_a_pocket(self):
        """外轮廓没有"里面"要去，不能给它加环。"""
        d = ("0\nSECTION\n2\nENTITIES\n"
             "0\nLWPOLYLINE\n8\nCUT_-3.0\n90\n4\n70\n1\n"
             "10\n0\n20\n0\n10\n60\n20\n0\n10\n60\n20\n40\n10\n0\n20\n40\n"
             "0\nENDSEC\n0\nEOF\n")
        dr = drawing.read_dxf(d)
        plan = cam.build(dr, 'P', feeds.Tool(10.0, 3, 'carbide'), feeds.Machine(),
                         op='rough', ap_per_pass=1.5, offset='tool_radius',
                         plan_ok=False)
        self.assertEqual(plan.meta['pockets'], [])


class TestEllipticalArcs(unittest.TestCase):
    """非等比缩放的块里，圆变成真椭圆。

    第一版在这里报"未实现"，理由是"非等比缩放的块在机械图纸里极少"。
    那句话没错，错的是它的用法：它让一个能算出来的东西变成了读不懂，
    于是整条轮廓从图纸上消失，报告里只说一句"这个实体未实现"。
    """

    def _ellipse_dxf(self):
        b = _nl((0, "SECTION"), (2, "BLOCKS"))
        b += _nl((0, "BLOCK"), (8, "0"), (2, "CIRC"), (70, 0), (10, 0), (20, 0))
        b += _nl((0, "CIRCLE"), (8, "HOLE_-3.0"), (10, 0), (20, 0), (40, 5))
        b += _nl((0, "ENDBLK"), (0, "ENDSEC"))
        e = _nl((0, "SECTION"), (2, "ENTITIES"))
        e += _nl((0, "INSERT"), (8, "HOLE_-3.0"), (2, "CIRC"),
                 (10, 0), (20, 0), (41, 2), (42, 1))
        e += _nl((0, "ENDSEC"), (0, "EOF"))
        return b + e

    def test_nonuniform_scaled_circle_becomes_an_ellipse(self):
        dr = drawing.read_dxf(self._ellipse_dxf())
        self.assertEqual(dr.unsupported, [], '椭圆被报成未实现了：%s'
                         % (dr.unsupported,))
        self.assertEqual(len(dr.contours), 1)
        xs = [q[0] for q in dr.contours[0].pts]
        ys = [q[1] for q in dr.contours[0].pts]
        self.assertAlmostEqual(max(xs) - min(xs), 20.0, delta=0.2)   # 2 * 5 * 2
        self.assertAlmostEqual(max(ys) - min(ys), 10.0, delta=0.2)   # 2 * 5 * 1

    def test_ellipse_area_matches_pi_ab(self):
        """面积必须等于 pi*a*b，而不是 pi*r^2。

        第一版把 INSERT 的 42 组码当成了 LWPOLYLINE 的 bulge，
        于是 y 方向缩放被吃掉，椭圆**静默地没被压扁**：
        面积从 157 变成 314，没有一行报错。数值错比报错难查得多。
        """
        import math as _m
        dr = drawing.read_dxf(self._ellipse_dxf())
        self.assertAlmostEqual(abs(dr.contours[0].area()), _m.pi * 10 * 5,
                               delta=3.0)

    def test_bulge_and_insert_scale_do_not_collide(self):
        """组码 42 在 LWPOLYLINE 里是 bulge，在 INSERT 里是 y 缩放。

        同一个数字两个意思，按实体类型分派。混了就是上面那个静默的错。
        """
        # LWPOLYLINE 的 42 = 半圆凸度：两点之间画一个半圆
        d = ("0\nSECTION\n2\nENTITIES\n"
             "0\nLWPOLYLINE\n8\nCUT_-1.0\n90\n2\n70\n1\n"
             "10\n0\n20\n0\n42\n1\n10\n20\n20\n0\n"
             "0\nENDSEC\n0\nEOF\n")
        dr = drawing.read_dxf(d)
        self.assertEqual(len(dr.contours), 1)
        self.assertGreater(len(dr.contours[0].pts), 4,
                           'bulge 没被当成弧展开：只有直边')
        ys = [q[1] for q in dr.contours[0].pts]
        self.assertGreater(max(abs(v) for v in ys), 5.0, '半圆没鼓起来')


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
