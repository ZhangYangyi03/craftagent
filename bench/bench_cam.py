"""图纸 -> 参数 -> G 代码 -> 门 这条链的验收：要报数，不要报"能跑"。

这个文件回答四个问题，每一个都有分母：

  1. 手册查出来的参数，有多少比例能让门放行？（不带回退的一次性命中率）
  2. 关门不放行的那些，回退几轮能放行？（闭环的代价）
  3. 生成的程序里，有没有一刀超出毛坯？（生成器自己的错，不该让门来发现）
  4. 图纸里读不懂的东西，有没有被**说出来**？（静默跳过就是假装图纸干净）

第 1 与第 2 必须分开报：只报第 2，一个"什么都慢慢来"的策略能拿满分，
而它把材料性能浪费掉了；只报第 1，闭环的价值看不见。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from craftagent import gate_cnc, drawing as drawing_mod, cam as cam_mod, feeds as feeds_mod, collide


def _demo_dxf():
    """造一张有外轮廓、型腔、圆孔、一条开链、一个无深度图层的图纸。

    要素是刻意的：后两者是**必须被报出来**的脏东西。
    一张处处干净的测试图纸证明不了"读不懂的会被说出来"。
    """
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

    # 块引用：真实图纸里重复特征（型腔、孔组）一定是块 + 多次引用。
    # 不展开就等于把件少做一大半，而图纸仍然"读完了"。
    blocks = ("0\nSECTION\n2\nBLOCKS\n"
              "0\nBLOCK\n8\n0\n2\nBOLT\n70\n0\n10\n0\n20\n0\n"
              + poly("HOLE_-4.0", [(0, 0), (6, 0), (6, 6), (0, 6)])
              + "0\nENDBLK\n0\nENDSEC\n")

    d = d_blocks = blocks + "0\nSECTION\n2\nENTITIES\n"
    d += poly("CUT_-3.0", [(0, 0), (60, 0), (60, 40), (0, 40)])
    d += poly("POCKET_-1.5", [(10, 10), (30, 10), (30, 25), (10, 25)])
    d += circ("HOLE_-6.0", 45, 20, 4)
    d += line("CUT_-3.0", 0, 45, 20, 45)
    d += poly("NOTES", [(70, 0), (80, 0), (80, 10), (70, 10)])
    # 两处真实图纸里最常见的形态：一堆 LINE 首尾相接，以及一条伸出去的边
    # 造成的分叉。前者不该被报成"没东西可切"，后者必须被**数出来**。
    for e in ((0, 0, 15, 0), (15, 0, 15, 8), (15, 8, 0, 8), (0, 8, 0, 0)):
        d += line("RECT_-2.0", *e)
    d += ("0\nINSERT\n8\nHOLE_-4.0\n2\nBOLT\n10\n5\n20\n32\n"
          "0\nINSERT\n8\nHOLE_-4.0\n2\nBOLT\n10\n50\n20\n32\n")
    d += "0\nENDSEC\n0\nEOF\n"
    return d


def run(out=None, verbose=True):
    dr = drawing_mod.read_dxf(_demo_dxf())
    stock = (-10.0, -10.0, 70.0, 50.0, 0.0, 28.0)
    limits = gate_cnc.Limits()
    fx = collide.Fixture()
    gate = gate_cnc.Gate(limits, fixture=fx)

    # ---- 1. 一次性命中率：五组材料 x 两道工序 x 三把刀
    combos = []
    for mat in ('P', 'M', 'K', 'N', 'S'):
        for op in ('rough', 'finish', 'slot'):
            for dia, fl in ((3.0, 3), (6.0, 3), (12.0, 4)):
                combos.append((mat, op, feeds_mod.Tool(dia, fl, 'carbide')))
    one_shot_ok, one_shot_refused = 0, []
    for mat, op, tool in combos:
        plan, atts = cam_mod.solve(dr, mat, tool, feeds_mod.Machine(), stock=stock,
                                   op=op, ap_per_pass=1.0, max_tries=1)
        if atts and atts[0].verdict.ok:
            one_shot_ok += 1
        else:
            one_shot_refused.append({'material': mat, 'op': op, 'tool': tool.name,
                                     'reasons': atts[0].reason if atts else 'no plan'})

    # ---- 2. 闭环代价：对被拒的那些，回退几轮能过
    rounds_hist, closed_ok, closed_fail = {}, 0, []
    for row in one_shot_refused:
        mat, op = row['material'], row['op']
        tool = next(t for t in (feeds_mod.Tool(d, f, 'carbide')
                                for d, f in ((3.0, 3), (6.0, 3), (12.0, 4)))
                    if t.name == row['tool'])
        plan, atts = cam_mod.solve(dr, mat, tool, feeds_mod.Machine(), stock=stock,
                                   op=op, ap_per_pass=1.0, max_tries=8)
        n = len(atts)
        if atts and atts[-1].verdict.ok:
            closed_ok += 1
            rounds_hist[n] = rounds_hist.get(n, 0) + 1
        else:
            closed_fail.append({'material': mat, 'op': op, 'tool': row['tool'],
                                'tries': n, 'reasons': atts[-1].reason if atts else ''})

    # ---- 3. 生成器自检：每一刀都在毛坯之内、都到得了图要求的深度
    gen_problems = []
    for mat, op, tool in combos[:15]:
        plan, atts = cam_mod.solve(dr, mat, tool, feeds_mod.Machine(), stock=stock,
                                   op=op, ap_per_pass=1.0)
        try:
            cam_mod.self_check(plan, dr, stock)
        except cam_mod.CamError as e:
            gen_problems.append({'material': mat, 'op': op, 'tool': tool.name,
                                 'why': str(e)})

    # ---- 4. 读不懂的东西要说出来
    dr2 = drawing_mod.read_dxf(_demo_dxf())
    said = {
        'open_chain_reported': bool(dr2.stitch and dr2.stitch.open_chains),
        'unknown_depth_reported': len(dr2.unknown_depth) > 0,
        'unknown_depth_layers': sorted({l for l, _ in dr2.unknown_depth}),
        'contours_found': len(dr2.contours),
        'stitched_cycles': sum(v['cycles'] for v in dr2.stitch.layers.values())
                           if dr2.stitch else 0,
        'stitched_layers': dr2.stitch.as_dict()['layers'] if dr2.stitch else {},
    }

    # ---- 5. 刀心偏置：偏对了件才是图纸尺寸；偏错是沉默的（程序全合法）
    tool10 = feeds_mod.Tool(10.0, 3, 'carbide')
    off_check = {}
    for mode, allow in (('none', 0.0), ('tool_radius', 0.0),
                        ('tool_radius_minus_allow', 0.3)):
        pl = cam_mod.build(dr, 'P', tool10, feeds_mod.Machine(), stock=stock,
                           op='rough', ap_per_pass=1.5, offset=mode,
                           finish_allow=allow, plan_ok=False)
        rec = pl.meta['offsets'][0]
        off_check[mode] = {'offset_mm': round(rec['dist'], 4),
                           'kind': rec['kind'],
                           'clamped_vertices': len(rec['miter_clamped_vertices'])}

    res = {
        'drawing': dr2.summary(),
        'tool_offset': off_check,
        'one_shot': {'total': len(combos), 'admitted': one_shot_ok,
                     'admitted_rate': round(one_shot_ok / max(1, len(combos)), 4),
                     'refused': one_shot_refused},
        'closed_loop': {'refused_to_retry': len(one_shot_refused),
                        'recovered': closed_ok,
                        'recovered_rate': round(closed_ok / max(1, len(one_shot_refused)), 4),
                        'rounds_histogram': rounds_hist,
                        'still_failing': closed_fail},
        'generator_self_check': {'checked': 15, 'problems': gen_problems},
        'unsupported_said_out_loud': said,
    }
    if verbose:
        print("图纸：%d 条闭合轮廓；未采用实体 %d 个；无深度图层 %s"
              % (len(dr2.contours), len(dr2.unsupported), said['unknown_depth_layers']))
        print("一次性命中：%d / %d 放行（%.1f%%）"
              % (one_shot_ok, len(combos), 100.0 * one_shot_ok / len(combos)))
        print("闭环回退：%d / %d 被拒的最终放行；轮数分布 %s"
              % (closed_ok, len(one_shot_refused), rounds_hist))
        print("生成器自检：%d 条程序，问题 %d 个"
              % (15, len(gen_problems)))
        print("读不懂的说明了：开链 %s；无深度 %s"
              % (said['open_chain_reported'], said['unknown_depth_reported']))
        print("刀心偏置：%s"
              % "；".join("%s=%.2fmm" % (k, v['offset_mm'])
                          for k, v in off_check.items()))
        print("拼环：从 LINE 拼出 %d 个闭合轮廓（%s）"
              % (said['stitched_cycles'],
                 ", ".join("%s=%d" % (k, v['cycles'])
                           for k, v in said['stitched_layers'].items()) or "无"))
    if out:
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        if verbose:
            print("明细写入 %s" % out)
    return res


if __name__ == '__main__':
    run(out=os.path.join(ROOT, 'out', 'cam_verify.json'))
