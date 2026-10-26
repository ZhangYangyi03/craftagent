# -*- coding: utf-8 -*-
"""bench_cnc.py -- 安全门的两个数字，以及每个数字是怎么得到的。

    admitted / legal programs     一个"什么都拒绝"的门在第二个数上满分，
                                  第一个数上零分。所以两个必须一起报。
    caught   / planted breaks     每个故障都是手工植入的，落点记在案

再加**消融对照**，这是唯一让第二个数有意义的部分：每一条规则，
把同一批植入故障在**关掉那条规则**之后重跑一遍，要求拦截率塌掉。
一个门在关掉 joint_limit 之后还能拦住行程越界，那是碰巧拦住的，
它报出来的数就不属于那条规则。

这个消融对照是把 robot-spine 的 bench_gate.py 换了个对象——那边管关节，
这边管 G 代码行。
"""
import json
import os
import sys

from craftagent import gate_cnc, gcode, plan, plant

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------- 合法程序
def legal_programs(n=60, seed=11, fixture=None):
    """一组在机床行程、进给、主轴、半径、**以及夹具**上全部合规的程序。

    fixture 这个参数是加撞刀规则之后补的，补的理由本身值得写下来：
    这个生成器原来只按机床行程 x>=-10 来收圆弧，因为那时世界里没有压板。
    撞刀规则一开，18/60 条"合法"程序被 clamp_crash 拦下——而它们是真的撞：
    圆弧甩到 X=-10、Z=-0.7，正好落在压板 clamp0 里。
    所以那 18 条不是门太严，是**生成器在造会撞的程序还叫它合法**。
    按夹具收边界的是生成器，不是把压板挪开——挪压板就是让坏消息消失，
    而不是让程序变对。
    """
    import random
    rng = random.Random(seed)
    out = []
    for k in range(n):
        rows = rng.randint(2, 5)
        cols = rng.randint(2, 5)
        # 走刀范围必须真的落在毛坯里。原来 cols 到 5、pitch 到 20 会造出
        # 80 mm 宽的零件，而默认毛坯只有 60 mm——那 10 条被 clamp_crash
        # 拦下的"合法"程序，撞的其实是毛坯外面那块压板。
        # 收敛的是生成器，不是毛坯：毛坯是车间里给定的那块料。
        if fixture is not None:
            st = fixture.stock
            span_x = st.x1 - st.x0 - 4.0 * fixture.tool.radius
            span_y = st.y1 - st.y0 - 4.0 * fixture.tool.radius
            x_lo, y_lo = st.x0 + 2.0 + fixture.tool.radius, st.y0 + 2.0 + fixture.tool.radius
        else:
            span_x, span_y = 60.0, 60.0
            x_lo, y_lo = 2.0, 2.0
        pitch = rng.uniform(6.0, 20.0)
        while cols > 2 and (cols - 1) * pitch > span_x:
            cols -= 1
        while rows > 2 and (rows - 1) * pitch > span_y:
            rows -= 1
        if cols == 2:
            pitch = min(pitch, span_x)
        # 进给上限按门的质量规格取：这台机床在 8.5 um 规格下，
        # F750 已经临界（max 9.69），F900 明确超差（max 12.60）。
        # 生成器若继续用 F900，"合法程序"里就混进了不合格品，
        # 放行率会掉——那不是门太严，是生成器在造废品还叫它合格。
        feed = rng.choice([300.0, 400.0, 500.0, 600.0])
        depth = -rng.uniform(0.2, 1.5)
        rpm = rng.choice([4000, 6000, 8000, 10000])
        L = ['G21 G90', 'G0 Z5.0 F2000', 'M3 S%d' % rpm,
             'G0 X%.3f Y%.3f' % (x_lo, y_lo), 'G1 Z%.2f F150' % depth]
        y = y_lo
        for i in range(rows):
            x = x_lo + (cols - 1) * pitch if i % 2 == 0 else x_lo
            L.append('G1 X%.3f Y%.3f F%.0f' % (x, y, feed))
            if i < rows - 1:
                y += pitch
                L.append('G1 Y%.3f F%.0f' % (y, feed))
        # 一处半径合规的半圆。方向由**净空**决定，不由习惯决定：
        # 往 -X 甩没有地方就改往 +X 甩，而不是把半径压小到仍然飞出毛坯。
        r = rng.uniform(1.0, 8.0)
        x_cur = x_lo + (cols - 1) * pitch if (rows - 1) % 2 == 0 else x_lo
        clear_neg = x_cur - (x_lo + fixture.tool.radius if fixture else 2.0)
        clear_pos = (fixture.stock.x1 - 2.0 - fixture.tool.radius - x_cur
                     if fixture else 60.0 - x_cur)
        if clear_neg >= clear_pos:
            r = min(r, clear_neg / 2.0)
            x_end = x_cur - 2.0 * r
            i_off = -r
        else:
            r = min(r, clear_pos / 2.0)
            x_end = x_cur + 2.0 * r
            i_off = r
        r = max(0.6, r)
        if i_off < 0:
            x_end = x_cur - 2.0 * r
        else:
            x_end = x_cur + 2.0 * r
        L.append('G2 X%.3f Y%.3f I%.3f J0' % (x_end, y, i_off))
        L.append('G0 Z5.0')
        L.append('M5')
        out.append('\n'.join(L) + '\n')
    return out


# ---------------------------------------------------------------- 植入故障
def plant_faults(legal, rng_seed=3):
    """每一个故障只破坏一条规则，返回 (rule, text, note).

    落点是刻意的：如果一处改动同时破坏了三条规则，那这条故障就证明不了
    任何一条规则在工作，所以每条故障都只碰一个字段。
    """
    import random
    rng = random.Random(rng_seed)
    out = []
    for text in legal[:40]:
        lines = text.rstrip('\n').split('\n')
        cut = [i for i, l in enumerate(lines) if l.startswith('G1 X')]
        arc = [i for i, l in enumerate(lines) if l.startswith('G2')]
        # 1 行程：在程序末尾追加一处机床行程之外的快速定位。
        # 不在中途改坐标——那会把后面每一段的位置都改掉，
        # 于是圆弧的弦长也跟着变，一条"行程故障"会顺手破坏几何规则，
        # 消融对照就再也说不清是谁拦的。追加一行只碰行程这一个字段。
        L = list(lines[:-1])
        L.append('G0 X900.000 Y0 F2000')
        L.append('M5')
        out.append(('axis_limit', '\n'.join(L) + '\n', 'rapid to X=900 mm, outside travel'))
        # 2 进给：把一个 F 抬到额定之上
        if cut:
            i = rng.choice(cut)
            L = list(lines)
            parts = L[i].split()
            L[i] = ' '.join('F99999' if p.startswith('F') else p for p in parts)
            out.append(('feed_limit', '\n'.join(L) + '\n', 'F raised to 99999 mm/min'))
        # 3 半径：把圆弧半径压到刀具走不出来
        if arc:
            i = rng.choice(arc)
            L = list(lines)
            # 起点是上一行（G1 Y..）的终点，不是被改的这一行里的 X。
            # 拿错起点会让这条弧的半径对不上弦长，于是它同时违反 gcode_syntax，
            # 消融时就分不清是谁拦的——这正是"故障注入必须只破坏一条规则"的意思。
            x0, y0 = 0.0, 0.0
            for j in range(i - 1, -1, -1):
                w = dict((q[0], q[1:]) for q in L[j].split() if len(q) > 1)
                if w:
                    if 'X' in w:
                        try: x0 = float(w['X'])
                        except ValueError: pass
                    if 'Y' in w:
                        try: y0 = float(w['Y'])
                        except ValueError: pass
                    break
            # 一段半径 0.1 的合法小圆（几何自洽，只违反"刀具能走的最小半径"）：
            # 从 (x0,y0) 画半圆到 (x0+0.2, y0)，圆心 (x0+0.1, y0)。
            # 这样它只碰半径字段，不会顺手把弦长弄错。
            L[i] = 'G2 X%.3f Y%.3f I0.100 J0' % (x0 + 0.2, y0)
            out.append(('radius_limit', '\n'.join(L) + '\n',
                        'arc radius 0.1 mm, geometrically consistent'))
        # 4 主轴：把 S 字删掉
        L = [l for l in lines if not l.startswith('M3')]
        out.append(('spindle', '\n'.join(L) + '\n', 'M3 S removed'))
        # 5 语法：把圆弧改成几何上不可能的
        if arc:
            i = rng.choice(arc)
            L = list(lines)
            L[i] = 'G2 X50 Y50 I0 J1'
            out.append(('gcode_syntax', '\n'.join(L) + '\n', 'arc end off its circle'))
        # 6 加速度：一段极短行程要求极高进给
        L = list(lines)
        L.insert(4, 'G1 X0.05 F6000')
        out.append(('accel_limit', '\n'.join(L) + '\n', '0.05 mm move at F6000'))
        # 7 质量预测：把全部切削进给抬到额定之内、但质量做不到的档位。
        # 这一条不能碰别的字段——进给要在 feed_max(6000) 之内、行程不动、
        # 半径不动、主轴不动，只是"每个参数都合规、结果仍报废"。
        # 只替换 F 字，坐标一个都不动——动了坐标就会挪动圆弧的弦长，
        # 那条"质量故障"会顺手变成一条几何错误，消融就分不清是谁拦的。
        L = []
        for l in lines:
            parts = l.split()
            if parts and parts[0] in ('G1', 'G2', 'G3'):
                parts = ['F1500' if q.startswith('F') else q for q in parts]
                if not any(q.startswith('F') for q in parts):
                    parts.append('F1500')
            L.append(' '.join(parts))
        out.append(('quality_predict', '\n'.join(L) + '\n',
                    'all cutting feeds at F1500: within rating, still over spec'))
    return out


def run(n_legal=60, verbose=True):
    from craftagent import collide as collide_mod
    lim = gate_cnc.Limits()
    rng_seed = 3
    fx = collide_mod.Fixture()
    legal = legal_programs(n_legal, fixture=fx)
    gate = gate_cnc.Gate(lim, fixture=fx)
    # 撞刀那四条要有"必须放行"的反面, 否则误拒仍然看不见。
    # 这一句必须放在统计**之前**：上一版放在循环之后，于是分母从 60 变成
    # 64 而分子还是 60，"legal 60/64" 看起来像 4 条被误拒，其实那 4 条
    # 根本没被测——一个分母算不上错、但会读错的报表。
    legal = legal + [t for _, t in collide_mod.legal_programs(fx)]
    # 1. 合法程序：必须全部放行
    admitted, refused = 0, []
    for t in legal:
        v = gate.check_text(t)
        if v.ok:
            admitted += 1
        else:
            refused.append(v.detail)
    faults = plant_faults(legal, rng_seed)
    # 撞刀四条的植入故障单独来: 它们不该靠"把某个字段改坏"来造,
    # 而是由 collide.planted_faults 直接造出会撞的几何。
    # 混进 plant_faults 会让"每条故障只碰一个字段"这句话变成假的——
    # 撞刀故障碰的不是字段, 是工件与夹具的几何。
    from craftagent import collide as collide_mod
    for name, text, want in collide_mod.planted_faults(fx):
        faults.append((want, text, 'collision: %s' % name))
    # 2. 植入故障：必须全部拦住，且由**指定的那条规则**拦住
    caught, by_right_rule, missed, wrong_rule = 0, 0, [], []
    for rule, text, note in faults:
        v = gate.check_text(text)
        if v.ok:
            missed.append((rule, note))
        else:
            caught += 1
            if rule in v.reasons:
                by_right_rule += 1
            else:
                wrong_rule.append((rule, note, v.reasons))
    # 3. 消融对照：关掉一条规则，对应的拦截必须塌
    ablate = {}
    for rule in gate_cnc.ALL_RULES:
        if rule == 'gcode_syntax':
            continue                    # 语法在解析器里，关不掉也不该关
        g2 = gate_cnc.Gate(lim, fixture=fx,
                           rules=[r for r in gate_cnc.ALL_RULES if r != rule])
        sub = [(rr, t, nt) for rr, t, nt in faults if rr == rule]
        still = 0
        # 关掉一条规则之后还拦得住, 必须记下**是谁**接住的。
        # accel_limit 那 40 条就是例子: 关掉 accel 后它们仍被
        # quality_predict 拦住(0.05 mm 的短段在 F6000 下预测圆度超差),
        # 于是"仍拦住 40/40"看起来像消融失败, 实际是另一条规则在报同一个物理。
        # 不写下来, 这一行就会被读成"accel_limit 这条规则没用"。
        takers = {}
        for rr, t, nt in sub:
            v = g2.check_text(t)
            if not v.ok:
                still += 1
                for other in v.reasons:
                    takers[other] = takers.get(other, 0) + 1
        ablate[rule] = {'planted': len(sub), 'still_caught_with_rule_off': still,
                        'caught_by_with_rule_off': takers}
    res = {
        'legal': {'total': len(legal), 'admitted': admitted,
                  'false_refusal_rate': round(len(refused) / max(1, len(legal)), 6),
                  'examples': refused[:3]},
        'planted': {'total': len(faults), 'caught': caught,
                    'caught_by_intended_rule': by_right_rule,
                    'missed': missed[:5], 'wrong_rule': wrong_rule[:5]},
        'mutation_control': ablate,
    }
    if verbose:
        print("legal   %d / %d admitted  (false refusals %d)"
              % (admitted, len(legal), len(refused)))
        print("planted %d / %d caught, %d by the intended rule"
              % (caught, len(faults), by_right_rule))
        for rule, row in sorted(ablate.items()):
            takers = row.get('caught_by_with_rule_off') or {}
            tail = ''
            if takers:
                tail = '  <- by %s' % ', '.join('%s %d' % (k, v)
                                                for k, v in sorted(takers.items()))
            print("  rule off %-14s still caught %d / %d%s"
                  % (rule, row['still_caught_with_rule_off'], row['planted'], tail))
    return res


if __name__ == '__main__':
    r = run()
    os.makedirs(os.path.join(ROOT, 'out'), exist_ok=True)
    with open(os.path.join(ROOT, 'out', 'bench_cnc.json'), 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print('written out/bench_cnc.json')
