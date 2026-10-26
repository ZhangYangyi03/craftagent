# -*- coding: utf-8 -*-
"""pair.py -- C 内核与 Python 规划器的逐段对拍。

两份独立实现同意，才叫"规划结果是对的"。同一份代码跑两遍永远同意，
那测的是确定性，不是正确性——这一点值得写在这里，因为它是这个文件
唯一的存在理由。

对拍的是**规划表**（每段的 v_entry / v_peak / v_exit / t_s），不是
最终位置。位置由插补器从规划表推出来，规划表一致而插补器还能错，
那是另一个 bug，由 test 里的位置连续性检查负责。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CNCRT = os.path.join(ROOT, 'cncrt')

from craftagent import gcode, plan


def _write_cases(path, segs, params):
    def row(s):
        c = s.centre or (0.0, 0.0)
        vals = [s.line, 1 if s.kind == 'arc' else 0,
                s.start[0], s.start[1], s.start[2],
                s.end[0], s.end[1], s.end[2],
                s.feed / 60.0, s.length,
                s.t_in[0], s.t_in[1], s.t_in[2],
                s.t_out[0], s.t_out[1], s.t_out[2],
                s.radius or 0.0, 1 if s.ccw else 0, c[0], c[1]]
        return ' '.join('%d' % v if i in (0, 1, 17) else '%.9f' % v
                        for i, v in enumerate(vals))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('%.9f %.9f %.9f 0.0 0.0\n'
                % (params['a_max_mm_s2'], params['jerk_dev_mm'], params['v_max_mm_s']))
        for s in segs:
            f.write(row(s) + '\n')


def compare(text, plan_params=None, dt=0.001, name='<pair>'):
    """同一段程序，C 与 Python 各算一遍，逐段比。返回对比结果。"""
    exe = os.path.join(CNCRT, 'cncrt.exe')
    if not os.path.exists(exe):
        return {'ok': None, 'reason': 'cncrt.exe not built; run bench/build_cncrt.py first'}
    p = dict(plan.DEFAULT)
    p.update(plan_params or {})
    segs, meta = gcode.parse(text, name=name)
    cases = os.path.join(CNCRT, '_pair_cases.txt')
    _write_cases(cases, segs, p)
    r = subprocess.run([exe, cases, str(dt)], capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=120)
    if r.returncode != 0:
        return {'ok': False, 'reason': 'cncrt exited %d: %s'
                % (r.returncode, r.stderr[-300:])}
    C = json.loads(r.stdout)
    P = plan.plan(segs, p)
    cr, pr = C['rows'], P['rows']
    if len(cr) != len(pr):
        return {'ok': False, 'reason': 'segment count %d vs %d' % (len(cr), len(pr))}
    dv = dt_ = 0.0
    worst_v = worst_t = None
    for a, b in zip(cr, pr):
        d = max(abs(a['v_entry'] - b['v_entry']), abs(a['v_peak'] - b['v_peak']),
                abs(a['v_exit'] - b['v_exit']))
        if d > dv:
            dv, worst_v = d, {'line': a['line'], 'c_peak': a['v_peak'],
                              'py_peak': b['v_peak']}
        e = abs(a['t_s'] - b['t_s'])
        if e > dt_:
            dt_, worst_t = e, {'line': a['line'], 'c': a['t_s'], 'py': b['t_s']}
    return {'ok': dv < 1e-3 and dt_ < 1e-4,
            'max_dv_mm_s': dv, 'max_dt_s': dt_,
            'worst_v': worst_v, 'worst_t': worst_t,
            't_total_c': C['t_plan_s'], 't_total_py': P['t_total_s'],
            'c_perf': {'plan_us': C['plan_us'], 'ticks': C['ticks'],
                       'tick_us_mean': C['tick_us_mean'],
                       'tick_us_max': C['tick_us_max']},
            'segments': len(cr)}


def standard_programs():
    """几段形状不同的程序：全是直线的、含圆弧的、极短段的、极长段的。"""
    zig = ['G21 G90', 'G0 Z5 F2000', 'M3 S8000', 'G1 X0 Y0 F150', 'G1 Z-1 F150']
    for i in range(8):
        x = 30.0 if i % 2 == 0 else 0.0
        zig.append('G1 X%.3f Y%.3f F900' % (x, 12.0 * i))
        zig.append('G1 X%.3f Y%.3f F900' % (x, 12.0 * (i + 1)))
    zig += ['G2 X30 Y96 I15 J0', 'G3 X0 Y96 I-15 J0']
    tiny = ['G21 G90', 'M3 S8000', 'G1 X0 Y0 F500']
    for k in range(10):
        tiny.append('G1 X%.4f F500' % (0.05 * (k + 1)))
    longr = ['G21 G90', 'M3 S8000', 'G1 X0 Y0 F300', 'G1 X400 F1200', 'G1 Y300']
    return {'zigzag_arcs': '\n'.join(zig) + '\n',
            'tiny_segments': '\n'.join(tiny) + '\n',
            'long_rapid': '\n'.join(longr) + '\n'}


def run_all(verbose=True):
    out = {}
    for name, text in standard_programs().items():
        out[name] = compare(text, name=name)
    out['all_ok'] = all(v.get('ok') for v in out.values() if isinstance(v, dict))
    if verbose:
        for name, v in out.items():
            if not isinstance(v, dict) or 'segments' not in v:
                continue
            print('%-16s %-4s seg=%3d  max|dv|=%.2e mm/s  max|dt|=%.2e s  C tick %.3f us'
                  % (name, 'OK' if v['ok'] else 'FAIL', v['segments'],
                     v['max_dv_mm_s'], v['max_dt_s'], v['c_perf']['tick_us_mean']))
    return out


if __name__ == '__main__':
    r = run_all()
    print('all_ok:', r['all_ok'])
    sys.exit(0 if r['all_ok'] else 1)
