# -*- coding: utf-8 -*-
"""gcode.py -- 把一条 G 代码程序读成段（segment），并且每个错误都带行号。

这个解析器只认这个仓要量的东西：三轴、G0/G1/G2/G3、绝对/增量、毫米/英寸、
进给 F、主轴 S 与 M3/M4/M5。不认识的东西是**带行号的拒绝**，不是静默跳过——
计划器看不见的程序，门也拦不住，那"它跑起来了"就什么也不意味着。

每一段都带着源行号，所以下游任何一次拒绝都能点名是第几行造成的。
这就是它是个模块而不是一行正则的全部理由。
"""
import math

LETTERS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

GROUPS = {
    'motion': (0, 1, 2, 3),
    'plane': (17, 18, 19),
    'units': (20, 21),
    'distance': (90, 91),
}
SUPPORTED_G = {0, 1, 2, 3, 17, 18, 19, 20, 21, 90, 91, 94}
SUPPORTED_M = {3, 4, 5, 8, 9}


class GcodeError(Exception):
    def __init__(self, msg, line=None, text=None):
        self.line, self.text = line, text
        where = '' if line is None else 'line %s: ' % line
        tail = '' if not text else '  [%s]' % text.strip()
        Exception.__init__(self, where + msg + tail)


def strip_comments(code):
    """Drop a parenthesised comment and a ';' tail."""
    out, depth = [], 0
    for ch in code:
        if ch == '(':
            depth += 1
            continue
        if ch == ')':
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if ch == ';':
            break
        out.append(ch)
    return ''.join(out)


def words(code):
    """'G1X10.5F300' -> [('G',1.0),('X',10.5),('F',300.0)]."""
    out, i, n = [], 0, len(code)
    while i < n:
        ch = code[i]
        if ch in ' \t\r\n':
            i += 1
            continue
        if ch.upper() not in LETTERS:
            raise GcodeError('unexpected character %r' % ch)
        letter = ch.upper()
        i += 1
        j = i
        if j < n and code[j] in '+-':
            j += 1
        while j < n and (code[j].isdigit() or code[j] == '.'):
            j += 1
        if j == i or (j == i + 1 and code[i] in '+-'):
            raise GcodeError('word %r has no number' % letter)
        try:
            val = float(code[i:j])
        except ValueError:
            raise GcodeError('word %r has an unparseable number %r' % (letter, code[i:j]))
        out.append((letter, val))
        i = j
    return out


class Seg:
    """One move. A straight move and an arc are both segments: the planner only
    asks for length, endpoints, feed and entry/exit tangent, and an arc answers
    those exactly instead of being chopped into chords nobody counted."""
    __slots__ = ('line', 'text', 'kind', 'start', 'end', 'feed', 'length',
                 'centre', 'ccw', 'radius', 't_in', 't_out', 's', 'spindle', 'rpm')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return '<Seg L%s %s len=%.3f f=%.0f>' % (self.line, self.kind, self.length, self.feed)


def _unit(d):
    n = math.sqrt(sum(v * v for v in d))
    if n <= 0:
        return None
    return tuple(v / n for v in d)


def parse(text, name='<program>'):
    """Return (segments, meta). Raises GcodeError carrying the offending line number."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', 'replace')
    pos = [0.0, 0.0, 0.0]
    feed = None
    spindle = None
    motion = 1
    rpm = None
    plane = 17
    units = 21
    absolute = True
    segs, unknown, zero_moves = [], [], []
    for ln, raw in enumerate(text.splitlines(), 1):
        code = strip_comments(raw)
        if not code.strip():
            continue
        try:
            ws = words(code)
        except GcodeError as e:
            raise GcodeError(str(e), ln, raw)
        gs, ms, vals = [], [], {}
        for letter, v in ws:
            if letter in ('G', 'M'):
                if abs(v - round(v)) > 1e-9:
                    raise GcodeError('%s%s is not a whole number' % (letter, v), ln, raw)
                (gs if letter == 'G' else ms).append(int(round(v)))
            else:
                vals[letter] = v
        for grp, members in GROUPS.items():
            got = [g for g in gs if g in members]
            if len(got) > 1:
                raise GcodeError('two %s codes on one line: %s' % (grp, got), ln, raw)
            if got:
                if grp == 'motion':
                    motion = got[0]
                elif grp == 'plane':
                    plane = got[0]
                elif grp == 'units':
                    units = got[0]
                else:
                    absolute = (got[0] == 90)
        for g in gs:
            if g not in SUPPORTED_G:
                unknown.append((ln, 'G%d' % g))
        for m in ms:
            if m not in SUPPORTED_M:
                unknown.append((ln, 'M%d' % m))
            elif m == 3:
                spindle = 'cw'
            elif m == 4:
                spindle = 'ccw'
            elif m == 5:
                spindle = 'off'
        if 'F' in vals:
            feed = vals['F'] if units == 21 else vals['F'] * 25.4
            if feed <= 0:
                raise GcodeError('feed F%s is not positive' % vals['F'], ln, raw)
        if 'S' in vals:
            rpm = vals['S']
            if spindle is None:
                spindle = 'cw'
        if plane != 17:
            raise GcodeError('only the XY plane (G17) is supported, not G%d' % plane, ln, raw)
        want = any(k in vals for k in 'XYZ')
        # A line that names G2/G3 and gives no end point is a mistake. A line that
        # only sets a modal state (G91, M3, F400) is not a move at all and must not
        # be judged by the motion mode still in force -- that mistake refuses every
        # post-processor that emits a mode change on its own line.
        if not want:
            if any(g in (2, 3) for g in gs):
                raise GcodeError('arc G%d with no end point' % motion, ln, raw)
            continue
        k = 1.0 if units == 21 else 25.4
        target = list(pos)
        for i, axis in enumerate('XYZ'):
            if axis in vals:
                v = vals[axis] * k
                target[i] = v if absolute else pos[i] + v
        f = feed if feed is not None else 0.0
        if motion in (0, 1):
            d = [target[i] - pos[i] for i in range(3)]
            length = math.sqrt(sum(v * v for v in d))
            if length <= 1e-12:
                # 真机程序里真的会出现原地不动的 G1（重复点、后处理器对齐），
                # 它不是错误，是一个空操作。但它要被数出来：一个程序里成堆的
                # 空操作说明 CAM 那边在掉点，这件事值得在报告里有一行。
                zero_moves.append(ln)
                continue
            t = _unit(d)
            segs.append(Seg(line=ln, text=raw.strip(), kind='line', start=tuple(pos),
                            end=tuple(target), feed=f, length=length, centre=None,
                            ccw=None, radius=None, t_in=t, t_out=t, s=motion,
                            spindle=spindle, rpm=rpm))
        else:
            segs.append(_arc(ln, raw, pos, target, vals, k, False, f, motion == 3,
                             spindle, rpm))
        pos = target
    if not segs:
        raise GcodeError('the program contains no motion at all', None, name)
    meta = {'name': name, 'lines': len(text.splitlines()), 'segments': len(segs),
            'unknown_codes': unknown, 'spindle': spindle, 'spindle_rpm': rpm,
            'zero_length_moves': zero_moves,
            'path_mm': sum(s.length for s in segs)}
    return segs, meta


def _arc(ln, raw, pos, target, vals, k, _absolute, feed, ccw, spindle, rpm=None):
    """I/J (centre offset) or R (radius) form, both handled, both checked.

    The two forms disagree if you let them: I/J is a centre, R is a radius, and a
    radius that cannot reach both ends is a broken line, not a tighter arc.
    """
    if 'R' in vals and ('I' in vals or 'J' in vals):
        raise GcodeError('arc gives both R and I/J', ln, raw)
    x0, y0, z0 = pos
    x1, y1, z1 = target
    if 'R' in vals:
        r_in = vals['R'] * k
        dx, dy = x1 - x0, y1 - y0
        chord = math.sqrt(dx * dx + dy * dy)
        if chord <= 1e-12:
            raise GcodeError('arc with R has zero chord (start == end)', ln, raw)
        if abs(r_in) < chord / 2.0 - 1e-9:
            raise GcodeError('arc radius R=%.4f mm cannot reach both ends (chord %.4f mm)'
                             % (abs(r_in), chord), ln, raw)
        h = math.sqrt(max(0.0, r_in * r_in - (chord / 2.0) ** 2))
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        px, py = -dy / chord, dx / chord
        sign = 1.0 if (r_in > 0) == ccw else -1.0
        cx, cy = mx + sign * h * px, my + sign * h * py
        r = abs(r_in)
    else:
        cx = x0 + vals.get('I', 0.0) * k
        cy = y0 + vals.get('J', 0.0) * k
        r0 = math.hypot(x0 - cx, y0 - cy)
        r1 = math.hypot(x1 - cx, y1 - cy)
        if r0 <= 1e-9:
            raise GcodeError('arc centre coincides with its start point', ln, raw)
        if abs(r0 - r1) > max(1e-6, 0.01 * r0):
            raise GcodeError('arc end is %.4f mm off the circle through its start (r=%.4f)'
                             % (abs(r1 - r0), r0), ln, raw)
        r = r0
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    if ccw:
        sweep = (a1 - a0) % (2 * math.pi)
    else:
        sweep = (a0 - a1) % (2 * math.pi)
    if sweep < 1e-9:
        sweep = 2 * math.pi
    length = r * sweep
    dz = z1 - z0
    ca0, sa0 = math.cos(a0), math.sin(a0)
    ca1, sa1 = math.cos(a1), math.sin(a1)
    if ccw:
        t0 = [-sa0, ca0, 0.0]
        t1 = [-sa1, ca1, 0.0]
    else:
        t0 = [sa0, -ca0, 0.0]
        t1 = [sa1, -ca1, 0.0]
    if abs(dz) > 1e-12:
        step = dz / length
        t0[2] = step
        t1[2] = step
        t0 = list(_unit(t0))
        t1 = list(_unit(t1))
    return Seg(line=ln, text=raw.strip(), kind='arc', start=tuple(pos),
               end=tuple(target), feed=feed, length=length, centre=(cx, cy),
               ccw=ccw, radius=r, t_in=tuple(t0), t_out=tuple(t1),
               s=3 if ccw else 2, spindle=spindle, rpm=rpm)


def load(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return parse(f.read(), name=path)
