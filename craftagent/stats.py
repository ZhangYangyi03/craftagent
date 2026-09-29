"""抗离群的基础统计。纯标准库：工厂机器上少一个依赖就少一个装不上的理由。"""
import math

def median(xs):
    s = sorted(xs); n = len(s)
    if n == 0: return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m-1] + s[m])

def mad(xs):
    if not xs: return 0.0
    m = median(xs)
    return median([abs(x - m) for x in xs])

def scale_of(xs):
    s = 1.4826 * mad(xs)
    if s <= 0:
        m = median(xs)
        s = sum(abs(x - m) for x in xs) / max(1, len(xs))
    return s if s > 0 else 1.0

def stdev(xs):
    n = len(xs)
    if n < 2: return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))

def robust_z(xs):
    m = median(xs); s = scale_of(xs)
    return [(x - m) / s for x in xs]

def pearson(a, b):
    n = min(len(a), len(b))
    if n < 3: return 0.0
    a = a[:n]; b = b[:n]
    ma = sum(a) / n; mb = sum(b) / n
    sa = sum((x - ma) ** 2 for x in a); sb = sum((y - mb) ** 2 for y in b)
    if sa <= 0 or sb <= 0: return 0.0
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(sa * sb)

def ols_slope(y, x=None):
    n = len(y)
    if n < 2: return 0.0, (y[0] if y else 0.0)
    if x is None: x = list(range(n))
    mx = sum(x) / n; my = sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    if sxx <= 0: return 0.0, my
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    b = sxy / sxx
    return b, my - b * mx

_UNITS = {"_mm_s": "mm/s", "_um": "um", "_rh": "%", "_c": "C", "_a": "A", "_mpa": "MPa", "_rpm": "rpm"}
def unit_of(name):
    for suf, u in _UNITS.items():
        if name.endswith(suf): return u
    return ""

def deseasonalize(ts, xs):
    """扣掉"每天同一时刻"的中位数，去掉开机预热那种日周期。
    变点检测前必须做，否则每天的预热台阶会淹掉真正的漂移。
    前提：班次周期整除 1440 分钟，否则桶里混了不同相位，扣不干净。"""
    buckets = {}
    for t, v in zip(ts, xs):
        buckets.setdefault(t.hour * 60 + t.minute, []).append(v)
    base = {k: median(v) for k, v in buckets.items()}
    overall = median(list(base.values()))
    return [v - (base[t.hour * 60 + t.minute] - overall) for t, v in zip(ts, xs)]

def refine_two_level(x, cp, half=60):
    """把变点精修到"两段均值模型残差最小"的位置。CUSUM 给的是粗位置。
    用前缀和，O(half) 而不是 O(half*n)，否则 7000 点的序列会卡住。"""
    n = len(x)
    lo = max(1, cp - half); hi = min(n - 1, cp + half)
    pre = [0.0] * (n + 1); pre2 = [0.0] * (n + 1)
    for i, v in enumerate(x):
        pre[i+1] = pre[i] + v
        pre2[i+1] = pre2[i] + v * v
    def sse(a, b):                      # [a, b)
        m = b - a
        if m <= 0: return 0.0
        s = pre[b] - pre[a]; s2 = pre2[b] - pre2[a]
        return s2 - s * s / m
    best_sse = None; best_c = cp
    for c in range(lo, hi + 1):
        if c < 10 or n - c < 10: continue
        v = sse(0, c) + sse(c, n)
        if best_sse is None or v < best_sse:
            best_sse = v; best_c = c
    return best_c
