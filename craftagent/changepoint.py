"""变点检测：最小二乘二分分割 + 抗离群显著性检验。纯 Python，前缀和保证线性时间。
只用 CUSUM 会被慢漂移和噪声同时骗到：噪声段也能刷出高 CUSUM 值。
所以每一步分割都必须通过一个检验——两段中位数之差至少是 3 倍段内 MAD 尺度。"""
from .stats import median, scale_of

def _prefix(x):
    n = len(x); pre = [0.0]*(n+1); pre2 = [0.0]*(n+1)
    for i, v in enumerate(x):
        pre[i+1] = pre[i] + v; pre2[i+1] = pre2[i] + v*v
    return pre, pre2

def _sse(pre, pre2, a, b):
    m = b - a
    if m <= 0: return 0.0
    s = pre[b] - pre[a]; s2 = pre2[b] - pre2[a]
    return s2 - s*s/m

def find_changes(x, min_size=240, scale_mult=3.0, max_cp=4, max_depth=6, max_shift=None):
    """返回 [(index, 效应量), ...]，index 是采样点下标（1 点 = 1 分钟）。
    效应量 = 两段中位数之差 / 段内 MAD 尺度，可直接当"这个变点有多硬"来读。"""
    n = len(x); pre, pre2 = _prefix(x); out = []
    def rec(lo, hi, depth):
        if hi - lo < 2*min_size or depth > max_depth or len(out) >= max_cp: return
        best = None
        for c in range(lo + min_size, hi - min_size + 1):
            v = _sse(pre, pre2, lo, c) + _sse(pre, pre2, c, hi)
            if best is None or v < best[0]: best = (v, c)
        if best is None: return
        c = best[1]
        left = x[lo:c]; right = x[c:hi]
        shift = abs(median(right) - median(left))
        sc = scale_of(left + right)
        eff = shift / sc if sc > 0 else 0.0
        if eff < scale_mult: return
        if max_shift is not None and shift > max_shift: return
        out.append((c, round(eff, 2)))
        rec(lo, c, depth+1); rec(c, hi, depth+1)
    rec(0, n, 0)
    return sorted(out)
