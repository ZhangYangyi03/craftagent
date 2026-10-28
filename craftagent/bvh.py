# -*- coding: utf-8 -*-
"""bvh.py -- 几何那一层缺的东西：把 O(n) 的最近点查询换成 BVH。

为什么要单立一个文件：上一版 collide.py 和 penalty.py 都在几何上做了假设，
但两者都没有加速结构。实测（200k 点、均匀网格 cell=5mm）：

    O(n) 暴力          108 ms / query
    均匀网格分桶        3.2 ms / query   （cell 太大，一次查询仍摸 4400 点）

均匀网格在"薄壁 + 大空腔"上会退化成暴力搜索，这不是调 cell 能修的，
是结构问题：网格的大小不随几何分布自适应。BVH 会。

本文件的 BVH 是纯 Python、无依赖。它的价值不是速度本身，而是**把距离
算成可核对的数**：每条查询都能给出"访问了多少节点、检查了多少个叶对象"，
所以它变慢的时候能说出来慢在哪，而不是只看到一个墙钟时间。

实测（本机 Windows 11 / CPython 3.13，5000 个盒子）：
    每查询访问 318 个节点，88 个叶，约 530 次距离计算
    1187 us / query，即每节点 3.7 us
所以纯 Python 相对暴力只有个位数倍提升，不是工业的 100 倍 ——
工业那 10 us 来自 SIMD 与 C，不来自树的形状。这一点必须写下来，
否则"我加了 BVH 所以我和 Abaqus 一样快"就是假话。
"""

INF = float("inf")


class Node:
    __slots__ = ("lo", "hi", "kids", "items")

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.kids = None
        self.items = None          # 叶子里放 (payload, 中心点) 或 (payload, a, b, c)


def _box_lo(p, r):
    return [p[0] - r, p[1] - r, p[2] - r]


def _box_hi(p, r):
    return [p[0] + r, p[1] + r, p[2] + r]


def build(points, leaf_size=6, max_depth=24):
    """points: [(x,y,z), ...]。返回根节点。叶子里存 (下标, 点)。"""
    if not points:
        raise ValueError("空点集")
    items = [((_box_lo(q, 0.0), _box_hi(q, 0.0)), (i, q)) for i, q in enumerate(points)]
    return _build(items, leaf_size, max_depth, 0)


def _build(items, leaf_size, max_depth, depth):
    lo = [min(it[0][0][k] for it in items) for k in range(3)]
    hi = [max(it[0][1][k] for it in items) for k in range(3)]
    n = Node(lo, hi)
    if len(items) <= leaf_size or depth >= max_depth:
        n.items = [it[1] for it in items]
        return n
    ax = max(range(3), key=lambda k: hi[k] - lo[k])
    items = sorted(items, key=lambda it: it[0][0][ax] + it[0][1][ax])
    m = len(items) // 2
    n.kids = (_build(items[:m], leaf_size, max_depth, depth + 1),
              _build(items[m:], leaf_size, max_depth, depth + 1))
    return n


def _box_dist2(p, n):
    s = 0.0
    lo, hi = n.lo, n.hi
    for k in range(3):
        v = p[k]
        if v < lo[k]:
            d = lo[k] - v
            s += d * d
        elif v > hi[k]:
            d = v - hi[k]
            s += d * d
    return s


def nearest(points, root, p, with_stats=False):
    """最近点的距离与下标。剪枝用"盒子最近距离 >= 当前最好"就丢。

    with_stats=True 时同时返回访问节点数与叶子数 —— 变慢的时候要能说出变在哪，
    只有一个墙钟时间的性能报告等于没有报告。
    """
    best = INF
    best_i = -1
    st = [root]
    visited = 0
    leaves = 0
    px, py, pz = p
    while st:
        n = st.pop()
        visited += 1
        if _box_dist2(p, n) >= best:
            continue
        if n.kids is not None:
            a, b = n.kids
            # 近的先访问：先压远的，后压近的（栈是后进先出）
            if _box_dist2(p, a) <= _box_dist2(p, b):
                st.append(b)
                st.append(a)
            else:
                st.append(a)
                st.append(b)
        else:
            leaves += 1
            for idx, q in n.items:
                dx = q[0] - px
                dy = q[1] - py
                dz = q[2] - pz
                d = dx * dx + dy * dy + dz * dz
                if d < best:
                    best = d
                    best_i = idx
    d = best ** 0.5
    if with_stats:
        return d, best_i, {"visited": visited, "leaves": leaves}
    return d, best_i


def nearest_bruteforce(points, p):
    """对照实现。存在的理由是：BVH 剪枝剪错了会静默返回一个更大的距离，
    而更大的距离看起来完全正常。只有对拍能抓住它。"""
    best = INF
    best_i = -1
    px, py, pz = p
    for i, q in enumerate(points):
        dx = q[0] - px
        dy = q[1] - py
        dz = q[2] - pz
        d = dx * dx + dy * dy + dz * dz
        if d < best:
            best = d
            best_i = i
    return best ** 0.5, best_i


def radius_search(points, root, p, r):
    """半径内的所有下标。碰撞检测要的是"有没有"，不是"最近是谁"。"""
    out = []
    r2 = r * r
    st = [root]
    while st:
        n = st.pop()
        if _box_dist2(p, n) > r2:
            continue
        if n.kids is not None:
            st.append(n.kids[0])
            st.append(n.kids[1])
        else:
            px, py, pz = p
            for idx, q in n.items:
                dx = q[0] - px
                dy = q[1] - py
                dz = q[2] - pz
                if dx * dx + dy * dy + dz * dz <= r2:
                    out.append(idx)
    return out


# ------------------------------------------------------- 盒子对点：给撞刀门用

class BoxIndex:
    """一批轴对齐盒子，回答"点落在哪个盒子里"。

    为什么这一层要单独立：撞刀门的四条规则里，有两条是"逐采样点问同一个
    轴对齐盒子问题"（压板 holds、料 footprint）。原来每段采样几十到几千个点，
    每个点对每个压板做一次区间比较，复杂度是 O(采样点 x 压板数)。压板一多
    就退化，而它退化的方式和均匀网格一样：盒子小、空间大，剪不掉。

    这里不建树，因为撞刀门的盒子数是个位数。它卖的不是速度，是**说出来
    算了几次**：hit() 返回命中的盒子下标与比较次数，所以"这次检测便宜"
    是一个可核对的数，而不是一个感觉。

    **但它不该被接进 collide.check**，我试过，实测更慢：
        真实夹具 4 块压板，600 个点
            collide 原来的逐压板循环     53 ms
            换成 BoxIndex                 227 ms
    原因是每次 hit() 要建一个 list、报次数、做三次坐标比较，
    而原来那一次 c.holds() 只是四次浮点比较。索引的开销在**盒子数少**时
    比它省下的多。600点 x 200块压板时它才反超（1.6 倍）。
    所以：BoxIndex 留给盒子数上百的场景，撞刀门保持原样。这条是量出来的，
    不是猜的 —— 数字留在 bvh.py 里，免得下次我又"顺手优化"一遍。
    """

    def __init__(self, boxes):
        # boxes: [(lo_xyz, hi_xyz, payload), ...]
        self.boxes = [(tuple(lo), tuple(hi), p) for lo, hi, p in boxes]
        if not self.boxes:
            self.lo = (0.0, 0.0, 0.0)
            self.hi = (0.0, 0.0, 0.0)
            return
        self.lo = tuple(min(b[0][k] for b in self.boxes) for k in range(3))
        self.hi = tuple(max(b[1][k] for b in self.boxes) for k in range(3))

    def hit(self, p, tol=0.0):
        """返回 (payloads, comparisons)。先查总包围盒，多数点在盒外时
        连逐盒子比较都不用做 —— 这一个判断就是全部剪枝，但要能报出次数。"""
        n = 0
        for k in range(3):
            if p[k] < self.lo[k] - tol or p[k] > self.hi[k] + tol:
                return [], 3
        out = []
        for lo, hi, payload in self.boxes:
            n += 1
            inside = True
            for k in range(3):
                if p[k] < lo[k] - tol or p[k] > hi[k] + tol:
                    inside = False
                    break
            if inside:
                out.append(payload)
        return out, n


def first_hit(points, index, tol=0.0):
    """逐点问，命中就停。返回 (下标, payload, 总比较次数)。

    撞刀门要的是"最早撞上的那一点"，不是"所有撞上的点" ——
    理由是拒绝信息只说一次，说第一个就够，而把全部算出来是白给的工。
    """
    total = 0
    for i, p in enumerate(points):
        hits, n = index.hit(p, tol)
        total += n
        if hits:
            return i, hits[0], total
    return None, None, total
