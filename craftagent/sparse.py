# -*- coding: utf-8 -*-
"""稀疏矩阵与共轭梯度求解器 —— 为了把三维有限元从"几千自由度就跑不动"里救出来。

为什么要有这个文件
------------------
`fem2d.solve` 用的是稠密高斯消元，未知数一多就是 O(n^3)：二维 84 自由度
瞬间，三维一细化到一千自由度就要几十秒，一万自由度干脆跑不动。而有限元的
刚度矩阵**本来是稀疏的**——每个节点只和同一单元的邻居相连。稠密解法把这个
事实扔掉了，然后花时间在零上做乘加。

所以这里补两样：

  稀疏装配    dict-of-dict，只存非零。单元刚度矩阵 24x24 全满，但装配到
              整体后每行只有几十个非零。
  共轭梯度   只做矩阵-向量乘，不做分解。Jacobi 预条件（对角缩放）。

验收对着什么
------------
  * 与稠密高斯消元解同一个方程组，解必须一致（相对差 1e-9）。
    这是唯一能抓住"装配时行列写反"的检查 —— 行列写反的矩阵仍然对称正定，
    共轭梯度照样收敛，只是收敛到另一个问题的解。
  * 残差必须真的降下来：||b - Ax|| / ||b|| 达到 tol，否则报未收敛而不是
    把最后一步的 x 当答案返回。
  * 病态矩阵上必须报未收敛，不能安静地返回一个像样的向量。

没有做的
--------
不做预条件子以外的加速（没有多重网格、没有 Cholesky 分解）；
不做分块、不做并行；不存 CSR（dict-of-dict 在装配阶段更快，
乘的时候慢一点，但三维这点规模不值得换）。
"""
from __future__ import annotations

import math


def zeros(n):
    """n 行的空稀疏矩阵：每行一个 dict{列: 值}。"""
    return [dict() for _ in range(n)]


def add(A, i, j, v):
    """装配：A[i][j] += v。"""
    row = A[i]
    row[j] = row.get(j, 0.0) + v


def add_block(A, idx, Ke):
    """把一个单元刚度矩阵按自由度编号装配进整体矩阵。

    分开写是因为这里最容易错：`Ke[a][b]` 对应 `(idx[a], idx[b])`，
    写成 `(idx[b], idx[a])` 装配出来仍然对称（因为 Ke 对称），
    所以**看不出来**，直到边界条件一加、解出来的位移场整体偏了一点。
    """
    m = len(idx)
    for a in range(m):
        ia = idx[a]
        for b in range(m):
            add(A, ia, idx[b], Ke[a][b])


def matvec(A, x):
    out = []
    for row in A:
        s = 0.0
        for j, v in row.items():
            s += v * x[j]
        out.append(s)
    return out


def diag(A):
    return [row.get(i, 0.0) for i, row in enumerate(A)]


def to_dense(A):
    n = len(A)
    M = [[0.0] * n for _ in range(n)]
    for i, row in enumerate(A):
        for j, v in row.items():
            M[i][j] = v
    return M


def from_dense(M):
    n = len(M)
    A = zeros(n)
    for i in range(n):
        for j in range(n):
            if M[i][j] != 0.0:
                A[i][j] = M[i][j]
    return A


def solve_dense(A, b, pivot_tol=1e-14):
    """稠密高斯消元（部分选主元）。只用来给稀疏解当参照，不用于生产。

    主元阈值是**相对**的：绝对阈值在刚度矩阵上会让无约束系统解出一串 0.0，
    而零位移场恰好是无载荷时的正确答案，于是没有任何东西会因此变红。
    """
    n = len(A)
    M = to_dense(A)
    x = list(b)
    scale = max((abs(v) for row in M for v in row), default=1.0) or 1.0
    for k in range(n):
        p = max(range(k, n), key=lambda r: abs(M[r][k]))
        if abs(M[p][k]) < pivot_tol * scale:
            raise ValueError("矩阵奇异（第 %d 列主元 %.3e）" % (k, M[p][k]))
        if p != k:
            M[k], M[p] = M[p], M[k]
            x[k], x[p] = x[p], x[k]
        piv = M[k][k]
        for r in range(k + 1, n):
            f = M[r][k] / piv
            if f == 0.0:
                continue
            for c in range(k, n):
                M[r][c] -= f * M[k][c]
            x[r] -= f * x[k]
    out = [0.0] * n
    for k in range(n - 1, -1, -1):
        s = x[k] - sum(M[k][c] * out[c] for c in range(k + 1, n))
        out[k] = s / M[k][k]
    return out


def residual(A, x, b):
    r = matvec(A, x)
    return [b[i] - r[i] for i in range(len(b))]


def norm(v):
    return math.sqrt(sum(t * t for t in v))


def pcg(A, b, tol=1e-10, maxit=4000, x0=None):
    """Jacobi 预条件的共轭梯度。

    返回 dict(x, iters, rel_res, converged)。**不收敛时把 converged 置 False**，
    让调用方决定怎么办 —— 悄悄返回最后一步的 x 是这里最坏的失败方式，
    因为它在三维里看起来永远像一个光滑的位移场。
    """
    n = len(b)
    x = list(x0) if x0 else [0.0] * n
    d = diag(A)
    if any(abs(t) < 1e-300 for t in d):
        return {"x": x, "iters": 0, "rel_res": 1.0, "converged": False}
    r = residual(A, x, b)
    bnorm = norm(b) or 1.0
    z = [r[i] / d[i] for i in range(n)]
    p = list(z)
    rz = sum(r[i] * z[i] for i in range(n))
    rel = norm(r) / bnorm
    for it in range(1, maxit + 1):
        if rel <= tol:
            return {"x": x, "iters": it - 1, "rel_res": rel, "converged": True}
        Ap = matvec(A, p)
        pAp = sum(p[i] * Ap[i] for i in range(n))
        if abs(pAp) < 1e-300:
            break
        alpha = rz / pAp
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rel = norm(r) / bnorm
        z = [r[i] / d[i] for i in range(n)]
        rz_new = sum(r[i] * z[i] for i in range(n))
        beta = rz_new / rz if rz != 0.0 else 0.0
        rz = rz_new
        for i in range(n):
            p[i] = z[i] + beta * p[i]
    return {"x": x, "iters": maxit, "rel_res": rel, "converged": rel <= tol}


def apply_dirichlet(A, b, fixed):
    """把 fixed（{自由度: 位移}）钉住：行清零、列移项、对角置 1。

    只清行不清列，解出来的非约束自由度会被约束位移的反力污染 ——
    而结果仍然光滑、仍然满足约束，所以从位移场上看不出来。
    """
    n = len(b)
    bb = list(b)
    for dof, val in fixed.items():
        for j, v in A[dof].items():
            if j != dof:
                bb[j] -= v * val
        A[dof] = {dof: 1.0}
    for i in range(n):
        row = A[i]
        drop = [j for j in row if j in fixed and j != i]
        for j in drop:
            del row[j]
    for dof, val in fixed.items():
        bb[dof] = val
    return bb
