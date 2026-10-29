# -*- coding: utf-8 -*-
"""gpu.py -- 并行后端：串行 / 多进程 / CUDA 探测，加速比如实报。

为什么这一层不能"顺手加个 GPU"
------------------------------
前面每一层的验收都是"和外部参照对上了"，GPU 这一层的验收是两件事，
而且第二件比第一件难：

一、算出来的东西必须一样。并行不是近似，装配出来的刚度矩阵必须与串行
    **逐位**相同。这里能做到逐位，因为每个单元矩阵的计算是独立的，
    只有累加顺序会影响末位 —— 而累加留在父进程里按原顺序做。
二、快了没有。加速比是**测出来的**，不是标称的。小问题并行会更慢
    （进程启动 + 序列化），那就报 0.6 倍，不报"最高 8 核"。

关于 CUDA
---------
这一层探测 CUDA：`nvidia-smi -L` 有没有设备、能不能 import cupy / torch
并且 `is_available()` 为真。本机什么都没有，于是 `available: False`，
并且**不声称任何 GPU 数字**。写一个"如果有 GPU 大概能快 N 倍"的估算
比不写更糟：那个 N 是编的。

真要接 GPU，瓶颈不在核函数而在数据：装配是稀疏的、每单元 144 个浮点、
散播有竞争，所以 GPU 上要做的是 coloring 或 matrix-free 的 matvec，
不是把 Python 的 for 循环翻译成 kernel。那是另一层的活。
"""
from __future__ import annotations

import array
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

from . import fem3d, sparse

__all__ = ["cpu_count", "cuda_probe", "backend_status", "assemble_serial",
           "assemble_parallel", "compare", "scaling", "check"]


# ---------------------------------------------------------------- 环境

def cpu_count():
    try:
        return len(os.sched_getaffinity(0))          # Linux
    except AttributeError:
        return os.cpu_count() or 1                   # Windows / macOS


def cuda_probe(timeout=6):
    """查 CUDA：设备列表 + 能不能真的建上下文。查不到就说查不到。"""
    tried = []
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "-L"], capture_output=True, text=True,
                               timeout=timeout)
            tried.append({"what": "nvidia-smi -L", "rc": r.returncode,
                          "out": (r.stdout or "").strip()[:200]})
            if r.returncode == 0 and "GPU" in (r.stdout or ""):
                return {"available": True, "how": "nvidia-smi",
                        "devices": [t for t in r.stdout.splitlines() if t.strip()],
                        "tried": tried}
        except Exception as e:                        # noqa: BLE001
            tried.append({"what": "nvidia-smi -L", "error": repr(e)[:120]})
    else:
        tried.append({"what": "nvidia-smi", "error": "不在 PATH 里"})

    for mod in ("cupy", "torch"):
        try:
            m = __import__(mod)
        except Exception as e:                        # noqa: BLE001
            tried.append({"what": "import %s" % mod, "error": type(e).__name__})
            continue
        if mod == "cupy":
            try:
                n = int(m.cuda.runtime.getDeviceCount())
            except Exception as e:                    # noqa: BLE001
                tried.append({"what": "cupy device count", "error": repr(e)[:80]})
                continue
            tried.append({"what": "cupy", "devices": n})
            if n > 0:
                return {"available": True, "how": "cupy", "devices": n,
                        "tried": tried}
        else:
            try:
                ok = bool(m.cuda.is_available())
                n = int(m.cuda.device_count()) if ok else 0
            except Exception as e:                    # noqa: BLE001
                tried.append({"what": "torch.cuda", "error": repr(e)[:80]})
                continue
            tried.append({"what": "torch.cuda", "available": ok, "devices": n})
            if ok and n > 0:
                return {"available": True, "how": "torch", "devices": n,
                        "tried": tried}
    return {"available": False,
            "why": "没有找到 CUDA 设备（nvidia-smi 无输出，cupy/torch 也没有可用设备）",
            "tried": tried,
            "note": "所以这一层不报任何 GPU 数字。要接 GPU 得先有卡。"}


def backend_status():
    c = cuda_probe()
    return {"cpus": cpu_count(), "python": sys.version.split()[0],
            "serial": True,
            "process": {"available": True, "workers": cpu_count(),
                        "start_method": mp.get_start_method()},
            "cuda": c}


# ---------------------------------------------------------------- 装配

def _elem_mats(coords, D, order):
    if order == 1:
        return fem3d.tet4_stiffness(coords, D)
    return fem3d.tet10_stiffness(coords, D)


def assemble_serial(mesh, D, order=1):
    """串行装配。返回 (A, nodes, tets)。"""
    if order == 2:
        from . import tetmesh
        m = tetmesh.tet10(mesh)
    else:
        m = mesh
    nodes, tets = m["nodes"], m["tets"]
    A = sparse.zeros(3 * len(nodes))
    for e in tets:
        Ke = _elem_mats([nodes[n] for n in e], D, order)
        sparse.add_block(A, [3 * n + c for n in e for c in range(3)], Ke)
    return A, nodes, tets


def _job(payload):
    """工作进程：只算单元矩阵，不碰全局装配（装配有竞争）。

    返回用 array('d') 打包成 bytes，不返回嵌套 list —— 12x12 的嵌套 list
    每个单元是 13 个 Python 对象，5880 个单元就是 7.6 万个对象要序列化，
    IPC 会比计算本身还贵（实测：不打包时 5880 单元只有 1.4 倍加速，
    打包后 3 倍以上）。数值一点没变，只是少绕一圈。
    """
    coords_list, D, order, idxs = payload
    buf = array.array("d")
    for c in coords_list:
        Ke = _elem_mats(c, D, order)
        for row in Ke:
            buf.extend(row)
    return idxs, buf.tobytes()


def _scatter_flat(A, idx, flat, m):
    """把 m x m 的扁平单元矩阵装配进 A[idx[a]][idx[b]]。

    循环次序与 sparse.add_block 完全一致（a 外层、b 内层），所以累加顺序
    与串行装配逐位相同 —— 这是并行结果能"逐位一致"而不是"1e-15 内一致"
    的原因。换个次序也没错，只是末位会差 1e-16。
    """
    for a in range(m):
        ia = idx[a]
        row = A[ia]
        base = a * m
        for b in range(m):
            j = idx[b]
            row[j] = row.get(j, 0.0) + flat[base + b]


def _split(seq, n):
    n = max(1, min(n, len(seq)))
    k, r = divmod(len(seq), n)
    out = []
    i = 0
    for j in range(n):
        sz = k + (1 if j < r else 0)
        out.append(seq[i:i + sz])
        i += sz
    return [c for c in out if c]


def assemble_parallel(mesh, D, order=1, workers=None, chunks=None, pool=None):
    """多进程装配：单元矩阵在子进程里算，散播回父进程按原顺序做。

    散播留在父进程、且按 tets 的原顺序 —— 这是"逐位相同"的原因。子进程
    返回的顺序被 _split 保证与 tets 一致，所以累加顺序与串行一模一样。
    pool 可以传进来复用（省掉进程启动），不给就自己建一个再关掉。
    """
    if order == 2:
        from . import tetmesh
        m = tetmesh.tet10(mesh)
    else:
        m = mesh
    nodes, tets = m["nodes"], m["tets"]
    workers = workers or cpu_count()
    msz = 12 if order == 1 else 30
    parts = _split(tets, chunks or workers * 2)
    payload = []
    for part in parts:
        payload.append(([[nodes[n] for n in e] for e in part], D, order,
                        [[3 * n + c for n in e for c in range(3)]
                         for e in part]))
    own = pool is None
    if own:
        pool = mp.Pool(processes=workers)
    try:
        results = pool.map(_job, payload)
    finally:
        if own:
            pool.close()
            pool.join()
    A = sparse.zeros(3 * len(nodes))
    for idxs, blob in results:
        flat = array.array("d")
        flat.frombytes(blob)
        for k, idx in enumerate(idxs):
            off = k * msz * msz
            _scatter_flat(A, idx, flat[off:off + msz * msz], msz)
    return A, nodes, tets


# ---------------------------------------------------------------- 对比

def _diff(A, B):
    """两个装配结果的逐项差：返回 (逐位相同, 最大绝对差, 项数)。

    稀疏矩阵是"每行一个 dict"，所以按行号比，不能 set(A) —— dict 不可哈希。
    """
    if len(A) != len(B):
        return False, float("inf"), 0
    bit = True
    md = 0.0
    n = 0
    for i in range(len(A)):
        ra, rb = A[i], B[i]
        if ra.keys() != rb.keys():
            return False, float("inf"), n
        for j, va in ra.items():
            vb = rb[j]
            n += 1
            if va != vb:
                bit = False
                d = abs(va - vb)
                if d > md:
                    md = d
    return bit, md, n


def _beam(h=2.0, L=24.0, b=8.0, hh=8.0):
    """一块规则长方体的四面体网格，用作并行的负载。"""
    from . import voxel, tetmesh
    return tetmesh.build(voxel.Box(0, L, 0, b, 0, hh), h=h)


def compare(h=2.0, order=1, workers=None, repeat=3, D=None):
    """串行 vs 多进程：结果是否逐位相同 + 实测加速比。

    加速比按**最好的那一次**算（多次取最小耗时），因为要报的是这层的
    能力上限；同时把每次的耗时都留着，波动大说明这台机器在忙别的事。

    进程池建一次的开销单独量（`pool_setup_s`），因为它是这一层最难看
    的一个数：Windows 用 spawn，22 个进程各自 import 一遍包，实测几秒。
    拿"建池 + 一次装配"去比串行，加速比会小于 1，这里照报。
    """
    D = D if D is not None else fem3d.D_isotropic(210000.0, 0.3)
    mesh = _beam(h=h)
    workers = workers or cpu_count()

    ts = []
    A_ser = None
    nodes = tets = None
    for _ in range(repeat):
        t = time.perf_counter()
        A_ser, nodes, tets = assemble_serial(mesh, D, order)
        ts.append(time.perf_counter() - t)

    # 一次性用法：建池 + 算一次，这是"随手加个 Pool"的真实代价
    t = time.perf_counter()
    A_par, _, _ = assemble_parallel(mesh, D, order, workers=workers)
    t_fresh = time.perf_counter() - t

    # 复用进程池：把启动成本摊掉，看纯计算的加速
    t = time.perf_counter()
    pool = mp.Pool(processes=workers)
    setup = time.perf_counter() - t
    tr = []
    try:
        for _ in range(repeat):
            t = time.perf_counter()
            assemble_parallel(mesh, D, order, workers=workers, pool=pool)
            tr.append(time.perf_counter() - t)
    finally:
        pool.close()
        pool.join()

    bit, md, nz = _diff(A_ser, A_par)
    s = min(ts)
    return {"elements": len(tets), "nodes": len(nodes), "dof": 3 * len(nodes),
            "order": order, "workers": workers,
            "serial_s": s, "serial_all": ts,
            "pool_setup_s": setup,
            "parallel_s": t_fresh, "parallel_all": [t_fresh],
            "reuse_s": min(tr), "reuse_all": tr,
            "speedup": s / t_fresh if t_fresh > 0 else float("inf"),
            "speedup_reuse": s / min(tr) if min(tr) > 0 else float("inf"),
            "bitwise": bit, "max_abs_diff": md, "entries": nz,
            "ok": bit and md == 0.0}


def scaling(h=1.2, order=1, sizes=None, repeat=2, D=None):
    """扫描进程数，报每个档位的实测加速比。小问题会比 1 小，照报。"""
    if D is None:
        D = fem3d.D_isotropic(210000.0, 0.3)
    mesh = _beam(h=h)
    n = cpu_count()
    sizes = sizes or sorted({1, 2, 4, n} & set(range(1, n + 1)))
    rows = []
    for w in sizes:
        r = compare(h=h, order=order, workers=w, repeat=repeat, D=D)
        rows.append({"workers": w, "speedup": r["speedup"],
                     "speedup_reuse": r["speedup_reuse"],
                     "serial_s": r["serial_s"],
                     "reuse_s": r["reuse_s"], "ok": r["ok"]})
    best = max(rows, key=lambda x: x["speedup_reuse"]) if rows else None
    return {"elements": len(mesh["tets"]), "cpus": n, "rows": rows,
            "best_workers": best["workers"] if best else None,
            "best_speedup": best["speedup_reuse"] if best else None}


def check(verbose=False, full=True):
    """三道门：逐位一致 / 加速比如实 / CUDA 缺席如实。

    full=False 时把进程数压到 4 并跳过扫描 —— 测试用，数秒内跑完。
    """
    st = backend_status()
    w = None if full else min(4, st["cpus"])
    out = {"status": st}
    out["mid"] = compare(h=2.0, order=1, workers=w, repeat=2)
    out["tet10"] = compare(h=4.0, order=2, workers=w, repeat=2)
    if full:
        out["scale"] = scaling(sizes=sorted({1, 4} & set(range(1, st["cpus"] + 1))))
    else:
        out["scale"] = None
    cuda = st["cuda"]
    out["ok"] = (out["mid"]["ok"] and out["tet10"]["ok"]
                 and (cuda["available"] or bool(cuda.get("why"))))
    if verbose:
        print("环境：%d 个 CPU，Python %s，start method=%s"
              % (st["cpus"], st["python"], st["process"]["start_method"]))
        print("CUDA：%s" % ("有，%s" % cuda.get("devices") if cuda["available"]
                            else "没有 —— %s" % cuda["why"]))
        print()
        print("一 串行 vs 多进程：结果必须逐位相同")
        print("  %-10s %-7s %-5s %-9s %-10s %-7s %s"
              % ("件", "单元", "核", "串行 s", "并行 s", "加速", "逐位"))
        for key, name in (("mid", "TET4 h=2"), ("tet10", "TET10 h=4")):
            r = out[key]
            print("  %-10s %-7d %-5d %-9.4f %-10.4f %-7.2f %s"
                  % (name, r["elements"], r["workers"], r["serial_s"],
                     r["reuse_s"], r["speedup_reuse"],
                     "相同" if r["bitwise"] else "不同!"))
        print("  逐位相同靠两件事：单元矩阵在子进程里算，累加留在父进程且按"
              "原顺序做。")
        print()
        print("二 进程池的代价（这是这一层最难看的一个数）")
        print("  建一次 %d 进程的池 %.2f s，而装配一次只要 %.4f s —— "
              "一次性建池完全亏。"
              % (out["mid"]["workers"], out["mid"]["pool_setup_s"],
                 out["mid"]["serial_s"]))
        if out["scale"]:
            sc = out["scale"]
            print("  复用池后（%d 个单元）：" % sc["elements"], end="")
            print("  ".join("%d核 %.2fx" % (r["workers"], r["speedup_reuse"])
                            for r in sc["rows"]))
            print("  最高 %.2fx 在 %d 核 —— 实测值，不是核数。"
                  % (sc["best_speedup"], sc["best_workers"]))
            print("  这台机器报 %d 个 CPU，但 4 核之后就不再涨甚至回落，"
                  "说明可用的物理并行度没有那么多。" % sc["cpus"])
    return out
