# -*- coding: utf-8 -*-
"""booleans.py -- OCC 精确 B-rep 布尔（并 / 交 / 差）的子进程适配器。

这一层和 csg.py 的分工
----------------------
csg.py 在三角网格上用 BSP 做布尔：任何地方都能跑，能在本机对着解析体积
验收，但结果是折线逼近，而且会留 T 型接点（实测带孔件 320 条裂边）。

这里走另一条路：把布尔交给 OpenCASCADE 的 B-rep 内核，结果仍是精确的
曲面，面数不随逼近精度膨胀。代价是必须有 OCP，而这台机器上没有 ——
所以这一层**在本机只能验"缺东西时它说不"，不能验数值**。

为什么不干脆只留 csg.py
------------------------
两条路回答的不是同一个问题。csg 那条能回答"体积对不对"，回答不了
"这张面还是不是原来那张圆柱面"。下游要做刀路或者出图的时候，需要的
恰恰是后者。所以两个都留着，各自标明边界。

关于"本机验不了"
----------------
这不是可以糊过去的地方。`selftest()` 在本机返回的是 available=False 和
一句为什么，CI 上（装了 cadquery-ocp 的两个 job）才跑真正的数值验收。
把"没验过"说成"验过了"是这一层最容易犯的错，所以状态字段叫
`verified_here`，而不是 `ok`。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from . import cad as _cad

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "tools", "occ_boolean.py")
_SELFTEST = os.path.join(_HERE, "tools", "occ_selftest.py")

__all__ = ["status", "boolean", "selftest", "OPS"]

OPS = ("fuse", "cut", "common")


def status(candidates=None):
    """这一层可不可用，以及凭什么这么说。复用 cad 的解释器探测。"""
    st = _cad.backend_status(candidates)
    out = {"available": bool(st.get("available")), "worker": _WORKER,
           "tried": st.get("tried")}
    if st.get("available"):
        out["python"] = st["python"]
    else:
        out["why"] = st.get("why")
        out["how_to_fix"] = st.get("how_to_fix")
    return out


def boolean(op, path_a, path_b, out_step=None, tol=1e-6, timeout=600,
            candidates=None):
    """跑一次布尔。缺 OCP 时**明确失败**，不返回空结果。

    返回空结果是最坏的失败方式：下游看到"体积 0"会理解成"没有材料"或
    "没有干涉"，而那是安全方向上的假阴性。
    """
    if op not in OPS:
        raise ValueError("op 必须是 %s 之一，收到 %r" % (OPS, op))
    st = status(candidates)
    if not st["available"]:
        return {"ok": False, "why": "no_opencascade", "detail": st.get("why"),
                "tried": st.get("tried"), "verified_here": False}
    out_step = out_step or os.path.join(tempfile.mkdtemp(), "bool.step")
    cmd = [st["python"], _WORKER, op, path_a, path_b, out_step, "--tol", str(tol)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:                                # pragma: no cover
        return {"ok": False, "why": "worker_exception", "detail": str(exc)}
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return {"ok": False, "why": "no_output", "stderr": (r.stderr or "")[-800:]}
    try:
        out = json.loads(line[-1])
    except ValueError:
        return {"ok": False, "why": "bad_json", "stdout": line[-1][:400]}
    out["verified_here"] = True
    return out


def selftest(candidates=None, timeout=600):
    """真的做一遍三个布尔，对解析体积。做不到就如实说没做。

    判据在 OCC 之外：两个 10 立方体偏移 5，并 1875 / 交 125 / 差 875，
    三个数都有闭式。结果还要过 BRepCheck_Analyzer，并且至少剩一个实体
    —— 空壳是最坏的失败方式，下游会读成"没有材料"或"没有干涉"。

    返回里的 `verified_here` 只有在**这个进程真的跑过那个子进程并拿到
    ok=True** 时才是 True。本机没有 OCP 时它是 False，而那不是"通过"。
    """
    st = status(candidates)
    if not st["available"]:
        return {"available": False, "verified_here": False,
                "why": st.get("why"), "tried": st.get("tried"),
                "how_to_fix": st.get("how_to_fix"),
                "ci_only": True,
                "note": "数值验收只能在装了 OCP 的解释器里做；"
                        "本机没有，所以这里不声称任何数字"}
    try:
        r = subprocess.run([st["python"], _SELFTEST], capture_output=True,
                           text=True, timeout=timeout)
    except Exception as exc:                                # pragma: no cover
        return {"available": True, "verified_here": False,
                "why": "worker_exception", "detail": str(exc)[:300],
                "python": st["python"]}
    lines = (r.stdout or "").strip().splitlines()
    if not lines:
        return {"available": True, "verified_here": False,
                "why": "no_output", "stderr": (r.stderr or "")[-800:],
                "python": st["python"]}
    try:
        out = json.loads(lines[-1])
    except ValueError:
        return {"available": True, "verified_here": False, "why": "bad_json",
                "stdout": lines[-1][:400], "python": st["python"]}
    out["available"] = True
    out["verified_here"] = bool(out.get("ok"))
    out["python"] = st["python"]
    return out
