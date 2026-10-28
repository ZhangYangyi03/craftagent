# -*- coding: utf-8 -*-
"""cad.py -- 把 STEP/IGES 读成一个能算的东西：三角面 + 包围盒 + 拓扑计数。

这一层补的是前面四层里唯一完全空着的一端：**读不进一个真实零件**。
前面 fem2d 能解我摆的矩形块、bvh 能在点云里找最近点，但都没有"零件"
这个东西。这里把它接上。

设计上的两个决定，都不是技术偏好：

一、OpenCASCADE 走独立进程（tools/occ_worker.py），不走 import。
   craftagent 的核心是纯标准库，CI 在四个平台上跑；把 300 MB 的 OCP
   变成 import 依赖，等于让 CI 在两个平台上永久变红。适配器缺失时
   这一层必须**明确报"没有"**，而不是静默返回一个空零件 —— 空零件
   在下游看起来就是"没有干涉"，那是安全方向上的假阴性。

二、返回的每一条都要带出处。
   counts 是 OCC 数的，volume/area 是 OCC 积分出来的，bbox 是从三角面
   顶点算的。三角化和拓扑计数对不上时（比如面数 >0 但三角面为 0）
   这里直接判失败，因为那正是"读进来了但是空的"。

这个文件不假装自己是 CAD 内核：它不做布尔、不做曲面求交、不做参数化建模。
它只做一件半事：读进来，三角化。布尔那些要写东西进 OCC，那是另一层。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "tools", "occ_worker.py")
# 本机上额外值得一试的解释器（装在项目里、PATH 上看不到的那些）。
# 这里**只是补充**，不是全部 —— 见 _interpreter_candidates() 的顺序。
_EXTRA_PYTHONS = (
    os.path.join(os.path.dirname(_HERE), ".venv-occ", "Scripts", "python.exe"),
    os.path.join(os.path.dirname(_HERE), ".venv-occ", "bin", "python"),
    r"C:\Users\china\miniconda3\python.exe",
)
SUPPORTED_EXT = (".step", ".stp", ".iges", ".igs")


def _interpreter_candidates():
    """按顺序给出所有值得一试的解释器。

    顺序是有教训的，第一版写错了：我原来只列了本机那几个硬编码路径，
    结果 CI 上装了 cadquery-ocp 却报"没有 OpenCASCADE" —— 因为 CI 把
    它装进的就是当前解释器（sys.executable），而那个根本不在我的列表里。

    这就是"读进来了但是空的"的另一种形态：探测范围太窄，于是适配器
    报告"没有"，而实际上有。所以 sys.executable 和 PATH 必须排在
    硬编码路径前面 —— 能装 OCP 的地方，最常见的就是当前解释器。
    """
    env = os.environ.get("CRAFTAGENT_OCC_PYTHON")
    out = []
    if env:
        out.append(env)
    out.append(sys.executable)
    for name in ("python3", "python"):
        w = shutil.which(name)
        if w:
            out.append(w)
    out.extend(_EXTRA_PYTHONS)
    seen, uniq = set(), []
    for p_ in out:
        if p_ and p_ not in seen:
            seen.add(p_)
            uniq.append(p_)
    return uniq


def find_interpreter(candidates=None):
    """找一个能 import OCP 的解释器，找不到返回 None。

    `candidates` 只为测试存在：想断言"一个都没有时它明确说不"，
    就得能把候选列表换成空。"""
    cands = _interpreter_candidates() if candidates is None else list(candidates)
    for py in cands:
        if not py or not os.path.exists(py):
            continue
        try:
            r = subprocess.run([py, "-c", "import OCP"], capture_output=True,
                               timeout=120)
        except Exception:
            continue
        if r.returncode == 0:
            return py
    return None


def backend_status(candidates=None):
    """这一层到底可不可用，以及凭什么这么说。给 CLI 用，也给测试用。

    `tried` 必须是真的试过的那一串，不是"我打算试的" —— 一个说不清
    自己试过什么的诊断，下一次还是查不出问题。"""
    cands = _interpreter_candidates() if candidates is None else list(candidates)
    py = find_interpreter(cands)
    if py is None:
        return {"available": False,
                "why": "没有找到能 import OCP 的解释器（OpenCASCADE）",
                "tried": cands,
                "how_to_fix": "python -m venv .venv-occ && .venv-occ/Scripts/pip "
                              "install cadquery-ocp"}
    return {"available": True, "python": py, "worker": _WORKER,
            "tried": cands}


def read_part(path, tol=0.5, timeout=300):
    """读一个 STEP/IGES，返回 dict。失败时 ok=False 并说明原因。

    tol 是三角化线性挠度（模型单位，通常 mm）。0.5 对装配级干涉检查够用，
    要拿去应力分析就得小一个量级 —— 这个数必须由调用方决定，不能内置一个
    看起来合理的默认值然后让大家以为它是精度。
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {"ok": False, "why": "no_such_file", "path": path}
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        return {"ok": False, "why": "unsupported_extension", "ext": ext,
                "supported": list(SUPPORTED_EXT)}
    py = find_interpreter()
    if py is None:
        return {"ok": False, "why": "no_opencascade",
                "detail": backend_status()["why"]}
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        r = subprocess.run([py, _WORKER, path, out, "--tol", str(tol)],
                           capture_output=True, text=True, timeout=timeout)
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            return {"ok": False, "why": "worker_produced_nothing",
                    "returncode": r.returncode,
                    "stderr": (r.stderr or "")[-800:]}
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    if not data.get("ok"):
        return data
    return _cross_checks(data, tol)


def _cross_checks(data, tol):
    """把"读进来了但是空的"和"拓扑与三角化对不上"挡在这里。

    这两条都不是理论上的担心：STEP 里可以有面而网格化失败（退化面、
    微小面），那时下游看到的是一堆空三角面，任何干涉检查都会说"没撞"。
    所以这里宁可判失败。
    """
    counts = data.get("counts") or {}
    tris = counts.get("triangles", 0)
    faces = counts.get("faces", 0)
    problems = []
    if faces and not tris:
        problems.append("有 %d 个面但三角面为 0：网格化失败或面全退化" % faces)
    if not data.get("coords"):
        problems.append("三角面顶点为空")
    if data.get("bbox") is None:
        problems.append("包围盒为空")
    if data.get("volume") is not None and data["volume"] < 0:
        problems.append("体积为负：法向朝里，几何方向反了")
    data["problems"] = problems
    data["ok"] = not problems
    if problems:
        data["why"] = "; ".join(problems)
    data["tol"] = tol
    return data


def summary(part):
    """一行话，给 CLI 和日志用。"""
    if not part.get("ok"):
        return "读失败：%s" % part.get("why", "未知")
    c = part["counts"]
    b = part["bbox"]
    return ("%s: %d 实体 %d 面 %d 三角面，体积 %.3f，包围盒 %.1f x %.1f x %.1f"
            % (part.get("source", "?"), c.get("solids", 0), c.get("faces", 0),
               c.get("triangles", 0), part.get("volume", 0.0),
               b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2]))


def make_sample(out_path, with_hole=True):
    """造一个已知尺寸的样板件，用来给这一层做验收。

    尺寸是刻意选的：50 x 30 x 10 的块，中心挖一个 R8 通孔。
    这样体积有解析值 50*30*10 - pi*8^2*10 = 15000 - 2010.6 = 12989.4，
    面积、包围盒、拓扑计数也都能手算。验收用一个"我说得出正确值"的零件，
    而不是随便一个下载来的模型 —— 后者只能验证"没崩"。
    """
    py = find_interpreter()
    if py is None:
        return {"ok": False, "why": "no_opencascade"}
    fd, script = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    code = r"""
import sys
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Trsf, gp_Vec
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
box = BRepPrimAPI_MakeBox(50.0, 30.0, 10.0).Shape()
if %r:
    cyl = BRepPrimAPI_MakeCylinder(8.0, 12.0).Shape()
    tr = gp_Trsf(); tr.SetTranslation(gp_Vec(25, 15, -1))
    cyl = BRepBuilderAPI_Transform(cyl, tr, True).Shape()
    box = BRepAlgoAPI_Cut(box, cyl).Shape()
w = STEPControl_Writer(); w.Transfer(box, STEPControl_AsIs); w.Write(%r)
""" % (bool(with_hole), out_path)
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(code)
    try:
        r = subprocess.run([py, script], capture_output=True, text=True,
                           timeout=300)
        if r.returncode != 0:
            return {"ok": False, "why": "sample_build_failed",
                    "stderr": (r.stderr or "")[-600:]}
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass
    return {"ok": True, "path": out_path}


def exact_volume_with_hole():
    """那块样板的解析体积。写成一个函数而不是散落在测试里，
    是为了让"验收用的正确值"只有一处，改样板时不会忘掉。"""
    import math
    return 50.0 * 30.0 * 10.0 - math.pi * 8.0 * 8.0 * 10.0
