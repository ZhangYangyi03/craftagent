# -*- coding: utf-8 -*-
"""occ_boolean.py -- 在另一个解释器里跑 OpenCASCADE 的 B-rep 布尔。

    <有 OCP 的 python> occ_boolean.py <op> <a.step> <b.step> <out.step> [--tol 0.1]

op 是 fuse / cut / common。输出一行 JSON 到 stdout：结果的体积、面积、
面数、实体数，以及是否有效（BRepCheck_Analyzer）。

纪律与 occ_worker.py 一致：这个脚本不做"看起来合理"的修补。布尔失败就
报失败，结果不是有效实体就报 invalid。B-rep 布尔最危险的失败是**静默
地返回一个空壳** —— 下游会读成"没有材料"或"没有干涉"。
"""
import json
import sys


def main(argv):
    if len(argv) < 5:
        print("usage: occ_boolean.py <fuse|cut|common> <a.step> <b.step> <out.step>",
              file=sys.stderr)
        return 2
    op, pa, pb, dst = argv[1], argv[2], argv[3], argv[4]
    tol = 1e-6
    if "--tol" in argv:
        tol = float(argv[argv.index("--tol") + 1])
    try:
        from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCP.BRepAlgoAPI import (BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut,
                                     BRepAlgoAPI_Common)
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    except Exception as exc:                                # pragma: no cover
        print(json.dumps({"ok": False, "why": "no_opencascade",
                          "detail": str(exc)}))
        return 0

    def read(path):
        r = STEPControl_Reader()
        st = r.ReadFile(path)
        if st != 1:
            raise RuntimeError("读不进 %s (status %s)" % (path, st))
        r.TransferRoots()
        sh = r.OneShape()
        if sh.IsNull():
            raise RuntimeError("%s 读进来是空的" % path)
        return sh

    try:
        a = read(pa)
        b = read(pb)
    except Exception as exc:                                # pragma: no cover
        print(json.dumps({"ok": False, "why": "read_failed", "detail": str(exc)}))
        return 0

    try:
        if op == "fuse":
            alg = BRepAlgoAPI_Fuse(a, b)
        elif op == "cut":
            alg = BRepAlgoAPI_Cut(a, b)
        elif op == "common":
            alg = BRepAlgoAPI_Common(a, b)
        else:
            print(json.dumps({"ok": False, "why": "bad_op", "detail": op}))
            return 0
        alg.SetFuzzyValue(tol)
        alg.Build()
        if not alg.IsDone():
            print(json.dumps({"ok": False, "why": "boolean_failed", "op": op}))
            return 0
        res = alg.Shape()
    except Exception as exc:                                # pragma: no cover
        print(json.dumps({"ok": False, "why": "boolean_exception",
                          "detail": str(exc)}))
        return 0

    if res.IsNull():
        print(json.dumps({"ok": False, "why": "null_result", "op": op}))
        return 0

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(res, props)
    vol = props.Mass()
    props2 = GProp_GProps()
    BRepGProp.SurfaceProperties_s(res, props2)
    area = props2.Mass()

    def count(kind):
        ex = TopExp_Explorer(res, kind)
        n = 0
        while ex.More():
            n += 1
            ex.Next()
        return n

    valid = bool(BRepCheck_Analyzer(res).IsValid())
    try:
        w = STEPControl_Writer()
        w.Transfer(res, STEPControl_AsIs)
        w.Write(dst)
    except Exception as exc:                                # pragma: no cover
        print(json.dumps({"ok": False, "why": "write_failed", "detail": str(exc),
                          "volume": vol}))
        return 0

    print(json.dumps({"ok": True, "op": op, "volume": vol, "area": area,
                      "solids": count(TopAbs_SOLID), "faces": count(TopAbs_FACE),
                      "valid": valid, "out": dst}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
