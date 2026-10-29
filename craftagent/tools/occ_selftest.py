# -*- coding: utf-8 -*-
"""occ_selftest.py -- 在另一个解释器里做 OCC 布尔的数值自检。

    <有 OCP 的 python> occ_selftest.py

两个 10x10x10 的长方体，B 相对 A 在三个方向各偏移 5。于是：

    fuse       10*10*10 + 10*10*10 - 5*5*5 = 1875
    common     5*5*5                        = 125
    cut        1000 - 125                   = 875

这三个数有闭式，所以判据在 OCC 之外。输出一行 JSON：三个体积、相对误差、
结果是否通过 BRepCheck_Analyzer。

为什么单独一个脚本而不是塞进 occ_boolean.py：那个脚本的接口是"两个 STEP
文件进、一个 STEP 文件出"，它不该自己造零件。这个脚本负责造零件和判据，
两者分开之后，"读文件那条路"和"布尔本身"可以各自被怀疑。
"""
import json
import sys


def main():
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.BRepAlgoAPI import (BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut,
                                     BRepAlgoAPI_Common)
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.gp import gp_Pnt
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID
    except Exception as exc:                                # pragma: no cover
        print(json.dumps({"ok": False, "why": "no_opencascade",
                          "detail": str(exc)}))
        return 0

    def box(x, y, z, d=10.0):
        return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), d, d, d).Shape()

    def volume(sh):
        p = GProp_GProps()
        BRepGProp.VolumeProperties_s(sh, p)
        return p.Mass()

    def solids(sh):
        n = 0
        e = TopExp_Explorer(sh, TopAbs_SOLID)
        while e.More():
            n += 1
            e.Next()
        return n

    a = box(0.0, 0.0, 0.0)
    b = box(5.0, 5.0, 5.0)
    want = {"fuse": 1875.0, "common": 125.0, "cut": 875.0}
    ops = {"fuse": BRepAlgoAPI_Fuse, "common": BRepAlgoAPI_Common,
           "cut": BRepAlgoAPI_Cut}
    rows = {}
    all_ok = True
    for name, cls in ops.items():
        try:
            algo = cls(a, b)
            algo.Build()
            if not algo.IsDone():
                rows[name] = {"ok": False, "why": "IsDone() false"}
                all_ok = False
                continue
            sh = algo.Shape()
            v = volume(sh)
            rel = abs(v - want[name]) / want[name]
            valid = bool(BRepCheck_Analyzer(sh).IsValid())
            good = rel < 1e-9 and valid and solids(sh) >= 1
            rows[name] = {"want": want[name], "got": v, "rel": rel,
                          "valid": valid, "solids": solids(sh), "ok": good}
            all_ok = all_ok and good
        except Exception as exc:                            # pragma: no cover
            rows[name] = {"ok": False, "why": repr(exc)[:200]}
            all_ok = False
    print(json.dumps({"ok": all_ok, "ops": rows,
                      "note": "两个 10 立方体偏移 5；三个体积都有闭式"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
