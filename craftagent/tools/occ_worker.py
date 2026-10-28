# -*- coding: utf-8 -*-
"""occ_worker.py -- 在另一个解释器里跑 OpenCASCADE，把 STEP/IGES 读成三角面。

为什么要在别的进程里跑：craftagent 的核心是纯标准库、无依赖，
CI 在四个平台上装不了 OpenCASCADE。所以 CAD 这一层做成**可选适配器**：
有 OCP 就用，没有就明确说没有，而不是让 import 在 CI 上炸掉。

调用方式（cad.py 负责）：
    <有 OCP 的 python> occ_worker.py <in.step> <out.json> [--tol 0.5]

输出 JSON 里每一条都是 OCC 算出来的，没有一条是我估的。
三角化用 BRepMesh_IncrementalMesh，线性挠度 tol、角度挠度 tol。

纪律：这个脚本不做任何"看起来合理"的修补。三角化失败就报失败，
面数对不上就报对不上。CAD 这一层最容易出的错是"读进来了，但是空的"，
而空的零件在下游会表现为"没有干涉"，那是安全方向上的假阴性。
"""
import json
import os
import sys


def main(argv):
    if len(argv) < 3:
        print("usage: occ_worker.py <in.step|in.iges> <out.json> [--tol 0.5]",
              file=sys.stderr)
        return 2
    src, dst = argv[1], argv[2]
    tol = 0.5
    if "--tol" in argv:
        tol = float(argv[argv.index("--tol") + 1])

    try:
        from OCP.TopoDS import TopoDS
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import (TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX,
                                TopAbs_SOLID, TopAbs_SHELL)
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IGESControl import IGESControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
    except Exception as exc:                       # noqa: BLE001
        json.dump({"ok": False, "why": "no_opencascade", "detail": str(exc)},
                  open(dst, "w", encoding="utf-8"))
        return 3

    ext = os.path.splitext(src)[1].lower()
    rd = STEPControl_Reader() if ext in (".step", ".stp") else IGESControl_Reader()
    status = rd.ReadFile(src)
    if status != IFSelect_RetDone:
        json.dump({"ok": False, "why": "read_failed", "status": int(status)},
                  open(dst, "w", encoding="utf-8"))
        return 4
    rd.TransferRoots()
    shape = rd.OneShape()

    def count(kind):
        ex = TopExp_Explorer(shape, kind)
        n = 0
        while ex.More():
            n += 1
            ex.Next()
        return n

    volumes = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volumes)
    areas = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, areas)

    BRepMesh_IncrementalMesh(shape, tol, False, tol, True)

    coords = []          # 扁平 [x,y,z, ...]
    tris = []            # 扁平 [i0,i1,i2, ...] 指向 coords
    per_face = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    while ex.More():
        face = TopoDS.Face_s(ex.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        nf = 0
        if tri is not None:
            base = len(coords) // 3
            trsf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                p = p.Transformed(trsf)
                coords.extend((p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                n1, n2, n3 = t.Get()
                tris.extend((base + n1 - 1, base + n2 - 1, base + n3 - 1))
                nf += 1
        per_face.append({"id": face_id, "triangles": nf,
                         "area": _face_area(face, BRepGProp, GProp_GProps)})
        face_id += 1
        ex.Next()

    xs = coords[0::3]
    ys = coords[1::3]
    zs = coords[2::3]
    out = {
        "ok": True,
        "source": os.path.basename(src),
        "tol": tol,
        "counts": {"solids": count(TopAbs_SOLID), "faces": count(TopAbs_FACE),
                   "edges": count(TopAbs_EDGE), "vertices": count(TopAbs_VERTEX),
                   "shells": count(TopAbs_SHELL), "triangles": len(tris) // 3},
        "volume": volumes.Mass(),
        "area": areas.Mass(),
        "bbox": [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]
                if coords else None,
        "coords": coords,
        "tris": tris,
        "per_face": per_face,
    }
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return 0


def _face_area(face, BRepGProp, GProp_GProps):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return props.Mass()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
