"""命令行入口。"""
import argparse, json, os, sys
from . import synth as synth_mod
from .pipeline import run as run_pipeline
from .ask import answer, summary_text
from . import report as report_mod, verify as verify_mod
from . import annotation as annot_mod, grader as grader_mod, fieldwork as fw_mod

def main(argv=None):
    p = argparse.ArgumentParser(prog="craftagent", description="精密制造工艺数据 -> 异常归因")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="生成合成演示数据（含埋好的真值）")
    s.add_argument("--out", default="data")
    s.add_argument("--json", action="store_true", help="把真值一并打印出来")

    a = sub.add_parser("analyse", help="对齐 + 变点 + 归因，写出报告")
    a.add_argument("--machine", default="data/machine_log.csv")
    a.add_argument("--cmm", default="data/cmm_report.csv")
    a.add_argument("--spec", type=float, default=8.5)
    a.add_argument("--quality", default=None)
    a.add_argument("--profile", default=None, help="工艺档案 json；不给就用内置的磨削范例")
    a.add_argument("--out", default="out/report.json")
    a.add_argument("--md", default="out/report.md")
    a.add_argument("--warmup-skip", type=int, default=0,
                   help="每个班次开头多少分钟整段丢掉（老师傅说'刚开机头两小时不算'就填 120）")

    q = sub.add_parser("ask", help="用中文问一句")
    q.add_argument("question")
    q.add_argument("--report", default="out/report.json")
    q.add_argument("--machine", default=None)

    v = sub.add_parser("verify", help="用合成数据里埋好的真值验收方法本身")
    v.add_argument("--out", default="out/verify.json")
    v.add_argument("--tol", type=int, default=40, help="变点位置容差（分钟）")

    t = sub.add_parser("template", help="写一条空的六要素标注表出来填")
    t.add_argument("--machine", default="M01")
    t.add_argument("--out", default="annotations/template.json")

    st = sub.add_parser("stamp", help="现场：师傅指认异常的那一刻，记下读数快照和原话")
    st.add_argument("--machine", default="M01")
    st.add_argument("--say", required=True, help="师傅的原话")
    st.add_argument("--master", default="")
    st.add_argument("--report", default="out/report.json")
    st.add_argument("--out-dir", default="annotations")

    g = sub.add_parser("grade", help="用机器算出来的东西，核对师傅的判断（方法恢复率）")
    g.add_argument("--report", default="out/report.json")
    g.add_argument("--annotations", default="annotations/annotations.jsonl")
    g.add_argument("--out", default="out/grade.json")
    g.add_argument("--topn", type=int, default=3)
    g.add_argument("--masked", default="out/report_masked.json",
                   help="按师傅声明的边界裁段后重算的报告；给了才判「边界遵守」那一问")
    g.add_argument("--auto-mask", action="store_true",
                   help="按标注里的 invalid_when 自动裁段重算（预热段默认 120 分钟）")

    ns = p.parse_args(argv)
    if ns.cmd == "template":
        rec = annot_mod.blank(machine=ns.machine)
        d = os.path.dirname(ns.out)
        if d: os.makedirs(d, exist_ok=True)
        json.dump(rec, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("空表写入 %s；六要素是 %s" % (ns.out, "、".join(annot_mod.FIELDS)))
        return 0
    if ns.cmd == "stamp":
        r = fw_mod.stamp(ns.machine, ns.say, ns.report, out_dir=ns.out_dir, master=ns.master)
        print("记下 %s -> %s" % (r["id"], r["path"]))
        for b in r["open_items"]: print("  还没填：%s" % b)
        return 0
    if ns.cmd == "grade":
        rep = json.load(open(ns.report, encoding="utf-8"))
        recs = annot_mod.load(ns.annotations)
        if not recs:
            print("没有标注可评（%s）。先 stamp 一条。" % ns.annotations); return 1
        masked = None
        if ns.auto_mask:
            pk = [r for r in recs if (r.get("boundary") or {}).get("invalid_when")]
            skip = 120 if any("开机" in w or "预热" in w for r in pk for w in r["boundary"]["invalid_when"]) else 0
            if skip:
                masked = run_pipeline(rep["machine_log"], rep["cmm_report"], spec=rep["spec"],
                                      quality=rep["quality_channel"], warmup_skip_min=skip)
                print("按师傅的边界裁段重算：每班次丢掉前 %d 分钟" % skip)
        elif os.path.exists(ns.masked):
            masked = json.load(open(ns.masked, encoding="utf-8"))
        res = grader_mod.grade_all(rep, recs, masked=masked, topn=ns.topn)
        d = os.path.dirname(ns.out)
        if d: os.makedirs(d, exist_ok=True)
        json.dump(res, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        for r in res["records"]:
            print("%s（%s）" % (r.get("machine"), r.get("record")))
            for q in r["questions"]:
                mark = "过" if q["ok"] else ("跳过" if q["ok"] is None else "不过")
                print("  [%s] %s -- %s" % (mark, q["name"], q["detail"]))
        print("恢复率 %s（%d/%d）" % (res["rate"], res["passed"], res["scored"]))
        print("明细写入 %s" % ns.out)
        return 0
    if ns.cmd == "verify":
        info = synth_mod.write(os.path.join(os.path.dirname(ns.out) or ".", "..", "data")
                               if False else os.path.join("data"))
        rep = run_pipeline(os.path.join("data", "machine_log.csv"),
                           os.path.join("data", "cmm_report.csv"), spec=8.5)
        chk = verify_mod.check(rep, info, tol_min=ns.tol)
        d = os.path.dirname(ns.out)
        if d: os.makedirs(d, exist_ok=True)
        json.dump(chk, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("验收：%s" % ("全过" if chk["passed"] else "有不过的"))
        for c in chk["checks"]:
            print("  [%s] %s -- %s" % ("过" if c["ok"] else "不过", c["name"], c["detail"]))
        print("明细写入 %s" % ns.out)
        return 0 if chk["passed"] else 1
    if ns.cmd == "synth":
        info = synth_mod.write(ns.out)
        print("合成数据写入 %s：%d 条工况 / %d 条三坐标（漂移起点 %s）"
              % (ns.out, info["machine"], info["cmm"], info["drift_start"]))
        if ns.json: print(json.dumps(info["truth"], ensure_ascii=False))
        return 0
    if ns.cmd == "analyse":
        rep = run_pipeline(ns.machine, ns.cmm, quality=ns.quality, spec=ns.spec,
                           warmup_skip_min=ns.warmup_skip)
        d = os.path.dirname(ns.out)
        if d: os.makedirs(d, exist_ok=True)
        json.dump(rep, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if ns.md:
            d = os.path.dirname(ns.md)
            if d: os.makedirs(d, exist_ok=True)
            open(ns.md, "w", encoding="utf-8", newline="\n").write(report_mod.render(rep))
        print(summary_text(rep))
        print("报告写入 %s%s" % (ns.out, (" 和 " + ns.md) if ns.md else ""))
        return 0
    if ns.cmd == "ask":
        rep = json.load(open(ns.report, encoding="utf-8"))
        print(answer(ns.question, rep, machine=ns.machine))
        return 0
    return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())
