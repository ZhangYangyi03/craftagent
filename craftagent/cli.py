"""命令行入口。"""
import argparse, json, os, sys
from . import synth as synth_mod
from .pipeline import run as run_pipeline
from .ask import answer, summary_text
from . import report as report_mod, verify as verify_mod

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

    q = sub.add_parser("ask", help="用中文问一句")
    q.add_argument("question")
    q.add_argument("--report", default="out/report.json")
    q.add_argument("--machine", default=None)

    v = sub.add_parser("verify", help="用合成数据里埋好的真值验收方法本身")
    v.add_argument("--out", default="out/verify.json")
    v.add_argument("--tol", type=int, default=40, help="变点位置容差（分钟）")

    ns = p.parse_args(argv)
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
        rep = run_pipeline(ns.machine, ns.cmm, quality=ns.quality, spec=ns.spec)
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
