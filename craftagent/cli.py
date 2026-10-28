"""命令行入口。"""
import argparse, json, os, sys
from . import synth as synth_mod
from .pipeline import run as run_pipeline
from .ask import answer, summary_text
from . import report as report_mod, verify as verify_mod
from . import annotation as annot_mod, grader as grader_mod, fieldwork as fw_mod
from . import cnc as cnc_mod, gate_cnc as gate_mod
from . import drawing as drawing_mod, cam as cam_mod, feeds as feeds_mod
from . import collide as collide_mod
from . import ident as ident_mod
from . import twin as twin_mod

def _ensure_utf8_stdout():
    """中文报告要能在 cp1252 控制台上打出来。

    CI 在 windows-latest 上就是这样炸的：第一行「按师傅的边界裁段重算」还没
    写完就 UnicodeEncodeError。控制台代码页是历史包袱，不该让程序本身死掉。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    _ensure_utf8_stdout()
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

    idp = sub.add_parser("identify",
                         help="从实测日志辨识数字孪生参数（时间常数 / 漂移率 / 症状-质量增益）")
    idp.add_argument("--machine", default="data/machine_log.csv")
    idp.add_argument("--cmm", default="data/cmm_report.csv")
    idp.add_argument("--cycle", type=int, default=480, help="班次长度（分钟），要能整除 1440")
    idp.add_argument("--out", default="out/identify.json")
    idp.add_argument("--md", default="out/identify.md")

    twp = sub.add_parser("twin",
                         help="把辨识结果装成孪生，并当场验收：赢不了整班恒定基线的通道剔掉")
    twp.add_argument("--machine", default="data/machine_log.csv")
    twp.add_argument("--cmm", default="data/cmm_report.csv")
    twp.add_argument("--cycle", type=int, default=480)
    twp.add_argument("--min-improvement", type=float, default=0.05)
    twp.add_argument("--out", default="out/twin.json")
    twp.add_argument("--md", default="out/twin.md")

    hd = sub.add_parser("hard",
                        help="高难度参数体检：颤振禁区 / 挠度 / 热变形 / 残留高度（每条标 grade）")
    hd.add_argument("--ap", type=float, default=2.0, help="轴向切深 mm")
    hd.add_argument("--fn", type=float, default=600.0, help="主模态固有频率 Hz（必须实测）")
    hd.add_argument("--k", type=float, default=5.0e4, help="刚度 N/mm（必须实测）")
    hd.add_argument("--zeta", type=float, default=0.025, help="阻尼比（必须实测）")
    hd.add_argument("--kt", type=float, default=700.0, help="切削力系数 N/mm^2")
    hd.add_argument("--teeth", type=int, default=4)
    hd.add_argument("--diameter", type=float, default=10.0, help="刀具直径 mm")
    hd.add_argument("--stickout", type=float, default=40.0, help="伸出长度 mm")
    hd.add_argument("--out", default=None, help="把完整结果写成 json")

    sds = sub.add_parser("standards",
                         help="标准库：公差用公式重算，并把与手册的差一起印出来")
    sds.add_argument("--size", type=float, nargs="+", default=[10, 30, 50, 100])
    sds.add_argument("--grades", type=int, nargs="+", default=[6, 7, 8, 9, 10])

    s3 = sub.add_parser("supply",
                        help="两档参数怎么被供上：采样清单 / 标准引用 / 试切表（这是路线图，不是数据）")
    s3.add_argument("--out", default="out/supply.json")

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

    c = sub.add_parser("cnc", help="G 代码 -> 内核 -> 门 -> 虚拟机床 -> 归因（端到端）")
    csub = c.add_subparsers(dest="cnc_cmd", required=True)
    ca_ = csub.add_parser("analyse", help="一条 .nc 程序跑完整链")
    ca_.add_argument("program", help="G 代码文件路径")
    ca_.add_argument("--spec", type=float, default=8.5)
    ca_.add_argument("--out", default="out/cnc_report.json")
    ca_.add_argument("--gate-off", action="store_true", help="关掉安全门，直接规划")
    ca_.add_argument("--no-c", action="store_true", help="不跑 C 内核对拍")
    cv = csub.add_parser("verify", help="拿真值验收这条链，并跑 C/Python 对拍")
    cv.add_argument("--cases", type=int, default=24)
    cv.add_argument("--out", default="out/cnc_verify.json")
    cg = csub.add_parser("gate", help="只跑安全门，报放行/拒绝和理由")
    cg.add_argument("program")
    cb = csub.add_parser("bench", help="安全门的两个数 + 每条规则的消融对照")
    cb.add_argument("--out", default="out/bench_cnc.json")

    # ---- 图纸 -> 参数 -> G 代码 -> 门（这条链的上游）
    cm = csub.add_parser("cam", help="DXF 图纸 -> 查表定参数 -> G 代码 -> 过门")
    cm.add_argument("drawing", help="DXF 图纸路径")
    cm.add_argument("--material", default="P",
                    help="材料组：P 钢 / M 不锈钢 / K 铸铁 / N 铝 / S 钛")
    cm.add_argument("--op", default="rough", choices=list(feeds_mod.OPERATIONS))
    cm.add_argument("--diameter", type=float, default=6.0)
    cm.add_argument("--flutes", type=int, default=3)
    cm.add_argument("--tool-material", default="carbide",
                    choices=list(feeds_mod.TOOL_MATERIALS))
    cm.add_argument("--ap", type=float, default=1.5, help="每刀轴向切深 mm")
    cm.add_argument("--conservative", type=float, default=1.0,
                    help="乘到每齿进给上的保守系数（<1 更慢）")
    cm.add_argument("--stock", default=None,
                    help="毛坯 x0;y0;x1;y1;top_z;thickness（用分号，负号开头会被 "
                         "argparse 当成选项）；不给就按轮廓包围盒推")
    cm.add_argument("--offset", default="tool_radius_minus_allow",
                    choices=('none', 'tool_radius', 'tool_radius_minus_allow'),
                    help="刀心偏置。默认让刀心离开图纸线一个半径再留 0.3 留量，"
                         "所以切出来的件是图纸尺寸减 0.3（留给精铣）。"
                         "选 none 意味着刀心走在图纸线上，件会小一个半径。")
    cm.add_argument("--finish-allow", type=float, default=0.3,
                    help="留量 mm，只在 offset=tool_radius_minus_allow 下生效")
    cm.add_argument("--ramp-deg", type=float, default=5.0,
                    help="下刀斜插角度。0 = 直插（钻孔用）；铣削用 3~10")
    cm.add_argument("--out-nc", default="out/cam.nc")
    cm.add_argument("--out", default="out/cam.json")

    cv2 = csub.add_parser("cam-verify", help="图纸->参数->G代码->门 这条链的验收（出数）")
    cv2.add_argument("--out", default="out/cam_verify.json")

    cf = csub.add_parser("feeds", help="打印切削参数手册，或查一档参数")
    cf.add_argument("--material", default=None)
    cf.add_argument("--op", default="rough", choices=list(feeds_mod.OPERATIONS))
    cf.add_argument("--diameter", type=float, default=10.0)
    cf.add_argument("--flutes", type=int, default=4)
    cf.add_argument("--tool-material", default="carbide",
                    choices=list(feeds_mod.TOOL_MATERIALS))
    cf.add_argument("--ap", type=float, default=1.0)

    ns = p.parse_args(argv)
    if ns.cmd == "supply":
        from . import supply as sp
        rows = sp.route_table()
        print("两档参数怎么被供上：每一小块钉到一条路，而不是钉到一个数")
        print()
        print("%-26s %-11s %s" % ("项目", "走哪条路", "凭什么"))
        for r in rows:
            print("%-26s %-11s %s" % (r["item"], r["route"], r["how"]))
        print()
        print("第三档（必须实测）要采什么：")
        for k, v in sp.MEASURE_PLAN.items():
            print("  %s" % v["what"])
            print("    怎么做：%s" % v["how"])
            print("    拒绝：%s" % v["refuse"])
        if ns.out:
            import json as _json
            os.makedirs(os.path.dirname(ns.out) or ".", exist_ok=True)
            with open(ns.out, "w", encoding="utf-8") as fh:
                _json.dump({"routes": rows, "measure": sp.MEASURE_PLAN}, fh,
                           indent=2, ensure_ascii=False)
            print("\n写到 %s" % ns.out)
        return 0

    if ns.cmd == "hard":
        from . import hard as hard_mod
        rep = hard_mod.report(ap_mm=ns.ap, fn_hz=ns.fn, k_n_mm=ns.k, zeta=ns.zeta,
                              kt=ns.kt, teeth=ns.teeth,
                              d_tool_mm=ns.diameter, stickout_mm=ns.stickout)
        print("高难度参数体检   ap=%.2f mm  D=%.0f mm  伸出=%.0f mm"
              % (ns.ap, ns.diameter, ns.stickout))
        print()
        print("一、无条件稳定切深 a_min = %.3f mm（解析式，任何转速都不能超）"
              % rep["a_min_mm"])
        st = rep["stability"]
        print("    网格扫出的最坏点         %.3f mm @ %d rpm（与解析式差 %.1f%%）"
              % (st["worst"]["a_lim_mm"], st["worst"]["rpm"],
                 100.0 * abs(st["worst"]["a_lim_mm"] / rep["a_min_mm"] - 1.0)))
        print("    说明：%s" % st["note"])
        av = rep["avoid"]
        if av["avoid"]:
            print("二、ap=%.2f mm 时不能碰的转速段（%d 段，占扫过范围的 %.1f%%）："
                  % (ns.ap, len(av["avoid"]), 100.0 * av["bad_fraction"]))
            for a, b in av["avoid"][:8]:
                print("      %5d ~ %5d rpm" % (a, b))
            if len(av["avoid"]) > 8:
                print("      ...（共 %d 段）" % len(av["avoid"]))
        else:
            print("二、ap=%.2f mm 低于 a_min，扫过的转速范围内都能切" % ns.ap)
        print("三、切削力 %.0f N（均值）-> 刀具挠度 %.1f um"
              % (rep["force"]["force_n"], rep["deflection"]["deflection_mm"] * 1000))
        print("    对照：IT7 在 30-50 mm 段是 25 um —— 让刀量自己就吃掉了这个量级")
        print("四、热变形：500 mm 钢件每 1 K 长 %.2f um（IT7@50 段由 5 K 吃掉）"
              % rep["thermal_1k_500mm_um"])
        print()
        print("哪一档数能信：")
        print("  fn/k/zeta  = 必须实测（随刀柄、伸出、转速变）")
        print("  kt         = 可引用（Kienzle 常用值，出处见 hard.py）")
        print("  上面四条公式 = 可重算（解析解与网格解互相对上，见 tests/test_hard.py）")
        if ns.out:
            import json as _json
            os.makedirs(os.path.dirname(ns.out) or ".", exist_ok=True)
            with open(ns.out, "w", encoding="utf-8") as fh:
                _json.dump(rep, fh, indent=2, ensure_ascii=False)
            print("\n完整结果写到 %s" % ns.out)
        return 0

    if ns.cmd == "standards":
        from . import standards as s_mod
        print("ISO 286 公差：i = 0.45*D^(1/3) + 0.001*D，IT 等级是 i 的整数倍")
        print("公式值 / 取整值 / 手册值 三者并列，差多少由你自己判断该不该信")
        print()
        for size in ns.size:
            print("尺寸 <= %g mm" % size)
            print("  等级    i(um)   公式    取整   手册   差")
            for r in s_mod.tolerance_table(size, tuple(ns.grades)):
                hb = "-" if r["handbook_um"] is None else "%g" % r["handbook_um"]
                dl = "-" if r["delta_um"] is None else "%+.1f" % r["delta_um"]
                warn = "  <-- 表格区，请查表" if not r["formula_applies"] else ""
                print("  IT%-3d %7.3f %8.2f %7.1f %6s %6s%s"
                      % (r["grade"], r["i_um"], r["raw_um"], r["um"], hb, dl, warn))
            print()
        f = s_mod.fit_summary(50, 7, 6)
        print("配合 %s @50mm：孔 %.0f um + 轴 %.0f um => 最大间隙 %.0f um"
              % (f["fit"], f["hole_um"], f["shaft_um"], f["max_clearance_um"]))
        print("  %s" % f["note"])
        return 0

    if ns.cmd == "cnc":
        if ns.cnc_cmd == "feeds":
            if ns.material is None:
                print("%-4s %-16s %6s %8s %8s %8s"
                      % ("组", "材料", "HSS", "carbide", "coated", "kc"))
                for key, label, hss, carb, coat, kc in feeds_mod.table():
                    print("%-4s %-16s %6.0f %8.0f %8.0f %8.0f"
                          % (key, label, hss, carb, coat, kc))
                return 0
            tool = feeds_mod.Tool(ns.diameter, ns.flutes, ns.tool_material)
            ch = feeds_mod.choose(ns.material, ns.op, tool,
                                  feeds_mod.Machine(), ap_mm=ns.ap)
            print("材料 %s / 工序 %s / 刀具 %s"
                  % (ns.material, ns.op, tool.name))
            print("  vc        %.1f m/min" % ch.vc_m_min)
            print("  主轴 S    %.0f rpm%s"
                  % (ch.rpm, ("（表中值 %.0f 被机床上限夹住）" % ch.rpm_raw)
                     if 'rpm_high' in ch.clamped else ""))
            print("  进给 F    %.0f mm/min（每齿 %.3f mm，每转 %.3f mm）"
                  % (ch.feed_mm_min, ch.fz_mm, ch.feed_per_rev_mm))
            print("  切削功率  %.2f kW / 可用 %.2f kW" % (ch.power_kw, ch.power_available_kw))
            print("  单位切削力 kc %.0f N/mm2（切屑 %.3f mm）" % (ch.kc_n_mm2, ch.fz_mm))
            if ch.clamped:
                print("  被夹：%s" % ", ".join(ch.clamped))
            return 0
        if ns.cnc_cmd == "cam-verify":
            from bench import bench_cam
            res = bench_cam.run(out=ns.out)
            return 0 if not res['closed_loop']['still_failing'] else 1
        if ns.cnc_cmd == "cam":
            text = open(ns.drawing, encoding="utf-8").read()
            dr = drawing_mod.read_dxf(text)
            print("图纸 %s：%d 条闭合轮廓，%d 个实体"
                  % (ns.drawing, len(dr.contours), sum(dr.entities_seen.values())))
            if dr.unsupported:
                print("  未采用的实体 %d 个（先说出来，不假装图纸是干净的）："
                      % len(dr.unsupported))
                for etype, layer, why in dr.unsupported[:6]:
                    print("    %-12s 图层 %-12s %s" % (etype, layer, why))
            if dr.unknown_depth:
                print("  图层名里读不出层深的 %d 个：%s"
                      % (len(dr.unknown_depth),
                         ", ".join(sorted({l for l, _ in dr.unknown_depth}))))
            stock = None
            if ns.stock:
                sep = ';' if ';' in ns.stock else ','
                vals = [float(v) for v in ns.stock.split(sep)]
                if len(vals) != 6:
                    print("--stock 要 6 个数：x0;y0;x1;y1;top_z;thickness")
                    return 2
                stock = tuple(vals)
            tool = feeds_mod.Tool(ns.diameter, ns.flutes, ns.tool_material)
            # 默认走闭环：查到被拒就降进给再来。一次性参数常常被门拦下，
            # 而"能用"和"能跑"是两件事——默认输出就该是过得了门的程序。
            plan, attempts = cam_mod.solve(
                dr, ns.material, tool, feeds_mod.Machine(),
                stock=stock, op=ns.op, ap_per_pass=ns.ap,
                conservative=ns.conservative,
                offset=ns.offset, finish_allow=ns.finish_allow,
                ramp_deg=ns.ramp_deg)
            for a in attempts:
                print("  尝试 k=%.3f  %s%s"
                      % (a.conservative, "放行" if a.verdict.ok else "拒绝",
                         "" if a.verdict.ok else "：" + a.reason))
            d = os.path.dirname(ns.out_nc)
            if d: os.makedirs(d, exist_ok=True)
            with open(ns.out_nc, "w", encoding="utf-8") as fh:
                fh.write(plan.text)
            # 门必须用**生成器同一块料**。CLI 这里原来调 check_text 的默认
            # 毛坯（0,0,60,84），于是 build/solve 里已经放行的程序，
            # 在这里又被报一次 clamp_crash——同一个程序两次检查两个答案，
            # 而两次都对，因为它们看的不是同一块料。
            _sx0, _sy0, _sx1, _sy1, _tz, _th = plan.meta['stock']
            _fx = collide_mod.Fixture(stock=collide_mod.Stock(
                _sx0, _sy0, _sx1, _sy1, _tz, _th))
            v = gate_mod.check_text(plan.text, name=ns.out_nc, fixture=_fx)
            print("生成 %s：%d 行，%d 次参数决策、%d 条轮廓"
                  % (ns.out_nc, plan.text.count(chr(10)),
                     len(plan.choices), plan.meta.get('contours', 0)))
            rec = (plan.meta.get('offsets') or [{}])[0]
            print("  刀心偏置 %.2f mm（%s）  下刀 %s  留量 %.2f mm"
                  % (rec.get('dist', 0.0),
                     '近似法向' if rec.get('dist') else '无',
                     ('斜插 %.0f°' % ns.ramp_deg) if ns.ramp_deg > 0 else '直插',
                     ns.finish_allow if ns.offset == 'tool_radius_minus_allow' else 0.0))
            print("门：%s" % ("放行" if v.ok else "拒绝"))
            for rule in v.reasons:
                print("  [%s] %s" % (rule, v.detail.get(rule)))
            rep = {'drawing': dr.summary(), 'plan': plan.meta,
                   'choices': [{'op': c.op, 'rpm': c.rpm,
                                'feed_mm_min': c.feed_mm_min,
                                'fz_mm': c.fz_mm, 'ap_mm': c.ap_mm,
                                'power_kw': c.power_kw,
                                'clamped': c.clamped} for c in plan.choices],
                   'gate': {'ok': v.ok, 'reasons': v.reasons, 'detail': v.detail}}
            d = os.path.dirname(ns.out)
            if d: os.makedirs(d, exist_ok=True)
            json.dump(rep, open(ns.out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            return 0 if v.ok else 1
        if ns.cnc_cmd == "gate":
            text = open(ns.program, encoding="utf-8").read()
            v = gate_mod.check_text(text, name=ns.program)
            print("%s：%s" % (ns.program, "放行" if v.ok else "拒绝"))
            for rule in v.reasons:
                print("  [%s] %s" % (rule, v.detail.get(rule)))
            return 0 if v.ok else 1
        if ns.cnc_cmd == "analyse":
            text = open(ns.program, encoding="utf-8").read()
            rep = cnc_mod.analyse(text, name=ns.program, spec=ns.spec,
                                  gate_on=not ns.gate_off)
            if not rep.get('gate', {}).get('ok', True):
                print("门拒绝：%s" % rep['gate']['detail'])
                return 1
            d = os.path.dirname(ns.out)
            if d: os.makedirs(d, exist_ok=True)
            if not ns.no_c:
                from bench import pair as pair_mod
                pr = pair_mod.compare(text, name=ns.program)
                rep['parity'] = pr
            json.dump(rep, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            q = rep['quality']
            print("程序 %s：%d 段，%.1f mm，%.2f s" % (ns.program, rep['program']['segments'],
                                                    rep['program']['path_mm'],
                                                    rep['program']['t_total_s']))
            print("质量：中位 %.2f um，最大 %.2f um，超差 %d/%d（规格 %.1f）"
                  % (q['median'], q['max'], q['over_spec'], q['n'], q['spec']))
            print("瞬态 %d 次走刀未计入稳态；稳态 %d 次" % (rep['warmup'], q['steady_n']))
            if rep['change_points']:
                for cp in rep['change_points']:
                    print("变点：第 %d 次走刀（G 代码第 %d 行），分数 %.1f"
                          % (cp['index'], cp['line'], cp['score']))
            else:
                print("变点：无（稳态段内没有显著变化）")
            if rep['parity'] and rep['parity'].get('ok') is not None:
                pr = rep['parity']
                print("对拍：%s  max|dv|=%.2e mm/s  max|dt|=%.2e s"
                      % ("一致" if pr['ok'] else "不一致", pr['max_dv_mm_s'], pr['max_dt_s']))
            print("归因（按效应量）：%s" % " > ".join(rep['attribution']['by_effect']))
            print("报告写入 %s" % ns.out)
            return 0
        if ns.cnc_cmd == "verify":
            v = cnc_mod.verify(n_cases=ns.cases)
            from bench import pair as pair_mod
            pr = pair_mod.run_all(verbose=False)
            d = os.path.dirname(ns.out)
            if d: os.makedirs(d, exist_ok=True)
            json.dump({'verify': v, 'parity': pr}, open(ns.out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            all_ok = all(g['passed'] for g in v.values()) and pr['all_ok']
            print("验收：%s" % ("全过" if all_ok else "有不过的"))
            for gname, g in v.items():
                lbl = "门开（事前拦截）" if g['gate_on'] else "门关（事后归因）"
                print(" %s：%s" % (lbl, "过" if g['passed'] else "不过"))
                for ck in g['checks']:
                    mk = "不适用" if ck['ok'] is None else ("过" if ck['ok'] else "不过")
                    print("  [%s] %s -- %s" % (mk, ck['name'], ck['detail']))
            for nm, pv in pr.items():
                if isinstance(pv, dict) and 'segments' in pv:
                    print("  [%s] 对拍 %s：%d 段，max|dv|=%.2e，max|dt|=%.2e"
                          % ("过" if pv['ok'] else "不过", nm, pv['segments'],
                             pv['max_dv_mm_s'], pv['max_dt_s']))
            print("明细写入 %s" % ns.out)
            return 0 if all_ok else 1
        if ns.cnc_cmd == "bench":
            from bench import bench_cnc
            r = bench_cnc.run()
            d = os.path.dirname(ns.out)
            if d: os.makedirs(d, exist_ok=True)
            json.dump(r, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print("明细写入 %s" % ns.out)
            return 0 if r['legal']['false_refusal_rate'] == 0 else 1
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
    if ns.cmd == "identify":
        from . import align as align_mod, changepoint as cp_mod
        rows_all = align_mod.load_machine_log(ns.machine)
        cmm, qn = align_mod.load_cmm(ns.cmm)
        rows_by, al_by, cp_by = {}, {}, {}
        for mid in sorted({r["id"] for r in rows_all}):
            rows = [r for r in rows_all if r["id"] == mid]
            rows_by[mid] = rows
            al, _ = align_mod.align(rows, [c for c in cmm if c["id"] == mid])
            al_by[mid] = al
            cps = cp_mod.find_changes([a["quality"] for a in al], min_size=60)
            if cps:
                cp_by[mid] = cps[0][0]
        res = ident_mod.identify(rows_by, al_by, cycle_min=ns.cycle, cp_by_machine=cp_by)
        res["quality_channel"] = qn
        for d in (os.path.dirname(ns.out), os.path.dirname(ns.md)):
            if d:
                os.makedirs(d, exist_ok=True)
        json.dump(res, open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        text = ident_mod.render(res)
        open(ns.md, "w", encoding="utf-8", newline="\n").write(text + "\n")
        print(text)
        print("参数表写入 %s 和 %s" % (ns.out, ns.md))
        return 0
    if ns.cmd == "twin":
        from . import align as align_mod, changepoint as cp_mod
        rows_all = align_mod.load_machine_log(ns.machine)
        cmm, qn = align_mod.load_cmm(ns.cmm)
        rows_by, al_by, cp_by = {}, {}, {}
        for mid in sorted({r["id"] for r in rows_all}):
            rows = [r for r in rows_all if r["id"] == mid]
            rows_by[mid] = rows
            al, _ = align_mod.align(rows, [c for c in cmm if c["id"] == mid])
            al_by[mid] = al
            cps = cp_mod.find_changes([a["quality"] for a in al], min_size=60)
            if cps:
                cp_by[mid] = cps[0][0]
        res = ident_mod.identify(rows_by, al_by, cycle_min=ns.cycle, cp_by_machine=cp_by)
        twins, checks, dump = {}, {}, {}
        for mid in rows_by:
            tw = twin_mod.from_identify(res, mid)
            tw.prune(rows_by[mid], min_improvement=ns.min_improvement)
            twins[mid] = tw
            checks[mid] = tw.validation
            dump[mid] = {"installed": tw.channels, "symptom_map": tw.symptom_map,
                         "rejected": tw.rejected, "not_identifiable": tw.unidentifiable,
                         "validation": tw.validation}
        for d in (os.path.dirname(ns.out), os.path.dirname(ns.md)):
            if d:
                os.makedirs(d, exist_ok=True)
        json.dump({"cycle_min": ns.cycle, "quality_channel": qn, "machines": dump,
                   "needs": res["needs"]},
                  open(ns.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        text = twin_mod.render(twins, checks)
        open(ns.md, "w", encoding="utf-8", newline="\n").write(text + "\n")
        print(text)
        n_ok = sum(1 for m in checks.values() for r in m.values() if r["beats_baseline"])
        n_all = sum(len(m) for m in checks.values())
        print("孪生：%d/%d 条通道通过验收，写入 %s 和 %s" % (n_ok, n_all, ns.out, ns.md))
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
