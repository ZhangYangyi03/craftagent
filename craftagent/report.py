"""把报告渲染成给人看的中文。厂房里那台电脑不会装 jupyter。"""
def render(rep, md=True):
    prof = rep.get("profile", {})
    sig = prof.get("signals", {})
    lab = lambda c: (sig.get(c) or {}).get("label", c)
    L = []
    A = L.append
    A("# %s  工艺异常归因报告" % prof.get("process", "工艺"))
    A("")
    A("质量指标 %s，规格 %s %s。数据 %s（工况 %d 条 / 抽检 %d 条）。"
      % (rep["quality_channel"], rep["spec"], rep["spec_unit"],
         rep["machine_log"].split("\\")[-1], sum(m["n_machine"] for m in rep["machines"].values()),
         sum(m["n_cmm"] for m in rep["machines"].values())))
    A("")
    v = rep["verdict"]
    A("## 结论")
    A("")
    A("- 在漂的机台：%s" % ("、".join(v["drifting"]) or "无"))
    A("- 正常的机台：%s" % ("、".join(v["stable"]) or "无"))
    A("")
    for mid, m in rep["machines"].items():
        o = m["out_of_spec"]; cps = m["change_points"].get(rep["quality_channel"]) or []
        A("### %s" % mid)
        A("")
        A("- 超差 %d / %d 件（%s）" % (o["count"], o["total"], "%.1f%%" % (o["ratio"] * 100)))
        if o["first_ts"]:
            A("- 首件超差时间 %s；最集中时段 %s" % (o["first_ts"], o["cluster_ts"]))
        if cps:
            A("- 质量变点 %s（效应量 %.1f 倍段内波动）" % (cps[0]["ts"], cps[0]["score"]))
        if not cps:
            A("- 样本期内没有检出质量变点")
            A("")
            continue
        t = m["trend"]
        if t:
            A("- 变点后圆度以 %.2f um/小时 上升；最近 2 小时中位数 %.2f um%s"
              % (t["slope_per_hour"], t["last_median"],
                 ("；已经越过规格线" if t.get("already_over") else "")))
        A("")
        A("谁先动（按领先质量的时间排序）")
        A("")
        A("    信号          领先分钟   相关系数   变点后变化")
        for c in m["attribution"]["candidates"]:
            if c["below_threshold"]:
                continue
            A("    %-12s  %+8d   %7.3f   %+8.2f" % (lab(c["channel"]), c["lead_lag_min"],
                                                   c["corr"], c["effect_after_anchor"] or 0.0))
        A("")
        syn = m.get("syndromes") or []
        if syn:
            s = syn[0]
            A("**匹配到已知症候：%s**" % s["name"])
            A("")
            A("- 现场动作：%s" % s["action"])
            if s.get("master_note"):
                A("- %s" % s["master_note"])
        else:
            A("**没有匹配到档案里已知的症候。** 把这次的证据补进 profile.json，下次就能自动认出。")
        A("")
        ad = m.get("adjust")
        if ad:
            role = ad.get("role", "symptom")
            if role == "knob":
                A("- 可调量参考：%s 大约需要%s %.2f %s（%s → %s）。"
                  % (lab(ad["channel"]), ad["direction"], ad["delta"], ad["unit"],
                     ad["current"], ad["target"]))
            elif role == "symptom":
                A("- 要让圆度回到规格线，%s 这个症状需要%s到约 %.2f %s（现在是 %.2f）。"
                  % (lab(ad["channel"]), ad["direction"], ad["target"], ad["unit"], ad["current"]))
            else:
                A("- %s 是环境读数量，不是旋钮——它动了说明它上游的某个量动了，要查的是那个旋钮。" % lab(ad["channel"]))
            A("  %s" % ad["basis"])
        if not m.get("knob_channels_present"):
            A("- 本批数据里没有记录可调旋钮（进给 / 转速 / 修整量），所以给不出"
              "\"参数改成多少\"。要给出调整量，得把 CNC 的设定参数一并采集进来——"
              "这是第一个客户现场第一件要做的事。")
        A("")
    A("---")
    A("方法：%s" % rep["method"])
    A("")
    A("注意：本报告给的是相关性排序，不是因果证明；调整量是线性外推，只当量级参考。"
      "落地前必须在机上小步试，一次只动一个量，动完复测首件。")
    return "\n".join(L)
