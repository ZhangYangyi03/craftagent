# -*- coding: utf-8 -*-
"""CI 用的两个断言，写成文件而不是命令行内联。

为什么不用 `python -c "..."`：GitHub 的 windows runner 跑的是 PowerShell，
内联的字符串要在 PowerShell、cmd、Python 三层引号之间活着，出问题的样子是
"命令找不到"或静默截断，而不是一个能读的错误。这一步在 CI 上真的红过一次，
本地复现不出来——所以把它变成文件，命令行里只剩一个路径。

    python scripts/check_outputs.py identify
    python scripts/check_outputs.py twin
"""
import json
import sys


def _ascii_stdout():
    """Windows runner 的控制台是 cp1252，打一行中文就 UnicodeEncodeError。

    这个仓的 CLI 早就为这件事写过一段（commit c9a6b31），这里又踩了一次：
    上一次这个步骤红是因为引号，这一次是因为中文——两次看起来都像"断言失败"。
    所以输出一律用 ASCII，同时把 stdout 兜底成 utf-8。
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def check_identify(path="out/identify.json"):
    d = json.load(open(path, encoding="utf-8"))
    assert d.get("machines"), "identify produced no machines"
    assert d.get("needs"), "NEEDS must not be empty: unidentifiable params must be listed"
    for mid, m in d["machines"].items():
        for ch, r in m["time_constants"].items():
            # 要么给出 tau，要么给出为什么给不出。两者都没有就是编数。
            assert r.get("identifiable") or r.get("why"), (mid, ch)
    print("ok identify:", ", ".join(sorted(d["machines"])))


def check_twin(path="out/twin.json"):
    d = json.load(open(path, encoding="utf-8"))
    ms = d["machines"]
    assert ms, "twin produced no machines"
    # 健康机：装进去的通道必须全部赢过整班恒定基线
    for c, r in ms["M02"]["validation"].items():
        assert r["beats_baseline"], "M02 channel %s did not beat the baseline and must not be installed" % c
    assert len(ms["M02"]["installed"]) >= 4, ms["M02"]["installed"].keys()
    # 在漂的机：必须有通道被自己剔掉，并写明原因
    assert ms["M01"]["rejected"], "M01 is drifting: the twin must reject channels, not pretend"
    for c, why in ms["M01"]["rejected"].items():
        assert why and len(why) > 20, (c, why)
    # 质量映射必须带留出检验的数
    for c, m in ms["M01"]["symptom_map"].items():
        assert m["r2_holdout"] > 0.5, (c, m["r2_holdout"])
    print("ok twin: M02 all channels beat baseline; M01 rejected:",
          ", ".join(sorted(ms["M01"]["rejected"])))


if __name__ == "__main__":
    _ascii_stdout()
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "identify":
        check_identify()
    elif what == "twin":
        check_twin()
    else:
        sys.exit("usage: python scripts/check_outputs.py identify|twin")
