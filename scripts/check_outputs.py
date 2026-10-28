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


def check_identify(path="out/identify.json"):
    d = json.load(open(path, encoding="utf-8"))
    assert d.get("machines"), "identify 没有产出任何机器"
    assert d.get("needs"), "NEEDS 清单不能是空的：认不出来的参数必须列出来"
    for mid, m in d["machines"].items():
        for ch, r in m["time_constants"].items():
            # 要么给出 tau，要么给出为什么给不出。两者都没有就是编数。
            assert r.get("identifiable") or r.get("why"), (mid, ch)
    print("ok identify:", ", ".join(sorted(d["machines"])))


def check_twin(path="out/twin.json"):
    d = json.load(open(path, encoding="utf-8"))
    ms = d["machines"]
    assert ms, "twin 没有产出任何机器"
    # 健康机：装进去的通道必须全部赢过整班恒定基线
    for c, r in ms["M02"]["validation"].items():
        assert r["beats_baseline"], "M02 的 %s 没赢基线，不该装" % c
    assert len(ms["M02"]["installed"]) >= 4, ms["M02"]["installed"].keys()
    # 在漂的机：必须有通道被自己剔掉，并写明原因
    assert ms["M01"]["rejected"], "M01 在漂，孪生必须剔掉通道而不是假装能用"
    for c, why in ms["M01"]["rejected"].items():
        assert why and len(why) > 20, (c, why)
    # 质量映射必须带留出检验的数
    for c, m in ms["M01"]["symptom_map"].items():
        assert m["r2_holdout"] > 0.5, (c, m["r2_holdout"])
    print("ok twin: M02 全过, M01 剔掉", ", ".join(sorted(ms["M01"]["rejected"])))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "identify":
        check_identify()
    elif what == "twin":
        check_twin()
    else:
        sys.exit("用法: python scripts/check_outputs.py identify|twin")
