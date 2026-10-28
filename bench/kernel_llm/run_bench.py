# -*- coding: utf-8 -*-
"""bench_llm/run_bench.py -- 同一道内核题，多个模型，编译后真跑。

为什么是"编译后真跑"而不是看代码像不像：内核这一类代码的失败几乎全都编译得过、
跑得起来，然后给出一个不像样的数（负的刚度、装配错一个自由度、边界条件加错方向）。
所以评分只在**可执行文件的行为**上做：

    compiles  能不能编过
    runs      能不能按命令行跑出 tip_uy=
    sane      比值落在 [0.80, 1.05]（梁理论略高估挠度，有限元应该略低）
    converges 网格细化时比值单调靠近 1

题目给定，CLI 给定，参考答案不给定 —— 给了它就可以把常数写死。
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
GCC = r"C:\msys64\mingw64\bin\gcc.exe"


def beam_reference(L, H, total_y, E=210000.0, nu=0.3):
    """平面应变梁理论端部挠度。E' = E/(1-nu^2)，I = H^3/12（单位厚度）。"""
    Eps = E / (1.0 - nu * nu)
    I = H ** 3 / 12.0
    return total_y * L ** 3 / (3.0 * Eps * I)


def extract_code(text):
    m = re.search(r"```(?:c|c99|C)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def compile_and_run(code, label):
    d = tempfile.mkdtemp(prefix="llmkernel_")
    src = os.path.join(d, label + ".c")
    exe = os.path.join(d, label + ".exe")
    with open(src, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(code)
    out = {"label": label, "lines": code.count("\n") + 1}
    r = subprocess.run([GCC, "-O2", "-std=c99", "-o", exe, src, "-lm"],
                       capture_output=True, text=True, timeout=300)
    out["compiles"] = r.returncode == 0
    if not out["compiles"]:
        out["compiler_errors"] = r.stderr[-600:]
        out["exit_reason"] = "compile_failed"
        return out
    runs = []
    for (L, H, nx, ny, F) in ((100.0, 10.0, 10, 2, 1000.0),
                              (100.0, 10.0, 20, 4, -1000.0),
                              (100.0, 10.0, 40, 8, -1000.0)):
        try:
            rr = subprocess.run([exe, str(L), str(H), str(nx), str(ny), str(F)],
                                capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            runs.append({"mesh": "x".join(map(str, (nx, ny))), "error": "timeout"})
            continue
        txt = rr.stdout.strip()
        m = re.search(r"tip_uy\s*=\s*([-+0-9.eE]+)", txt)
        val = float(m.group(1)) if m else None
        ref = beam_reference(L, H, abs(F))
        runs.append({"mesh": "%dx%d" % (nx, ny), "rc": rr.returncode,
                     "stdout_head": txt[:120], "tip": val,
                     "beam": ref,
                     "ratio": (val / ref) if val else None,
                     "stderr_head": rr.stderr.strip()[:150]})
    out["runs"] = runs
    got = [x for x in runs if x.get("ratio") is not None]
    out["runs_ok"] = len(got)
    if not got:
        out["exit_reason"] = "no_parseable_output"
        return out
    out["ratio_first"] = got[0]["ratio"]
    out["ratio_fine"] = got[-1]["ratio"]
    out["sane"] = 0.80 <= got[-1]["ratio"] <= 1.05
    ratios = [x["ratio"] for x in got]
    out["converges"] = all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:])) \
        if len(ratios) > 1 else None
    out["exit_reason"] = "ok" if out["sane"] else "insane_number"
    return out


def ask(model, task, max_tokens=32000, temperature=0.3, timeout=1800):
    import urllib.request
    AK = os.environ.get("BENCH_API_KEY") or "QC-5afdc3dc6cc9a3c9a17f48cd0722abd5-bc12b1de10d530303f29c4cdeb2fe262"
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": task}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request("https://aiping.cn/api/v1/chat/completions",
                                 data=body,
                                 headers={"Authorization": "Bearer " + AK,
                                          "Content-Type": "application/json",
                                          "User-Agent": "x"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    dt = time.time() - t
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    txt = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = d.get("usage") or {}
    # 推理型模型会在 reasoning_content 里烧掉整个预算、正文一个字没有，
    # 这时"编译失败"看起来像代码写错了，其实是调用方的预算给错了。
    # 所以这两条必须跟着结果一起进去。
    usage["finish_reason"] = ch.get("finish_reason")
    usage["reasoning_chars"] = len(reasoning)
    usage["content_chars"] = len(txt)
    return txt, dt, usage


if __name__ == "__main__":
    model = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    task = open(os.path.join(HERE, "TASK.txt"), encoding="utf-8").read()
    res = []
    for i in range(rounds):
        try:
            txt, dt, usage = ask(model, task)
        except Exception as exc:                                  # noqa: BLE001
            res.append({"round": i, "api_error": str(exc)[:200]})
            continue
        code = extract_code(txt)
        tag = "m%s_r%d" % (re.sub(r"[^A-Za-z0-9]", "", model), i)
        # 原始代码必须留下来：评分只给一个数，而"为什么错"要看代码。
        with open(os.path.join(HERE, tag + ".c"), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(code)
        r = compile_and_run(code, tag)
        r["round"] = i
        r["gen_seconds"] = round(dt, 1)
        r["completion_tokens"] = usage.get("completion_tokens")
        r["finish_reason"] = usage.get("finish_reason")
        r["reasoning_chars"] = usage.get("reasoning_chars")
        r["content_chars"] = usage.get("content_chars")
        r["raw_lines"] = txt.count("\n") + 1
        res.append(r)
        # 每一轮落盘一次：这批生成单轮可能上百秒，被上游掐断是常事，
        # 攒到最后再写等于把已经花掉的钱丢掉。
        fn = os.path.join(HERE, "result_%s.json" % re.sub(r"[^A-Za-z0-9]", "", model))
        old = json.load(open(fn, encoding="utf-8")) if os.path.exists(fn) else []
        json.dump(old + [r], open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "runs"}, ensure_ascii=False))
    fn = os.path.join(HERE, "result_%s.json" % re.sub(r"[^A-Za-z0-9]", "", model))
    print("wrote", fn)
