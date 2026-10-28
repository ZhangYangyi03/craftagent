# -*- coding: utf-8 -*-
"""跑完整个矩阵。独立进程，写 JSONL 进度，主对话只负责轮询。

被上游掐断在这台机器上是常态，所以每一轮结束就落盘一行，
不攒到最后。重跑时已有的 (model, round) 会跳过。
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["GLM-5.3", "GLM-5.3-Flash", "GLM-5.2", "DeepSeek-V4.1-Flash",
          "DeepSeek-V4-Pro", "Kimi-K2.7-Code", "Qwen3.7-Max"]
ROUNDS = 3
LOG = os.path.join(HERE, "progress.log")


def done():
    got = set()
    for fn in os.listdir(HERE):
        if fn.startswith("result_") and fn.endswith(".json"):
            for r in json.load(open(os.path.join(HERE, fn), encoding="utf-8")):
                if "round" in r and "api_error" not in r:
                    got.add((fn[7:-5], r["round"]))
    return got


def main():
    have = done()
    for model in MODELS:
        for rnd in range(ROUNDS):
            tag = model.replace(".", "").replace("-", "")
            if (tag, rnd) in have:
                print("skip", model, rnd, flush=True)
                continue
            t = time.time()
            r = subprocess.run([sys.executable,
                                os.path.join(HERE, "run_bench.py"),
                                model, "1"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=3600)
            line = "%s round=%d %.0fs rc=%d %s" % (
                model, rnd, time.time() - t, r.returncode,
                (r.stdout or "").strip().splitlines()[-1][:200] if r.stdout else "")
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            print(line, flush=True)


if __name__ == "__main__":
    main()
