# -*- coding: utf-8 -*-
"""check_cad_ran.py -- 在 CI 里断言 CAD 那一组**真的跑了**。

tests/test_cad.py 在没有 OpenCASCADE 时会 skip，这是有意的：本地没装不能
算失败。但在装了 OCP 的 CI job 里，skip 就意味着这一层根本没过验收，
而 `unittest` 对 skip 的退出码和通过一样，都是 0。

所以这个脚本存在的唯一理由是：把"跳过 6 条"和"通过 6 条"区分开。
"""
import os
import sys

# Windows runner 的 stdout 是 cp1252，一句中文诊断就能把它撞成
# UnicodeEncodeError —— 于是"检查有没有 OCP"这个步骤失败，而失败原因
# 和 OCP 一点关系都没有。这一行是给那件事写的，不是为了好看。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from craftagent import cad  # noqa: E402


def main():
    st = cad.backend_status()
    if not st.get("available"):
        print("CAD 后端不可用：%s" % st.get("why"))
        print("这个 job 装了 cadquery-ocp，所以不可用本身就是失败。")
        return 1
    print("CAD 后端：%s" % st["python"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
