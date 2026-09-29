"""AutoDL 的 GPU 出口。本机没有 CUDA（gpu_probe 实测），所以重型训练走这里。

设计边界，必须说清楚：
  这里跑的是"训练和调参"，不是"现场推理"。工厂不让工艺数据出厂，所以训完的权重
  要落回本机或现场那台机器上离线跑。远程只用来做那件本机做不了的事。

用法（先设好实例，再从本机 ssh 过去；本机已有 OpenSSH 和 id_ed25519）：
  set CRAFTAGENT_REMOTE=root@region-1.autodl.com -p 12345
  python -m craftagent.remote probe
  python -m craftagent.remote run --script train.py --send data/ --pull out/
"""
import argparse, os, shlex, subprocess, sys

def target():
    t = os.environ.get("CRAFTAGENT_REMOTE", "").strip()
    if not t:
        raise SystemExit(
            "没有配远程实例。先设环境变量 CRAFTAGENT_REMOTE，例如：\n"
            "  set CRAFTAGENT_REMOTE=root@region-1.autodl.com -p 12345\n"
            "AutoDL 实例关机后 IP 和端口都会变，所以别写死在代码里。")
    return t

def sh(argv, timeout=120):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout or "", r.stderr or ""

def cmd_probe():
    t = target()
    rc, out, err = sh(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
                      + shlex.split(t) + ["nvidia-smi -L; python -c 'import torch;print(torch.__version__)'"],
                      timeout=60)
    print("rc", rc); print(out or err)

def cmd_run(ns):
    t = target()
    if ns.send:
        rc, out, err = sh(["scp", "-r"] + shlex.split(t) + [ns.send])
        rc, out, err = sh(["scp", "-r", ns.send, shlex.split(t)[0] + ":" + (ns.dest or "~/work/")], timeout=600)
        print("scp rc", rc, out, err)
    rc, out, err = sh(["ssh"] + shlex.split(t) + ["cd ~/work && python %s" % ns.script], timeout=ns.timeout)
    print("rc", rc); print(out); print(err[-2000:])
    if ns.pull:
        rc, out, err = sh(["scp", "-r", shlex.split(t)[0] + ":" + ns.pull, "."], timeout=600)
        print("pull rc", rc, out, err)

def main(argv=None):
    p = argparse.ArgumentParser(prog="craftagent.remote")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="看实例活着没有、有没有 GPU 和 torch")
    r = sub.add_parser("run", help="把脚本送过去跑，再把结果拉回来")
    r.add_argument("--script", required=True)
    r.add_argument("--send", default=None, help="要上传的本地文件或目录")
    r.add_argument("--dest", default=None)
    r.add_argument("--pull", default=None, help="跑完要拉回来的远程路径")
    r.add_argument("--timeout", type=int, default=3600)
    ns = p.parse_args(argv)
    if ns.cmd == "probe": return cmd_probe() or 0
    return cmd_run(ns) or 0

if __name__ == "__main__":
    sys.exit(main())
