# -*- coding: utf-8 -*-
"""build_cncrt.py -- 编 C 内核。找不到编译器就明确说不跑，不假装成功。

为什么 C 内核值得存在：规划表是纯算术，Python 跑一遍 60 微秒，
对离线分析完全够用。它存在的理由是**对拍**——两份独立实现同意，
才说明规划结果是对的。同一份代码跑两遍永远同意，那测的是确定性。

所以这个脚本只做一件事：把 cncrt/src/*.c 编成 cncrt/cncrt.exe。
编不出来时 bench/pair.py 会报 'cncrt.exe not built' 并跳过对拍，
测试里那一条会 skip——跳过是诚实的，假装通过不是。
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CNCRT = os.path.join(ROOT, 'cncrt')

CANDIDATES = [
    'gcc', 'cc', 'clang',
    r'C:\msys64\mingw64\bin\gcc.exe',
    r'C:\mingw64\bin\gcc.exe',
]


def find_cc():
    for c in CANDIDATES:
        p = shutil.which(c) if os.path.basename(c) == c else (
            c if os.path.exists(c) else None)
        if p:
            return p
    return None


def build(verbose=True):
    cc = find_cc()
    if not cc:
        if verbose:
            print('没有 C 编译器（找过 %s）' % ', '.join(CANDIDATES))
            print('对拍会被跳过。内核本身不需要编译器才能读，但只有对拍能证明它对。')
        return None
    src = [os.path.join(CNCRT, 'src', f)
           for f in sorted(os.listdir(os.path.join(CNCRT, 'src')))
           if f.endswith('.c')]
    out = os.path.join(CNCRT, 'cncrt.exe' if os.name == 'nt' else 'cncrt')
    cmd = [cc, '-O2', '-std=c99', '-Wall', '-Wextra', '-o', out] + src + ['-lm']
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    if r.returncode != 0:
        if verbose:
            print('编译失败 rc=%d' % r.returncode)
            print(r.stderr[-2000:])
        return None
    if verbose:
        print('编译成功：%s' % out)
        if r.stderr.strip():
            # 有警告就报出来。零警告是这个仓对 C 代码的要求：
            # 一个看起来正常的警告，往往就是那个把浮点边界判成越界的东西。
            print('警告：')
            print(r.stderr[-2000:])
    return out


if __name__ == '__main__':
    sys.exit(0 if build() else 1)
