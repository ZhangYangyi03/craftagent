# Contributing to craftagent

Thanks for looking. The single most valuable contribution to this repository is
**running the checks on your own machine and telling us what happened** —
especially if it failed. This portfolio has strong internal verification and
almost no external reproduction; an independent result, positive or negative,
is worth more than a star.

## Ground rules

- MIT licensed. By opening a pull request you agree your contribution is under the same license.
- Bug reports, reproduction attempts and arguments about the claims are contributions. You do not need to write code.
- One change per pull request.
- If a number in the README disagrees with the code, that is a bug. Please open an issue rather than editing the prose to match.

## What this repository is

Machine-condition logs and CMM inspection data aligned on one time axis, with change-point detection and cause attribution, using only the Python standard library. Synthetic data has planted ground truth; acceptance checks are run against that truth, including the cases the method is not allowed to get wrong.

## Reproduce it

No API key, third-party package, GPU or factory data is needed:

```bash
python -m craftagent verify
```

That prints the five acceptance checks against the planted truth (drift machine M01, no false positive on M02, change-point location, signal lead, and the leading candidate). A longer path exercises the whole chain:

```bash
python -m craftagent synth
python -m craftagent analyse --spec 8.5
python -m craftagent verify
python -m craftagent grade --auto-mask
```

CI runs the unit suite on Ubuntu and Windows for Python 3.11 / 3.13, asserts that no third-party module is imported on the analysis path, and executes the synth → analyse → verify → grade chain end to end. Some subcommands (for example `cad`) need OpenCASCADE; the checks above deliberately do not.

## Reporting an independent reproduction

Use the issue template "Independent reproduction". What makes a report actionable:

1. The commit hash or version you checked out.
2. `python -c "import sys, platform; print(sys.version, platform.platform())"`.
3. The exact commands and their full output (a failed check with a traceback is a good report).
4. Expected vs actual, in one sentence.
5. Whether all checks passed, some failed, numbers differed, or you could not run them.

Both "all checks passed on Ubuntu 24.04 / Python 3.12" and "check 3 fails on my
machine" are useful. Please do not fix a failing check silently in a PR — report
it first so the discrepancy is recorded.

## Claim scope

The published numbers come from synthetic data with planted ground truth and from this repository's own machines. They are internal checks, not an independent evaluation on real factory data. Please hold us to that boundary: a pull request that strengthens a claim without new evidence will be asked for the evidence.
