"""最小场景单元的标注表：把"一个师傅的一次判断"写成六要素。

库帕思那套六要素（情境/线索/判断/动作/边界/结果）在这里落地成一条 JSONL 记录。
关键不是"收集知识"——是让同一条记录可被 craftagent 复算，看方法能不能恢复师傅的判断。
所以 judgement.blamed_channel 和 judgement.window 是可验证字段，不是散文。

记录一旦写入就不再改（append-only）。师傅改主意了，追加一条新的，标注 supersedes。
"""
import json, os, time, uuid
from datetime import datetime

FIELDS = ("situation", "clues", "judgement", "action", "boundary", "result")

def blank(machine="M01", process="外圆磨削（滚道）"):
    """一条空记录。现场填表按这个骨架来，缺哪项就留空——空着比瞎填有价值。"""
    return {
        "id": uuid.uuid4().hex[:12],
        "recorded_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "source": {"master": "", "工号": "", "years": None, "跟岗人": ""},
        "supersedes": None,
        # 1 情境：这一段工况是什么
        "situation": {"machine": machine, "process": process,
                      "ts_start": "", "ts_end": "",
                      "batch": "", "material": "", "shift": "",
                      "note": "这一段设备在干什么（换批/换砂轮/开夜班/停机后重启…）"},
        # 2 线索：师傅当场看到/听到/摸到了什么，以及他从哪些读数上看出来的
        "clues": {"channels": [],            # ["spindle_current_a","vibration_mm_s"]
                  "observed": "",            # 原话：声音发闷、工件发烫、火花变黄
                  "at_ts": ""},
        # 3 判断：他认为是哪个量先动的，动在哪一刻，凭什么
        "judgement": {"blamed_channel": "",   # 可验证：必须是机器的信号名，不是"感觉砂轮不行"
                      "syndrome": "",         # 对得上 profile.syndromes 的名字就填
                      "window_start": "", "window_end": "",   # 他说的"从这时候起不对"
                      "confidence": None,     # 0-1，师傅自己给的
                      "reason": ""},
        # 4 动作：他做了什么，按顺序
        "action": {"steps": [],               # [{"do":"修整砂轮","param":"0.02mm","order":1}]
                   "why": ""},
        # 5 边界：什么条件下这个判断不成立 / 要改判
        "boundary": {"invalid_when": [],      # ["刚开机的头两小时","换批后首批"]
                     "leads_expected_min": None,  # 他预期症状领先质量多少分钟
                     "note": ""},
        # 6 结果：动作之后质量回来了没有
        "result": {"quality_after": None, "recovered": None,
                   "recover_min": None, "note": ""},
    }

def validate(rec):
    """只挑会让评分器算错的地方报出来。散文写得难看不管。"""
    bad = []
    for f in FIELDS:
        if f not in rec: bad.append("缺要素 %s" % f)
    j = rec.get("judgement") or {}
    if not j.get("blamed_channel"):
        bad.append("judgement.blamed_channel 空：没有可验证的判断，这条只能当访谈记录，不能用来验方法")
    b = rec.get("boundary") or {}
    if not (b.get("invalid_when") or b.get("leads_expected_min") is not None):
        bad.append("boundary 空：只说结论不说边界，这条经验不可复现")
    return bad

def append(rec, path):
    bad = validate(rec)
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return bad

def load(path):
    out = []
    if not os.path.exists(path): return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except ValueError: pass
    seen = {}
    for r in out: seen[r.get("id")] = r
    dropped = {r.get("supersedes") for r in out if r.get("supersedes")}
    return [r for r in out if r.get("id") not in dropped]
