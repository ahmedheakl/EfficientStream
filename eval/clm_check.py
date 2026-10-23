"""Zero-shot CLM capability check on OVO-Bench backward questions (ASI, EPM).

Following Context Language Models (arXiv 2609.37725): the context is a file the model may edit freely; without an edit, new content is
appended. Here the file is OneStreamer-4B's own caption memory, replayed from its saved memory files (run-id full8): captions arrive in
60 s chunks of video time and are appended to the file, which has a token limit L. Each policy manages the file without seeing the
question. The final file is then used as OneStreamer's memory in its own answer path (16 recent source-resolution frames,
MEMORY_QA_SYSTEM, greedy, 16 tokens), so only the memory content differs between policies.

Policies
  full         all captions, OneStreamer's 12,000-token budget (the Table 8 setting)            reference
  none         no memory (recent16)                                                            reference
  drop_oldest  append; delete the oldest lines while over L
  dedupe       append only captions whose text is not already in the file; then drop_oldest
  os_budget    OneStreamer's render_with_budget at L (keyword match with the question at answer time; not causal)
  summary      append; when over L the editor rewrites the whole file as a summary (Codex-style compaction)
  rewrite      append; after every chunk the editor may reply with the complete new file or NO_EDIT; if over L it is told by how much
               and retries (up to 2 retries)
  rewrite_skill  rewrite + a hand-written context-management guide in the prompt
  clm          append; after every chunk the editor may edit the file with a Python snippet on `s`, or reply NO_EDIT
For summary and clm, if the file is still over L after the editor's turn, the oldest lines are cut and a violation is counted.

Cost: prefix-reuse tokens of the memory file per step (tokens from the first changed token onward + tokens the editor generated), the
token form of the paper's prefix-reuse FLOPs.
"""
import argparse
import gc
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.home() / "onestreamer_repro"
EVAL = ROOT / "OneStreamer/Eval"
sys.path.insert(0, str(EVAL / "Proactive_Eval/OneStreamer/src"))
sys.path.insert(0, str(EVAL / "Proactive_Eval"))  # for `shared.inference_fast`, imported by onestreamer.engine
VIDEO_ROOT = ROOT / "path_to/datasets/OVOBench"
MODELS = ROOT / "path_to/models"
FULL8 = EVAL / "outputs/OneStreamer-4B/ovobench/full8/memory/ovobench"

CLM_SYSTEM = """You manage your own context. Your context is the text file shown below; it is your only memory of a video you are watching.
Later, someone will ask a question about anything that happened in the video. It will be answered using only this file plus the last 16 seconds of video.
The file must stay within {limit} tokens.

New observations are appended to the file automatically. After each update you may edit the file in any way you like: delete, shorten, merge, rewrite or reorganize lines, keep notes, anything.
To edit, reply with one Python code block that modifies the string variable `s` (the full file content); the module `re` is available.
To leave the file unchanged, reply exactly NO_EDIT."""

CLM_USER = """Video time: {t}s. The file is {n} tokens (limit {limit}).
<<<FILE
{s}
FILE>>>"""

REWRITE_SYSTEM = """You manage your own context. Your context is the text file shown below; it is your only memory of a video you are watching.
Later, someone will ask a question about anything that happened in the video. It will be answered using only this file plus the last 16 seconds of video.
The file must stay within {limit} tokens.

New observations are appended to the file automatically. After each update you may change the file in any way you like: delete, shorten, merge, rewrite or reorganize lines, keep notes, anything.
To change it, reply with the complete new file between a line <<<FILE and a line FILE>>>.
To leave the file unchanged, reply exactly NO_EDIT.{skill}"""

SKILL = """

Context-management guide:
- Repeated or near-identical lines usually describe one ongoing activity. Merge them into one line with the full time range, e.g. "[120s-300s] The person keeps stirring the pot."
- Keep what later questions are likely to need: which objects appeared and where they were put or found, the order of the main actions, what people did, any text or numbers seen, and changes of place.
- Keep a time range on every line. Keep lines short; drop camera motion and filler words.
- When space runs short, compress the oldest part into a few summary lines instead of deleting it; keep the most recent part more detailed.
- Never add details that are not in the file."""

OVER_LIMIT = """The file is now {n} tokens, {over} over the limit of {limit}. Reply with a shorter complete file between a line <<<FILE and a line FILE>>>."""

SUMMARY_SYSTEM = """You compact the memory of a video assistant. The memory file below is over its limit of {limit} tokens.
Rewrite it as a summary within {limit} tokens that keeps the information most likely to matter for later questions about the video.
Output only the new file content."""

SANDBOX = r"""
import json, re, sys
d = json.loads(sys.stdin.read())
safe = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in
        ("len", "range", "enumerate", "str", "int", "float", "min", "max", "sorted", "list", "dict", "set", "tuple", "any", "all",
         "zip", "reversed", "isinstance", "abs", "round", "sum", "map", "filter", "bool", "print")}
ns = {"s": d["s"]}
exec(d["code"], {"__builtins__": safe, "re": re}, ns)
if not isinstance(ns.get("s"), str):
    raise TypeError("s is not a string")
print(json.dumps({"s": ns["s"]}))
"""


def run_edit(code, s):
    try:
        out = subprocess.run([sys.executable, "-I", "-c", SANDBOX], input=json.dumps({"code": code, "s": s}),
                             capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if out.returncode != 0:
        return None, (out.stderr.strip().splitlines() or ["error"])[-1][:200]
    return json.loads(out.stdout.strip().splitlines()[-1])["s"], None


def line_of(e):
    t = lambda v: str(int(float(v))) if float(v).is_integer() else f"{float(v):.2f}".rstrip("0").rstrip(".")
    tag = "SUMMARY " if e["kind"] == "summary" else ""
    return f"[{tag}{t(e['start'])}s-{t(e['end'])}s] {e['text']}"


def body_of(line):
    return re.sub(r"^(\[[^\]]*\]\s*)+", "", line).strip()


class Counter:
    def __init__(self, tok):
        self.tok = tok

    def ids(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def n(self, s):
        return len(self.ids(s))


def prefix_reuse(prev_ids, ids):
    k = 0
    for a, b in zip(prev_ids, ids):
        if a != b:
            break
        k += 1
    return len(ids) - k


def cut_oldest(s, limit, cnt):
    lines = s.split("\n")
    while lines and cnt.n("\n".join(lines)) > limit:
        lines.pop(0)
    return "\n".join(lines)


class Editor:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        path = MODELS / name
        self.name = name
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                                 attn_implementation="flash_attention_2").to("cuda").eval()

    def chat(self, system, user, max_new_tokens, history=()):
        import torch
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for role, text in history:
            msgs.append({"role": role, "content": [{"type": "text", "text": text}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": user}]})
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                      top_k=None)
        gen = out[0, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip(), int(gen.shape[-1])

    def close(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def chunks(events, qtime, step=60.0):
    for k in range(max(1, math.ceil(qtime / step))):
        lo, hi = k * step, min((k + 1) * step, qtime)
        yield hi, [e for e in events if lo < float(e["end"]) <= hi or (k == 0 and float(e["end"]) <= 0)]


def manage(policy, events, qtime, limit, cnt, editor=None, log=None):
    s, prev_ids = "", []
    stats = dict(steps=0, edits=0, no_edit=0, invalid=0, violations=0, cost_tokens=0, gen_tokens=0, errors=[])
    for t, new in chunks(events, qtime):
        prev_s = s
        add = "\n".join(line_of(e) for e in new)
        s = (s + "\n" + add).strip("\n") if add else s
        if policy == "dedupe":
            seen = {body_of(ln) for ln in prev_s.split("\n")} if prev_s else set()
            keep = []
            for e in new:
                body = body_of(line_of(e))
                if body not in seen:
                    keep.append(line_of(e))
                    seen.add(body)
            s = (prev_s + "\n" + "\n".join(keep)).strip("\n") if keep else prev_s
            s = cut_oldest(s, limit, cnt)
        elif policy == "drop_oldest":
            s = cut_oldest(s, limit, cnt)
        elif policy == "summary" and cnt.n(s) > limit:
            out, g = editor.chat(SUMMARY_SYSTEM.format(limit=limit), s, max_new_tokens=2 * limit)
            stats["gen_tokens"] += g
            stats["cost_tokens"] += g
            stats["edits"] += 1
            s = out
            if cnt.n(s) > limit:
                stats["violations"] += 1
                s = cut_oldest(s, limit, cnt)
        elif policy in ("rewrite", "rewrite_skill") and s:
            system = REWRITE_SYSTEM.format(limit=limit, skill=SKILL if policy == "rewrite_skill" else "")
            user = CLM_USER.format(t=int(t), n=cnt.n(s), limit=limit, s=s)
            history = []
            for attempt in range(3):
                out, g = editor.chat(system, user, max_new_tokens=2 * limit + 64, history=history)
                stats["gen_tokens"] += g
                stats["cost_tokens"] += g
                if log is not None:
                    log.write(json.dumps(dict(t=t, attempt=attempt, out=out[:4000])) + "\n")
                m = re.search(r"<<<FILE\s*\n?(.*?)\n?\s*FILE>>>", out, re.S)
                if m:
                    s = m.group(1).strip("\n")
                    stats["edits"] += 1
                elif out.strip().upper().startswith("NO_EDIT"):
                    stats["no_edit"] += 1
                else:
                    stats["invalid"] += 1
                    stats["errors"].append("no file block: " + out[:120].replace("\n", " "))
                n = cnt.n(s)
                if n <= limit:
                    break
                stats["retries"] = stats.get("retries", 0) + 1
                history = history + [("user", user), ("assistant", out)]
                user = OVER_LIMIT.format(n=n, over=n - limit, limit=limit)
            if cnt.n(s) > limit:
                stats["violations"] += 1
                s = cut_oldest(s, limit, cnt)
        elif policy == "clm" and s:
            out, g = editor.chat(CLM_SYSTEM.format(limit=limit), CLM_USER.format(t=int(t), n=cnt.n(s), limit=limit, s=s),
                                 max_new_tokens=768)
            stats["gen_tokens"] += g
            stats["cost_tokens"] += g
            m = re.search(r"```(?:python)?\s*\n(.*?)```", out, re.S)
            if m:
                new_s, err = run_edit(m.group(1), s)
                if err is None:
                    s = new_s
                    stats["edits"] += 1
                else:
                    stats["invalid"] += 1
                    stats["errors"].append(err)
            elif out.strip().upper().startswith("NO_EDIT"):
                stats["no_edit"] += 1
            else:
                stats["invalid"] += 1
                stats["errors"].append("no code block: " + out[:120].replace("\n", " "))
            if log is not None:
                log.write(json.dumps(dict(t=t, out=out[:4000])) + "\n")
            if cnt.n(s) > limit:
                stats["violations"] += 1
                s = cut_oldest(s, limit, cnt)
        ids = cnt.ids(s)
        stats["cost_tokens"] += prefix_reuse(prev_ids, ids)
        prev_ids = ids
        stats["steps"] += 1
    stats["final_tokens"] = cnt.n(s)
    return s, stats


def pick_samples(n_per):
    manifest = json.load(open(FULL8 / "manifest.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS / "OneStreamer-4B")
    out = []
    for sub in ("ASI", "EPM"):
        cands = []
        for row in manifest["samples"]:
            f = FULL8 / "memory/captions/OneStreamer-4B" / f"{row['sample_id']}.json"
            if row["subtask"] == sub and f.exists():
                d = json.load(open(f))
                cands.append((sum(len(tok.encode(e["text"])) for e in d["events"]), row["sample_id"], row))
        cands.sort(key=lambda x: -x[0])
        out += [r for _, _, r in cands[:n_per]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["manage", "answer"], required=True)
    ap.add_argument("--editor", default="-", help="model name for summary/clm; '-' runs drop_oldest and dedupe")
    ap.add_argument("--policies", default="", help="comma list overriding the default policies of the manage phase")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--n-per", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    cnt = Counter(AutoTokenizer.from_pretrained(MODELS / "OneStreamer-4B"))
    sp = out / "samples.json"
    if sp.exists():
        ids = json.load(open(sp))
        rows = {r["sample_id"]: r for r in json.load(open(FULL8 / "manifest.json"))["samples"]}
        samples = [rows[i] for i in ids]
    else:
        samples = pick_samples(args.n_per)
        json.dump([r["sample_id"] for r in samples], open(sp, "w"))
    mem = {r["sample_id"]: json.load(open(FULL8 / "memory/captions/OneStreamer-4B" / f"{r['sample_id']}.json")) for r in samples}
    evs = {k: [e for e in d["events"] if e["kind"] in ("response", "summary") and e["text"]] for k, d in mem.items()}

    if args.phase == "manage":
        name = args.editor
        policies = ("drop_oldest", "dedupe") if name == "-" else ("summary", "clm")
        if args.policies:
            policies = tuple(args.policies.split(","))
        ed = None if name == "-" else Editor(name)
        tag = name + ("_" + "+".join(policies) if args.policies else "")
        with open(out / f"managed_{tag}.jsonl", "w") as f, open(out / f"clm_trace_{tag}.jsonl", "w") as log:
            for r in samples:
                sid = r["sample_id"]
                for policy in policies:
                    t0 = time.time()
                    log.write(json.dumps(dict(sample=sid, policy=policy)) + "\n")
                    s, st = manage(policy, evs[sid], float(mem[sid]["question_time"]), args.limit, cnt, ed,
                                   log if policy in ("clm", "rewrite", "rewrite_skill") else None)
                    st["seconds"] = round(time.time() - t0, 1)
                    f.write(json.dumps(dict(sample_id=sid, policy=policy, editor=name, file=s, stats=st)) + "\n")
                    f.flush()
                    log.flush()
                    print(name, sid, policy, {k: v for k, v in st.items() if k != "errors"}, flush=True)
        print("MANAGE_DONE", name, flush=True)
        return

    from onestreamer.engine import OneStreamerEngine
    from onestreamer.memory import MemoryBank, MemoryEvent
    from onestreamer.video import read_recent_frames
    from onestreamer.prompts import MEMORY_QA_SYSTEM, MEMORY_PREAMBLE, MEMORY_POSTAMBLE
    from onestreamer.ovobench import build_question_text, score_prediction
    from onestreamer.schema import Sample
    managed = [json.loads(l) for p in sorted(out.glob("managed_*.jsonl")) for l in open(p) if l.strip()]
    engine = OneStreamerEngine(model_path=str(MODELS / "OneStreamer-4B"), dtype="bfloat16", attn_implementation="flash_attention_2",
                               min_pixels=3136, max_pixels=100352)
    tok = engine.processor.tokenizer
    with open(out / "results.jsonl", "w") as f:
        for r in samples:
            sid = r["sample_id"]
            sample = Sample.from_dict(r)
            frames = read_recent_frames(VIDEO_ROOT / sample.video, sample.start, sample.end, recent_frames=16, sample_fps=1.0)
            bank = MemoryBank(MemoryEvent.from_dict(e) for e in mem[sid]["events"])
            variants = {("full", "-"): (bank.render_with_budget(mode="hierarchical", question=sample.question, tokenizer=tok,
                                                                max_tokens=12000)[0], {}),
                        ("none", "-"): ("", {}),
                        ("os_budget", "-"): (bank.render_with_budget(mode="hierarchical", question=sample.question, tokenizer=tok,
                                                                     max_tokens=args.limit)[0], {})}
            for m in managed:
                if m["sample_id"] == sid:
                    variants[(m["policy"], m["editor"])] = (f"{MEMORY_PREAMBLE}\n{m['file']}\n{MEMORY_POSTAMBLE}" if m["file"] else "",
                                                            m["stats"])
            for (policy, name), (text, st) in variants.items():
                pred = engine.answer(system=MEMORY_QA_SYSTEM, memory_text=text, recent_frames=frames,
                                     question_text=build_question_text(sample), max_tokens=16, recent_resize_policy="source_pixels")
                parsed, hit = score_prediction(sample, pred)
                f.write(json.dumps(dict(sample_id=sid, subtask=r["subtask"], policy=policy, editor=name, limit=args.limit,
                                        prediction=pred, parsed=parsed, hit=hit, memory_text=text,
                                        memory_tokens=len(tok.encode(text, add_special_tokens=False)) if text else 0, stats=st)) + "\n")
                f.flush()
    print("ANSWER_DONE", flush=True)


if __name__ == "__main__":
    main()
