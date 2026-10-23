"""OneStreamer manages its own memory with its own summaries (no extra model, no training), answer time only.

OneStreamer writes </Summary> lines (one per ~114 s of video on OVO; 79.7% of its captions fall inside a summary's time range). Its
release config has a `consolidate_summary` switch (false, and no released code reads it). Here it is applied at answer time:
  full          its memory as in the Table 8 setting (all captions + summaries, render_with_budget at 12,000 tokens)
  consolidated  captions whose [start, end] lies inside a summary's [start, end] are removed; summaries and later captions stay
  none          no memory (last 16 frames only)
  keep60        consolidated, but captions that end in the last 60 s before the question stay even if a summary covers them (the
                article's "keep the most recent items verbatim"); added after the first run lost 1.1 points on real-time questions
  hybrid        the article's three ideas together: summaries for the older part; the last 60 s of captions verbatim; repeated
                captions merged into one line with the full time range (clearing); and at question time the K=5 archived captions (those
                a summary replaced) that share the most words with the question are put back (just-in-time retrieval from the saved
                log, word overlap with OneStreamer's own tokenization and stop words; only captions sharing at least one word)
Same frames, prompt, renderer and scorer for all three; only the memory differs. All 1,468 OVO-Bench questions of run-id full8
(backward + real-time), so group scores are comparable to Table 8. The memory files are not changed: OneStreamer wrote them while
seeing its full memory; feeding the consolidated memory back into the watching loop is the next step.
"""
import argparse
import collections
import glob
import json
import math
import random
import re
from pathlib import Path

import clm_check as C

REALTIME = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
BACKWARD = ("EPM", "ASI", "HLD")
PAPER = {"OCR": 94.6, "ACR": 80.7, "ATR": 81.9, "STU": 70.2, "FPD": 78.2, "OJR": 82.6, "EPM": 62.6, "ASI": 71.6, "HLD": 99.5}


def consolidate(events, keep_after=None):
    sums = [e for e in events if e["kind"] == "summary" and e["text"]]
    inside = lambda e: any(float(s["start"]) <= float(e["start"]) and float(e["end"]) <= float(s["end"]) for s in sums)
    recent = lambda e: keep_after is not None and float(e["end"]) >= keep_after
    return [e for e in events if not (e["kind"] == "response" and inside(e) and not recent(e))]


STOP = {"a", "an", "the", "is", "are", "was", "were", "what", "where", "who", "how", "did", "do", "to", "in", "on", "of", "and", "i"}


def merge_repeats(events):
    """Captions with the same text (time prefix removed) become one caption covering all their time ranges."""
    out, seen = [], {}
    for e in events:
        if e["kind"] != "response":
            out.append(e)
            continue
        body = C.body_of(e["text"])
        if body in seen:
            m = seen[body]
            m["start"], m["end"], m["text"] = min(float(m["start"]), float(e["start"])), max(float(m["end"]), float(e["end"])), body
        else:
            seen[body] = dict(e)
            out.append(seen[body])
    return out


def hybrid(events, qtime, question, k=5, recent=60.0):
    kept = consolidate(events, keep_after=qtime - recent)
    ids = {id(e) for e in kept}
    archived = [e for e in events if e["kind"] == "response" and id(e) not in ids]
    terms = set(re.findall(r"[a-z0-9]+", question.lower())) - STOP
    scored = []
    for i, e in enumerate(archived):
        overlap = len(terms & set(re.findall(r"[a-z0-9]+", C.body_of(e["text"]).lower())))
        if overlap:
            scored.append((overlap, i, e))
    picked, bodies = [], set()
    for _, _, e in sorted(scored, key=lambda x: (-x[0], -x[1])):
        if C.body_of(e["text"]) not in bodies and len(picked) < k:
            picked.append(e)
            bodies.add(C.body_of(e["text"]))
    return sorted(merge_repeats(kept + picked), key=lambda e: (float(e["end"]), e["kind"]))


def answer(args, out):
    from onestreamer.engine import OneStreamerEngine
    from onestreamer.memory import MemoryBank, MemoryEvent
    from onestreamer.video import read_recent_frames
    from onestreamer.prompts import MEMORY_QA_SYSTEM
    from onestreamer.ovobench import build_question_text, score_prediction
    from onestreamer.schema import Sample
    samples = json.load(open(C.FULL8 / "manifest.json"))["samples"][args.shard::args.nshards]
    engine = OneStreamerEngine(model_path=str(C.MODELS / "OneStreamer-4B"), dtype="bfloat16", attn_implementation="flash_attention_2",
                               min_pixels=3136, max_pixels=100352)
    tok = engine.processor.tokenizer
    with open(out / f"results_{args.shard:02d}.jsonl", "w") as f:
        for i, r in enumerate(samples):
            sample = Sample.from_dict(r)
            frames = read_recent_frames(C.VIDEO_ROOT / sample.video, sample.start, sample.end, recent_frames=16, sample_fps=1.0)
            mem = json.load(open(C.FULL8 / "memory/captions/OneStreamer-4B" / f"{r['sample_id']}.json"))
            events = mem["events"]
            render = lambda ev: MemoryBank(MemoryEvent.from_dict(e) for e in ev).render_with_budget(
                mode="hierarchical", question=sample.question, tokenizer=tok, max_tokens=12000)[0]
            texts = {"full": lambda: render(events), "consolidated": lambda: render(consolidate(events)), "none": lambda: "",
                     "keep60": lambda: render(consolidate(events, keep_after=float(mem["question_time"]) - 60)),
                     "hybrid": lambda: render(hybrid([e for e in events if e["kind"] in ("response", "summary") and e["text"]],
                                                     float(mem["question_time"]), sample.question))}
            for variant in args.variants.split(","):
                text = texts[variant]()
                pred = engine.answer(system=MEMORY_QA_SYSTEM, memory_text=text, recent_frames=frames,
                                     question_text=build_question_text(sample), max_tokens=16, recent_resize_policy="source_pixels")
                parsed, hit = score_prediction(sample, pred)
                f.write(json.dumps(dict(sample_id=r["sample_id"], subtask=r["subtask"], variant=variant, prediction=pred, parsed=parsed,
                                        hit=hit, memory_tokens=len(tok.encode(text, add_special_tokens=False)) if text else 0)) + "\n")
            f.flush()
            print(f"shard {args.shard}: {i + 1}/{len(samples)}", flush=True)
    print("ANSWER_DONE", args.shard, flush=True)


def mcnemar(a, b, qs):
    g = sum(b[q] and not a[q] for q in qs)
    l = sum(a[q] and not b[q] for q in qs)
    n = g + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(g, l) + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(0)
    d = [b[q] - a[q] for q in qs]
    m = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
    return f"+{g}/-{l} diff {100 * sum(d) / len(d):+.1f} [{100 * m[250]:+.1f},{100 * m[9750]:+.1f}] p={p:.3f}"


def report(out, extra):
    rows = [json.loads(l) for d in [out] + extra for p in sorted(glob.glob(str(d / "results_*.jsonl"))) for l in open(p) if l.strip()]
    hit, tok, sub = collections.defaultdict(dict), collections.defaultdict(dict), {}
    for r in rows:
        hit[r["variant"]][r["sample_id"]] = r["hit"]
        tok[r["variant"]][r["sample_id"]] = r["memory_tokens"]
        sub[r["sample_id"]] = r["subtask"]
    qs = sorted(set.intersection(*(set(v) for v in hit.values())))
    full8 = {}
    for p in glob.glob(str(C.FULL8 / "shards/predictions-*.jsonl")):
        for l in open(p):
            d = json.loads(l)
            full8[d["sample_id"]] = d["hit"]
    print(f"{len(qs)} questions; 'full' here vs the full8 run: same outcome on {sum(hit['full'][q] == full8[q] for q in qs)}/{len(qs)}\n")
    acc = lambda v, s: 100 * sum(hit[v][q] for q in qs if sub[q] == s) / max(1, sum(sub[q] == s for q in qs))
    vs = [v for v in ("full", "consolidated", "keep60", "hybrid", "none") if v in hit]
    print(f"{'subtask':14s} " + " ".join(f"{v[:12]:>12s}" for v in vs) + f" {'paper':>6s}")
    for s in REALTIME + BACKWARD:
        print(f"{s:14s} " + " ".join(f"{acc(v, s):12.1f}" for v in vs) + f" {PAPER[s]:6.1f}")
    for name, grp, paper in (("Real-time", REALTIME, 81.4), ("Backward", BACKWARD, 77.9)):
        print(f"{name + ' avg':14s} " + " ".join(f"{sum(acc(v, s) for s in grp) / len(grp):12.1f}" for v in vs) + f" {paper:6.1f}")
    print("\nmemory tokens (mean / median):")
    for v in [v for v in vs if v != "none"]:
        t = sorted(tok[v][q] for q in qs)
        ae = sorted(tok[v][q] for q in qs if sub[q] in ("ASI", "EPM"))
        print(f"  {v:13s} all {sum(t) / len(t):6.0f} / {t[len(t) // 2]:5d}   ASI+EPM {sum(ae) / len(ae):6.0f} / {ae[len(ae) // 2]:5d}")
    print("\npaired tests (exact McNemar; questions gained/lost; accuracy difference with bootstrap 95% CI):")
    for label, sel in (("all", lambda q: True), ("ASI+EPM", lambda q: sub[q] in ("ASI", "EPM")), ("ASI", lambda q: sub[q] == "ASI"),
                       ("EPM", lambda q: sub[q] == "EPM"), ("real-time", lambda q: sub[q] in REALTIME)):
        q2 = [q for q in qs if sel(q)]
        for v in [v for v in vs if v not in ("full", "none")]:
            print(f"  {label:10s} {v} vs full: {mcnemar(hit['full'], hit[v], q2)}   {v} vs none: {mcnemar(hit['none'], hit[v], q2)}")
        print(f"  {label:10s} full vs none: {mcnemar(hit['none'], hit['full'], q2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["answer", "report"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--variants", default="full,consolidated,none")
    ap.add_argument("--extra", default="", help="report: more result dirs to merge (e.g. the first run's)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answer(args, out) if args.phase == "answer" else report(out, [Path(x) for x in args.extra.split(",") if x])


if __name__ == "__main__":
    main()
