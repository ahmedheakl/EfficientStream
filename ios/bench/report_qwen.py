#!/usr/bin/env python3
"""Tables for the streaming Qwen3.5-0.8B's phone runs (qwen_results.jsonl), next to the EfficientStream pipeline's
0.5 fps run on the same questions.

    python ios/bench/report_qwen.py ios/bench/results/qwen_results.jsonl ios/bench/results/qwen_grades.json \
        [ios/bench/results/results.jsonl ios/bench/results/grades.json]

Open-ended answers are graded by hand (qwen_grades.json: {run: {id: 0 or 1}}): correct when the reference's key fact
is stated and not contradicted, as for grades.json.
"""
import json
import statistics as st
import sys
from collections import defaultdict


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main(results, grades, es_results=None, es_grades=None):
    rs = load(results)
    gr = json.load(open(grades)) if grades else {}
    runs = defaultdict(list)
    for r in rs:
        runs[r["run"]].append(r)

    print("## Latency (mean per question)\n")
    cols = list(runs)
    print("| Stage | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))

    def row(name, f, fmt="{:.3f}"):
        print(f"| {name} | " + " | ".join(fmt.format(mean([f(r) for r in runs[c]])) for c in cols) + " |")

    row("Frames in the prompt", lambda r: r["timings"]["frames"], "{:.1f}")
    row("Frame: resize/pad + patches (s, when it arrives)", lambda r: r["timings"]["framePrep"])
    row("Frame: vision tower (s, when it arrives)", lambda r: r["timings"]["frameVision"])
    row("Prompt tokens", lambda r: r["timings"]["promptTokens"], "{:.0f}")
    row("of which visual", lambda r: r["timings"]["visualTokens"], "{:.0f}")
    row("Tokenize (s)", lambda r: r["timings"]["tokenize"])
    row("Prefill (s)", lambda r: r["timings"]["prefill"])
    row("Prefill (tokens/s)",
        lambda r: r["timings"]["promptTokens"] / r["timings"]["prefill"] if r["timings"]["prefill"] else None, "{:.0f}")
    row("**Question -> first words (s)**", lambda r: r["timings"]["timeToFirstToken"])
    row("Answer tokens", lambda r: r["timings"]["generatedTokens"], "{:.1f}")
    row("Decode (s)", lambda r: r["timings"]["decode"])
    row("Decode (tokens/s)", lambda r: (r["timings"]["generatedTokens"] - 1) / r["timings"]["decode"]
        if r["timings"]["decode"] > 0 and r["timings"]["generatedTokens"] > 1 else None, "{:.1f}")
    row("**Question -> full answer (s)**", lambda r: r["timings"]["total"])
    row("Multiple choice: question -> letter (s)", lambda r: r["timings"]["mcTotal"] if r.get("mcAnswer") else None)
    row("Memory after the question (MB)", lambda r: r["memoryMB"], "{:.0f}")
    print()
    for c in cols:
        th = defaultdict(int)
        for r in runs[c]:
            th[r["thermal"]] += 1
        print(f"- {c}: thermal states {dict(th)}; frames {sorted({r['framePixels'] for r in runs[c]})}")
    print()

    print("## Accuracy\n")
    es = {}
    if es_results:
        eg = json.load(open(es_grades)) if es_grades else {}
        for r in load(es_results):
            if r["run"] == "0.5fps":
                r["_oe"] = eg.get("0.5fps", {}).get(str(r["id"]))
                es[r["id"]] = r
    names = cols + (["EfficientStream 0.5 fps"] if es else [])
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    benches = sorted({r["bench"] for r in rs})
    for b in benches:
        cells = []
        for c in cols:
            g = gr.get(c, {})
            xs = [g.get(str(r["id"])) for r in runs[c] if r["bench"] == b]
            n = sum(x is not None for x in xs)
            cells.append(f"{sum(x or 0 for x in xs)} / {len(xs)}" + ("" if n == len(xs) else f" ({len(xs) - n} ungraded)"))
        if es:
            xs = [r["_oe"] for r in es.values() if r["bench"] == b]
            cells.append(f"{sum(x or 0 for x in xs)} / {len(xs)}")
        print(f"| {b} open-ended | " + " | ".join(cells) + " |")
    for b in benches:
        cells = []
        for c in cols:
            xs = [r["mcCorrect"] for r in runs[c] if r["bench"] == b and r.get("mcCorrect") is not None]
            cells.append(f"{sum(xs)} / {len(xs)}" if xs else "-")
        if es:
            xs = [r["mcCorrect"] for r in es.values() if r["bench"] == b and r.get("mcCorrect") is not None]
            cells.append(f"{sum(xs)} / {len(xs)}" if xs else "-")
        if any(c != "-" for c in cells):
            print(f"| {b} multiple choice | " + " | ".join(cells) + " |")
    cells = []
    for c in cols:
        g = gr.get(c, {})
        cells.append(f"{sum(g.get(str(r['id'])) or 0 for r in runs[c])} / {len(runs[c])}")
    if es:
        cells.append(f"{sum(r['_oe'] or 0 for r in es.values())} / {len(es)}")
    print("| **All open-ended** | " + " | ".join(cells) + " |")
    cells = []
    for c in cols:
        xs = [r["mcCorrect"] for r in runs[c] if r.get("mcCorrect") is not None]
        cells.append(f"{sum(xs)} / {len(xs)}")
    if es:
        xs = [r["mcCorrect"] for r in es.values() if r.get("mcCorrect") is not None]
        cells.append(f"{sum(xs)} / {len(xs)}")
    print("| **All multiple choice** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1] if len(a) > 1 else None, a[2] if len(a) > 2 else None, a[3] if len(a) > 3 else None)
