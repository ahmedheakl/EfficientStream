"""Scores of es_eval.py runs: per benchmark, one setting (--arm) against another (--ref) on the questions both answered, with a paired
bootstrap over questions (5,000 resamples, stratified by subtask where the metric averages subtasks; 95% interval of the difference).
Metrics: OVO-Bench = mean of the real-time, backward and forward means of its subtask accuracies (answers scored with the OneStreamer
release's score_prediction); StreamingBench (real-time) and ODV-Bench = accuracy; OVBench = mean subtask accuracy; StreamBench = share of
answers the LLaMA-3-8B judge (streambench.py --phase judge --src <run dir>/streambench) calls correct. A benchmark is shown only when
both settings cover at least 99% of its questions.
usage (onestreamer-eval env, from this folder): python score_es.py --arm s16ret --ref s16 [--runs out/es] [--json out.json]"""
import argparse
import collections
import glob
import json

import numpy as np

import answer_variants as AV
from onestreamer.ovobench import score_prediction
from onestreamer.schema import Sample

RT, BW, FW = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR"), ("EPM", "ASI", "HLD"), ("REC", "SSR", "CRR")
BENCH = {"OVO": (("ovo", "ovo_fwd"), "ovo"), "StreamingBench": (("sb",), "acc"), "StreamBench": (("streambench",), "judge"),
         "ODV": (("odv",), "acc"), "OVBench": (("ovb",), "sub")}


def make_scorer(rows, qs, metric, rng):
    strata = collections.defaultdict(list)
    for i, q in enumerate(qs):
        strata[rows[q]["subtask"] if metric in ("ovo", "sub") else "all"].append(i)
    strata = {s: np.array(ix) for s, ix in strata.items()}
    idx = {s: rng.integers(0, len(ix), size=(5000, len(ix))) for s, ix in strata.items()}

    def score(v, draw=False):
        acc = {s: 100 * (v[ix].mean() if not draw else v[ix][idx[s]].mean(axis=1)) for s, ix in strata.items()}
        if metric == "ovo":
            return np.mean([np.mean([acc[s] for s in g if s in acc], axis=0) for g in (RT, BW, FW)], axis=0)
        return np.mean(list(acc.values()), axis=0)
    return score


def compare(rows, metric, A, B):
    qs = sorted(q for q in rows if q in A and q in B)
    if len(qs) < 0.99 * len(rows):
        return dict(incomplete=True, n=len(qs), of=len(rows))
    sc = make_scorer(rows, qs, metric, np.random.default_rng(0))
    a = np.array([B[q]["hit"] for q in qs], float); c = np.array([A[q]["hit"] for q in qs], float); d = sc(c - a, True)
    out = dict(n=len(qs), ref=float(sc(a)), arm=float(sc(c)), d=float(sc(c) - sc(a)), lo=float(np.percentile(d, 2.5)),
               hi=float(np.percentile(d, 97.5)), ref_input=float(np.mean([B[q].get("input_tokens", 0) for q in qs])),
               arm_input=float(np.mean([A[q].get("input_tokens", 0) for q in qs])))
    if metric == "ovo":
        for gname, g in (("RT", RT), ("BW", BW), ("FW", FW)):
            sel = [i for i, q in enumerate(qs) if rows[q]["subtask"] in g]
            out[gname] = (float(100 * a[sel].mean()), float(100 * c[sel].mean()))
    return out


def hits(runs, setting, parts, metric, rows):
    """{sample_id: row with "hit"} of one es_eval setting (judged rows for StreamBench)."""
    R = {}
    for p in parts:
        for f in glob.glob(f"{runs}/{setting}/{p}/{'judged' if metric == 'judge' else 'results'}_*.jsonl"):
            for l in open(f):
                x = json.loads(l)
                if metric == "judge":
                    x["hit"] = float(str(x.get("llama_pred", "")).strip().lower() == "yes")
                else:
                    r = rows[x["sample_id"]]
                    smp = Sample.from_dict({k: v for k, v in r.items() if k not in ("item", "domain", "category", "time")})
                    x["hit"] = int(bool(score_prediction(smp, x["prediction"])[1]))
                R[x["sample_id"]] = x
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", default="s16")
    ap.add_argument("--runs", default="out/es")
    ap.add_argument("--benches", default=",".join(BENCH))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for b in a.benches.split(","):
        parts, metric = BENCH[b]
        rows = {r["sample_id"]: r for p in parts for r in AV.load_bench(p)[0]}
        x = res[b] = compare(rows, metric, hits(a.runs, a.arm, parts, metric, rows), hits(a.runs, a.ref, parts, metric, rows))
        if x.get("incomplete"):
            print(f"{b:<15} incomplete ({x['n']}/{x['of']} questions)")
            continue
        s = f"{b:<15} {a.ref} {x['ref']:.2f} -> {a.arm} {x['arm']:.2f} ({x['d']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]), n={x['n']}"
        s += f", input tokens {x['ref_input']:.0f} -> {x['arm_input']:.0f}"
        if "RT" in x:
            s += f" | RT {x['RT'][0]:.1f}->{x['RT'][1]:.1f} BW {x['BW'][0]:.1f}->{x['BW'][1]:.1f} FW {x['FW'][0]:.1f}->{x['FW'][1]:.1f}"
        print(s)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
