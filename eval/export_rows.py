"""Question rows for es_eval.py (out/es/rows_<bench>.json), exported with answer_variants.load_bench, the benchmark loader of this
harness. --split dev: the analysis half (videos whose path has an even md5); all: every question (rows_all_<bench>.json); with --rest
<run dir>, also rows_rest_<bench>.json, the questions of `all` that <run dir>/<bench>/results_*.jsonl does not have yet. Options become
"A. ..." strings, as the 0.8B's ask() expects. Rows hold no answers; score_es.py scores against the loader's rows.
usage (onestreamer-eval env, from this folder): python export_rows.py --split dev|all [--rest out/es/s16] [--benches ovo,sb,...]"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import answer_variants as A

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "all"], required=True)
ap.add_argument("--rest", default="", help="with --split all: a run dir whose finished questions rows_rest_<bench>.json leaves out")
ap.add_argument("--benches", default="ovo,ovo_fwd,sb,odv,ovb,streambench")
ap.add_argument("--out", default="out/es")
a = ap.parse_args()
done = {json.loads(l)["sample_id"] for f in glob.glob(f"{a.rest}/*/results_*.jsonl") for l in open(f)} if a.rest else set()
Path(a.out).mkdir(parents=True, exist_ok=True)
for part in a.benches.split(","):
    rows, vroot, _ = A.load_bench(part)
    out = []
    for r in rows:
        if a.split == "dev" and int(hashlib.md5(r["video"].encode()).hexdigest(), 16) % 2:
            continue
        opts = r.get("options")
        if isinstance(opts, dict):
            opts = [f"{k}. {v}" for k, v in opts.items()]
        elif isinstance(opts, list) and opts and not str(opts[0])[:2].rstrip().endswith("."):
            opts = [f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)]
        out.append(dict(sample_id=r["sample_id"], subtask=r.get("subtask", ""), video=str(vroot / r["video"]), start=float(r.get("start") or 0),
                        end=float(r["end"]), question=r["question"], options=opts or None, eval_mode=r.get("eval_mode", "")))
    json.dump(out, open(f"{a.out}/rows_{part}.json" if a.split == "dev" else f"{a.out}/rows_all_{part}.json", "w"))
    msg = f"{part} {a.split} {len(out)}"
    if a.rest:
        rest = [x for x in out if x["sample_id"] not in done]
        json.dump(rest, open(f"{a.out}/rows_rest_{part}.json", "w"))
        msg += f" rest {len(rest)}"
    print(msg)
