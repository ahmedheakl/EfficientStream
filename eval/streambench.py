"""StreamBench (StreamChat, arXiv 2501.13468; HF Barry-12138/StreamBench_v0.3, streaming_bench_v0.3.json) for OneStreamer-4B.

307 videos (Ego, WebVideo, Movie), 1,838 open-ended questions at timestamps, 6 types: OS (object search), LM (long-term memory search),
SM (short-term memory search), CI (conversational interaction), KG (knowledge-based QA), SF (simple factual). Memory types: OS, LM, SM, CI.

Protocol, from the authors' code (github.com/hmxiong/StreamChat @ f8bcd22e):
  - each question is asked on its own (their baseline scripts pass history=None), here with the video only up to the question time
    (streaming: start 0, end = question time; 9 questions timed after their video ends see the whole video, end = its last full second);
  - judge = LLaMA-3 8B Instruct with the prompt of eval_ego_streaming_with_llama3.py, verbatim, formatted by its prepare_prompt
    (pieces joined by spaces, no assistant header); answer parsed as the Python dict {'llama_pred': yes/no, 'score': int};
  - accuracy = share of 'yes', score = mean score (calculate_avg_score.py), coherence = mean |score_i - score_i+1| over consecutive rows.
  Deviation: the authors' generate call leaves sampling to the model's generation_config (sampling, temperature 0.6, top-p 0.9 for
  Meta-Llama-3-8B-Instruct); the judge here decodes greedily so the scores are reproducible.

Memory: one caption pass per video up to its last question, a boundary at every question time (caption_optimized.run_shared_caption_pass,
OneStreamer's StreamingBench caption config, configs/streamingbench.json); each question reads only the memory finished at its own time.
Answers: answer_variants.py --bench streambench (open-ended, MEMORY_DIRECT_QA_SYSTEM, up to 128 new tokens). Rows carry
eval_mode substring_match only because OneStreamer's Sample accepts no open-ended mode; it gives the plain question text and the direct
answer prompt, and answer_variants does not score StreamBench rows (the judge does).

Phases: build | caption (workers claim videos; stale .lock files are removed by the driver before a restart) | judge (sharded; --src answer dir) | report.
"""
import argparse
import ast
import collections
import glob
import json
import math
import os
import re
import time
from pathlib import Path

import clm_check as C

ROOT = C.ROOT / "path_to/datasets/StreamBench"
OUT = Path(__file__).resolve().parent / "out/streambench"
JUDGE = "meta-llama/Meta-Llama-3-8B-Instruct"
MEMORY_TYPES = ("OS", "LM", "SM", "CI")


def build(args):
    import cv2
    rows, dur = [], {}
    for v, x in enumerate(json.load(open(ROOT / "streaming_bench_v0.3.json"))):
        video = f"{x['info']['class_1']}/{x['info']['video_path']}"
        if video not in dur:
            cap = cv2.VideoCapture(str(ROOT / video))
            dur[video] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
            cap.release()
        for j, b in enumerate(x["breakpoint"]):
            t = float(b["time"])  # some questions are timed after the video ends: they see the whole video (end = its last full second)
            rows.append(dict(task="streambench", subtask=b["class"], video=video, start=0.0,
                             end=t if t <= dur[video] else float(math.floor(dur[video])), time=t, question=b["question"],
                             answer=b["answer"], answer_text=b["answer"], index=v * 100 + j, sample_id=f"streambench-{v:03d}-{j}",
                             options={}, eval_mode="substring_match", item=v, domain=x["info"]["class_1"], category=x["info"]["class_2"]))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT / "rows.json", "w"))
    missing = sorted({r["video"] for r in rows if not (ROOT / r["video"]).exists()})
    print(f"{len(rows)} questions, {len({r['item'] for r in rows})} videos, missing videos {len(missing)} {missing[:3]}",
          collections.Counter(r["subtask"] for r in rows), "timed after the video end:", sum(r["end"] < r["time"] for r in rows))


def by_item(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["item"]].append(r)
    return sorted(d.values(), key=lambda rs: -max(r["end"] for r in rs))


def caption(args):
    import onestreamer.caption_optimized as CO
    from onestreamer.engine import OneStreamerEngine
    from onestreamer.schema import Sample
    rows = json.load(open(OUT / "rows.json"))
    cfg = json.load(open(C.EVAL / "Proactive_Eval/OneStreamer/configs/streamingbench.json"))["caption"]
    eng = OneStreamerEngine(model_path=str(C.MODELS / "OneStreamer-4B"), dtype="bfloat16", attn_implementation="flash_attention_2",
                            min_pixels=3136, max_pixels=100352)
    (OUT / "memory").mkdir(exist_ok=True)
    mine = by_item(rows)  # longest first; workers claim videos with a lock file, so workers can be added at any time
    for i, rs in enumerate(mine):
        dest = OUT / "memory" / f"item-{rs[0]['item']:03d}.json"
        if dest.exists():
            continue
        try:
            os.close(os.open(dest.with_suffix(".lock"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        last = max(rs, key=lambda r: r["end"])
        sample = Sample.from_dict({k: v for k, v in last.items() if k not in ("item", "domain", "category", "time")})
        t0 = time.time()
        res = CO.run_shared_caption_pass(eng, sample, ROOT / sample.video, sorted({r["end"] for r in rs}), cfg)
        json.dump(res, open(dest.with_suffix(".tmp"), "w"))
        dest.with_suffix(".tmp").rename(dest)
        print(f"shard {args.shard}: {i + 1}/{len(mine)} video {rs[0]['item']} to {sample.end:.0f}s in {time.time() - t0:.0f}s", flush=True)
    print("CAPTION_DONE", args.shard, flush=True)


def memory_fn():
    from onestreamer.caption_optimized import boundary_key
    cache = {}

    def mem(r):
        p = OUT / "memory" / f"item-{r['item']:03d}.json"
        if p not in cache:
            cache.clear()
            cache[p] = json.load(open(p))
        return cache[p]["boundary_results"][boundary_key(float(r["end"]))]["events"]
    return mem


def judge_prompt(question, answer, prediction):
    """eval_ego_streaming_with_llama3.py: messages and prepare_prompt, verbatim."""
    messages = [
        {"role": "system", "content":
            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer."},
        {"role": "user", "content":
            "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer: {prediction}\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'llama_pred' and 'score', where value of 'llama_pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'llama_pred': 'yes', 'score': 4.8}."}]
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts += ["<|start_header_id|>", m["role"], "<|end_header_id|>", "\n\n", m["content"], "<|eot_id|>"]
    return " ".join(parts)


def parse_judge(text):
    out = None
    for m in re.findall(r"\{.*?\}", text):
        try:
            out = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            pass
    if not isinstance(out, dict) or "llama_pred" not in out or "score" not in out:
        return None, None
    return str(out["llama_pred"]).strip().lower(), float(out["score"])


def judge(args):
    """Judge every answer row in --src/results_*.jsonl; writes --src/judged_<shard>.jsonl (resumable)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = {r["sample_id"]: r for r in json.load(open(OUT / "rows.json"))}
    src = Path(args.src)
    ans = [json.loads(l) for p in sorted(glob.glob(str(src / "results_*.jsonl"))) for l in open(p) if l.strip()]
    ans = ans[args.shard::args.nshards]
    key = lambda a: (a["sample_id"], a["variant"], a.get("max_side", 0), a.get("select"), a.get("source_latest", 0))
    dest = src / f"judged_{args.shard:02d}.jsonl"
    done = {key(json.loads(l)) for l in open(dest)} if dest.exists() else set()
    todo = [a for a in ans if key(a) not in done]
    print(f"judge shard {args.shard}: {len(ans)} answers, {len(todo)} to judge", flush=True)
    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
    with open(dest, "a") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            prompts = [judge_prompt(rows[a["sample_id"]]["question"], rows[a["sample_id"]]["answer"], a["prediction"]) for a in batch]
            x = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                y = model.generate(**x, max_new_tokens=64, do_sample=False, temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
            for a, out in zip(batch, tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)):
                pred, score = parse_judge(out)
                f.write(json.dumps(dict(a, judge_raw=out, llama_pred=pred, score=score)) + "\n")
            f.flush()
            print(f"judge shard {args.shard}: {min(i + args.batch, len(todo))}/{len(todo)}", flush=True)
    print("JUDGE_DONE", args.shard, flush=True)


def report(args):
    """Accuracy (share of 'yes'), score and coherence per configuration, per type, memory types vs the rest."""
    import numpy as np
    rows = json.load(open(OUT / "rows.json"))
    order = {r["sample_id"]: i for i, r in enumerate(rows)}
    rng = np.random.default_rng(0)
    out = []
    for src in args.src.split(","):
        js = [json.loads(l) for p in sorted(glob.glob(str(Path(src) / "judged_*.jsonl"))) for l in open(p) if l.strip()]
        cfgs = collections.defaultdict(list)
        for j in js:
            cfgs[(Path(src).name, j.get("max_side", 0), j.get("select") or "-", j.get("source_latest", 0), j["variant"])].append(j)
        for cfg, rs in sorted(cfgs.items(), key=str):
            rs.sort(key=lambda j: order[j["sample_id"]])
            bad = sum(j["llama_pred"] not in ("yes", "no") for j in rs)
            acc = np.array([j["llama_pred"] == "yes" for j in rs], float)
            sc = np.array([j["score"] or 0.0 for j in rs])
            boot = acc[rng.integers(0, len(acc), (10000, len(acc)))].mean(1) * 100
            row = dict(run=cfg[0], max_side=cfg[1] or "source", select=cfg[2], source_latest=cfg[3], method=cfg[4], n=len(rs), unparsed=bad,
                       acc=round(100 * acc.mean(), 2), acc_ci95_lo=round(float(np.percentile(boot, 2.5)), 2),
                       acc_ci95_hi=round(float(np.percentile(boot, 97.5)), 2), score=round(float(sc.mean()), 3),
                       coherence=round(float(np.abs(np.diff(sc)).mean()), 3))
            for name, keep in (("memory", lambda t: t in MEMORY_TYPES), ("other", lambda t: t not in MEMORY_TYPES)):
                m = np.array([keep(j["subtask"]) for j in rs])
                row[f"acc_{name}"], row[f"score_{name}"] = round(100 * acc[m].mean(), 2), round(float(sc[m].mean()), 3)
            for t in ("OS", "LM", "SM", "CI", "KG", "SF"):
                m = np.array([j["subtask"] == t for j in rs])
                row[f"acc_{t}"], row[f"score_{t}"] = round(100 * acc[m].mean(), 2), round(float(sc[m].mean()), 3)
            out.append(row)
            print(row, flush=True)
    if out:
        import csv
        dest = Path(__file__).resolve().parents[1] / "results/streambench.csv"
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print("wrote", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["build", "caption", "judge", "report"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--src", default="", help="judge: one answer dir; report: comma list of answer dirs")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    {"build": build, "caption": caption, "judge": judge, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
