"""ES (results/HYBRID_RULES.md): the teammate's streaming Qwen3.5-0.8B (EfficientStream, proactive/stream_qwen.py) on our analysis halves.
Settings:
  s16   his ask(): the 16 most recent frames, 1 fps, source pixels, as still images (OneStreamer's "recent frames" setting)
  v2fl  ours: the last 16 s at 2 fps as one video (frames capped at --side px, his video path with mm_token_type_ids) + the newest
        frame again as a still at source pixels; same prompt text and question format as his ask()
Rows come from out/es/rows_<bench>.json (exported with our loader); predictions are scored later in the onestreamer-eval env.
Usage (malow env, transformers 5): python es_eval.py --bench sb --setting s16 --out out/es/s16 [--sub stream-qwen3.5-0.8b-r4]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ES = Path.home() / "onestreamer_repro/EfficientStream/proactive"
sys.path.insert(0, str(ES))
import stream_qwen as SQ  # noqa: E402


def load_vr():
    """EXP6: the context-and-visual-memory branch of stream_qwen (es_vr), for its _ask_with_past (the past as one video)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("stream_qwen_vr", Path.home() / "onestreamer_repro/es_vr/proactive/stream_qwen.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def read_seq(video, times):
    """EXP6: frames at increasing times, decoded forward after one seek (folio_mem.SeqReader): [(t, PIL)]."""
    import cv2
    from PIL import Image
    cap, out = cv2.VideoCapture(video), []
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if times[0] > 1:
        cap.set(cv2.CAP_PROP_POS_MSEC, (times[0] - 0.5) * 1000)
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for t in times:
        want = min(int(round(t * fps)), n - 1) if n else int(round(t * fps))
        ok = True
        while pos < want and ok:
            ok = cap.grab(); pos += 1
        ok, fr = cap.read(); pos += 1
        if ok and fr is not None:
            out.append((t, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def ask_mem_past(VR, self, frames, question, options, max_new_tokens, past, mem_text):
    """KV-BUILD: es_vr StreamQwen._ask_with_past with the caption memory text kept in front of the past video."""
    from PIL import ImageOps
    tok, video_proc, image_proc = self.proc.tokenizer, self.proc.video_processor, self.proc.image_processor
    pair, merge = int(video_proc.temporal_patch_size), int(video_proc.merge_size) ** 2
    times, imgs = [t for t, _ in past], [im.convert("RGB") for _, im in past]
    while len(imgs) % pair:
        imgs.append(imgs[-1]); times.append(times[-1])
    video = video_proc(videos=[torch.stack([torch.from_numpy(np.array(x)).permute(2, 0, 1) for x in imgs])],
                       do_sample_frames=False, do_resize=False, return_tensors="pt")
    grid = video["video_grid_thw"][0]
    pads = "<|video_pad|>" * (int(grid[1] * grid[2]) // merge)
    blocks = "".join(f"<{sum(times[i:i + pair]) / pair:.1f} seconds><|vision_start|>{pads}<|vision_end|>" for i in range(0, len(times), pair))
    step = (past[-1][0] - past[0][0]) / max(1, len(past) - 1)
    memory = mem_text + (f"<LONG_TERM_VIDEO_MEMORY>\nA low-resolution video of the stream before the recent frames, from {past[0][0]:.0f} s "
                         f"to {past[-1][0]:.0f} s, one frame every {step:.1f} s, ordered from oldest to newest.\n{blocks}</LONG_TERM_VIDEO_MEMORY>")
    f = image_proc.patch_size * image_proc.merge_size
    padded = [ImageOps.expand(im, border=(0, 0, -im.size[0] % f, -im.size[1] % f), fill=0) for _, im in frames]
    images = image_proc(images=padded, do_resize=False, return_tensors="pt")
    stills = "".join(f"<{t:.3f}s><|vision_start|>" + "<|image_pad|>" * (int(g.prod()) // merge) + "<|vision_end|>"
                     for (t, _), g in zip(frames, images["image_grid_thw"]))
    q = question if options is None else "\n".join([f"Question:\n{question}", "Options:", *options, VR.POST_PROMPT, "Best option:("])
    user = (memory + "<RECENT_VISUAL_CONTEXT>\nThe following frames are ordered from oldest to newest. They retain their decoded source "
            "pixels; only minimal border padding may be added to align them to the model patch grid." + stills + "</RECENT_VISUAL_CONTEXT>\n" + q)
    chat = [{"role": "system", "content": VR.QA_SYSTEM if options else VR.QA_DIRECT_SYSTEM}, {"role": "user", "content": user}]
    x = tok(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False)
    x["pixel_values"], x["image_grid_thw"] = images["pixel_values"], images["image_grid_thw"]
    x["pixel_values_videos"], x["video_grid_thw"] = video["pixel_values_videos"], video["video_grid_thw"]
    ids = x["input_ids"]
    x["mm_token_type_ids"] = (ids == tok.convert_tokens_to_ids("<|image_pad|>")).int() + 2 * (ids == tok.convert_tokens_to_ids("<|video_pad|>")).int()
    return self._generate(x, max_new_tokens)


def store_times(times, budget):
    """EXP6: the times a store of `budget` frames keeps after admitting `times` in order (answer_variants.bounded_store_times)."""
    kept = []
    for t in times:
        kept.append(t)
        if len(kept) > budget:
            hi = len(kept) - 1
            kept.pop(1 if hi <= 1 else 1 + int(np.argmin([kept[j + 1] - kept[j - 1] for j in range(1, hi)])))
    return kept

V2FL_TEXT = ("<RECENT_VISUAL_CONTEXT>\nThe following video clip shows the most recent frames, ordered from oldest to newest, two frames "
             "per second; they have been resized while preserving aspect ratio.\n")


def resize_long(im, side):
    from PIL import Image
    w, h = im.size
    s = min(1.0, side / max(w, h))
    return im.resize((max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)), Image.BICUBIC)


def body_of(line):  # clm_check.body_of
    import re
    return re.sub(r"^(\[[^\]]*\]\s*)+", "", line).strip()


def compact(events, qtime):
    """Our compaction, as consolidate_eval.merge_repeats(consolidate(ev, keep_after=qtime - 60)): a summary replaces the captions it covers
    except those ending in the last 60 s; identical captions (time prefix removed) merge into one spanning their times."""
    sums = [e for e in events if e["kind"] == "summary" and e["text"]]
    inside = lambda e: any(float(x["start"]) <= float(e["start"]) and float(e["end"]) <= float(x["end"]) for x in sums)
    kept = [e for e in events if not (e["kind"] == "response" and inside(e) and not float(e["end"]) >= qtime - 60)]
    out, seen = [], {}
    for e in kept:
        if e["kind"] != "response":
            out.append(e); continue
        b = body_of(e["text"])
        if b in seen:
            m_ = seen[b]
            m_["start"], m_["end"], m_["text"] = min(float(m_["start"]), float(e["start"])), max(float(m_["end"]), float(e["end"])), b
        else:
            seen[b] = dict(e); out.append(seen[b])
    return out


def read_at(video, times):
    """Frames at the given times (s), source pixels: [(t, PIL)]."""
    import cv2
    from PIL import Image
    cap, out = cv2.VideoCapture(video), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if ok:
            out.append((float(t), Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    cap.release()
    return out


def question_text(question, options):
    return question if not options else "\n".join([f"Question:\n{question}", "Options:", *options, SQ.POST_PROMPT, "Best option:("])


def ask_v2fl(m, frames, question, options, side=768, max_new_tokens=16):
    """frames: [(t, PIL)] the last 16 s at 2 fps (oldest first), source pixels."""
    from PIL import ImageOps
    proc, tok = m.proc, m.proc.tokenizer
    vp_ = proc.video_processor; T = int(vp_.temporal_patch_size)
    small = [resize_long(im, side) for _, im in frames]
    ts = [t for t, _ in frames]
    while len(small) % T:
        small.append(small[-1]); ts.append(ts[-1])
    vt = torch.stack([torch.from_numpy(np.array(x.convert("RGB"))).permute(2, 0, 1) for x in small])
    vp = vp_(videos=[vt], do_sample_frames=False, do_resize=False, return_tensors="pt")
    g = vp["video_grid_thw"][0]; n = int(g[1] * g[2]) // (int(vp_.merge_size) ** 2)
    blocks = "".join(f"<{sum(ts[i:i + T]) / T:.1f} seconds><|vision_start|>" + "<|video_pad|>" * n + "<|vision_end|>" for i in range(0, len(ts), T))
    f = proc.image_processor.patch_size * proc.image_processor.merge_size
    t_last, last = frames[-1]
    w, h = last.size
    last = ImageOps.expand(last.convert("RGB"), border=(0, 0, -w % f, -h % f), fill=0)
    ip = proc.image_processor(images=[last], do_resize=False, return_tensors="pt")
    n_img = int(ip["image_grid_thw"][0].prod()) // (int(proc.image_processor.merge_size) ** 2)
    still = f"\nThe newest frame at full resolution:\n<{t_last:.3f}s><|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>"
    mem = "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>"
    user = mem + V2FL_TEXT + blocks + still + "</RECENT_VISUAL_CONTEXT>\n" + question_text(question, options)
    chat = [{"role": "system", "content": SQ.QA_SYSTEM if options else SQ.QA_DIRECT_SYSTEM}, {"role": "user", "content": user}]
    x = tok(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False)
    x["pixel_values_videos"], x["video_grid_thw"] = vp["pixel_values_videos"], vp["video_grid_thw"]
    x["pixel_values"], x["image_grid_thw"] = ip["pixel_values"], ip["image_grid_thw"]
    ids = x["input_ids"]
    x["mm_token_type_ids"] = (ids == tok.convert_tokens_to_ids("<|image_pad|>")).int() + (ids == tok.convert_tokens_to_ids("<|video_pad|>")).int() * 2
    n_vis = int((x["mm_token_type_ids"] > 0).sum())
    return m._generate(x, max_new_tokens), int(ids.shape[-1]), n_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--setting", choices=["s16", "v2fl", "s16ret", "phcm", "phcmc", "s16vm", "phcmret", "phcmvm", "phcmcret", "phcmretvm"], required=True)
    ap.add_argument("--mem-dir", default="", help="EXP3: per-question memory events (hismem/events/<model>)")
    ap.add_argument("--rows", default="", help="rows file (default out/es/rows_<bench>.json); EXP5 full sets: out/es/rows_all_<bench>.json")
    ap.add_argument("--ret-from", default="out/exp5", help="EXP5: OneStreamer-4B retrieval rows whose retrieved_ts are reused (model-independent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", default="stream-qwen3.5-0.8b-r4")
    ap.add_argument("--side", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="sdpa", help="sdpa on NVIDIA (transformers 5.6 flash-attention path fails here)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    rows = json.load(open(a.rows or f"out/es/rows_{a.bench}.json"))[a.shard::a.nshards]
    RET = {}
    if a.setting in ("s16ret", "phcmret", "phcmcret", "phcmretvm"):
        import glob as _g
        for f in _g.glob(f"{a.ret_from}/{a.bench}/results_*.jsonl"):
            for l in open(f):
                x = json.loads(l); RET[x["sample_id"]] = x.get("retrieved_ts") or []
    max_new = 128 if a.bench == "streambench" else 16
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_{a.shard}.jsonl"
    done = {json.loads(l)["sample_id"] for l in open(path)} if path.exists() else set()
    m = SQ.StreamQwen(str(Path.home() / "onestreamer_repro/path_to/models/efficientstream" / a.sub), attn=a.attn)
    gen, tok = m._generate, m.proc.tokenizer
    pads = [tok.convert_tokens_to_ids(x) for x in ("<|image_pad|>", "<|video_pad|>")]

    def counted(inputs, max_new_tokens):  # record prompt and visual token counts of every call
        ids = inputs["input_ids"]
        m.last_counts = (int(ids.shape[-1]), int(sum((ids == p).sum() for p in pads)))
        return gen(inputs, max_new_tokens)
    m._generate = counted
    VR = load_vr() if a.setting in ("s16vm", "phcmvm", "phcmretvm") else None
    with open(path, "a") as fo:
        for r in rows:
            if r["sample_id"] in done:
                continue
            t0 = time.perf_counter()
            if a.setting == "s16":
                frames = SQ.StreamQwen.recent_frames(r["video"], r["end"], n=16, fps=1.0, start=r["start"])
                pred = m.ask(frames, r["question"], r["options"], max_new_tokens=max_new)
                n_in, n_vis = m.last_counts
            elif a.setting in ("phcm", "phcmc", "phcmret", "phcmvm", "phcmcret", "phcmretvm"):  # EXP3 / KV-BUILD: his memory (+ compaction / retrieval / past)
                frames = SQ.StreamQwen.recent_frames(r["video"], r["end"], n=16, fps=1.0, start=r["start"])
                p_ = Path(a.mem_dir) / f"{r['sample_id']}.json"
                ev = [e for e in json.load(open(p_))["events"] if e["kind"] in ("response", "summary") and e["text"]] if p_.exists() else []
                if a.setting in ("phcmc", "phcmcret"):
                    ev = compact(ev, float(r["end"]))
                mem = [dict(kind="observe" if e["kind"] == "response" else "summary", start=float(e["start"]), end=float(e["end"]), text=e["text"])
                       for e in sorted(ev, key=lambda e: (float(e["end"]), e["kind"]))]
                past, t_w = [], float(frames[0][0])  # first recent frame
                if a.setting in ("phcmret", "phcmcret", "phcmretvm"):  # KV-BUILD: the Exp5 retrieved frames (384 px) before the 16 recent
                    frames = [(t, resize_long(im, 384)) for t, im in read_at(r["video"], sorted(RET.get(r["sample_id"], [])))] + frames
                if a.setting in ("phcmvm", "phcmretvm"):  # KV-BUILD: the 128 s before the recent frames at 8 fps, <=1024 frames, 320 px
                    t_s = max(float(r["start"]), t_w - 128.0)
                    times = [float(x) for x in np.arange(t_s, t_w - 1e-6, 1.0 / 8)]
                    if len(times) > 1024:
                        times = [times[i] for i in np.linspace(0, len(times) - 1, 1024).round().astype(int)]
                    past = [(t, resize_long(im, 320)) for t, im in read_seq(r["video"], times)] if len(times) >= 2 else []
                if len(past) >= 2:
                    pred = ask_mem_past(VR, m, frames, r["question"], r["options"], max_new, past, m.render_memory(mem, r["question"]) if mem else "")
                else:
                    pred = m.ask(frames, r["question"], r["options"], max_new_tokens=max_new, memory=mem)
                n_in, n_vis = m.last_counts
                mem_tok = len(m.proc.tokenizer.encode(m.render_memory(mem, r["question"]), add_special_tokens=False)) if mem else 0
            elif a.setting == "s16ret":  # EXP5: the 8 frames retrieved for this question (CLIP + sharpness), 384 px, before the 16 recent
                frames = SQ.StreamQwen.recent_frames(r["video"], r["end"], n=16, fps=1.0, start=r["start"])
                ret = read_at(r["video"], sorted(RET.get(r["sample_id"], [])))
                frames = [(t, resize_long(im, 384)) for t, im in ret] + frames
                pred = m.ask(frames, r["question"], r["options"], max_new_tokens=max_new)
                n_in, n_vis = m.last_counts
            elif a.setting == "s16vm":  # EXP6: the past before the 16 recent (1 fps into a 128-frame store, 320 px) as one video
                frames = SQ.StreamQwen.recent_frames(r["video"], r["end"], n=16, fps=1.0, start=r["start"])
                t_w, t_s = float(frames[0][0]), float(r["start"])
                times = store_times([float(x) for x in np.arange(t_s, t_w - 1e-6, 1.0)], 128)
                past = [(t, resize_long(im, 320)) for t, im in read_seq(r["video"], times)] if len(times) >= 2 else []
                if len(past) >= 2:
                    pred = VR.StreamQwen._ask_with_past(m, frames, r["question"], r["options"], max_new, past)
                else:
                    pred = m.ask(frames, r["question"], r["options"], max_new_tokens=max_new)
                n_in, n_vis = m.last_counts
            else:
                frames = SQ.StreamQwen.recent_frames(r["video"], r["end"], n=32, fps=2.0, start=r["start"])
                pred, n_in, n_vis = ask_v2fl(m, frames, r["question"], r["options"], side=a.side)
            fo.write(json.dumps(dict(sample_id=r["sample_id"], subtask=r["subtask"], setting=a.setting, variant=a.setting, sub=a.sub, prediction=pred,
                                     n_frames=len(frames), n_past=len(past) if a.setting in ("s16vm", "phcmvm", "phcmretvm") else 0, input_tokens=n_in, visual_tokens=n_vis,
                                     memory_tokens=mem_tok if a.setting in ("phcm", "phcmc", "phcmret", "phcmvm", "phcmcret", "phcmretvm") else 0,
                                     seconds=round(time.perf_counter() - t0, 3))) + "\n")
            fo.flush()
    print("ES_EVAL_DONE", a.bench, a.setting, flush=True)


if __name__ == "__main__":
    main()
