"""ST (results/HYBRID_RULES.md): build the FOLIO-style memory (folio.py) for analysis-half videos. Videos are processed together: at each
step every active video writes its next 8-s segment, and the writer calls of one step are batched. A video's memory runs from the earliest
start to the latest end of its questions. One JSON per video: <out>/<md5(video path)[:16]>.json.
Usage (from clm): python folio_mem.py --bench ovo,sb --out out/st/mem_qwen4b --writer Qwen3-VL-4B-Instruct [--videos keys.json]"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import folio as F


class SeqReader:
    """Forward-only frame reader: frames at increasing times, decoding sequentially (one seek at the start)."""

    def __init__(self, path, t_start):
        import cv2
        self.cv2, self.cap = cv2, cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if t_start > 1:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, (t_start - 0.5) * 1000)
        self.pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def at(self, times):
        from PIL import Image
        out = []
        for t in times:
            want = int(round(t * self.fps))
            if self.n and want >= self.n:
                want = self.n - 1
            ok, fr = True, None
            while self.pos < want and ok:
                ok = self.cap.grab(); self.pos += 1
            ok, fr = self.cap.read()
            self.pos += 1
            if ok and fr is not None:
                out.append((t, Image.fromarray(self.cv2.cvtColor(fr, self.cv2.COLOR_BGR2RGB))))
        return out


def resize_long(im, side):
    from PIL import Image
    w, h = im.size
    s = min(1.0, side / max(w, h))
    return im.resize((max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)), Image.BICUBIC)


def video_inputs(proc, clip, prompt):
    """One writer call with the clip as a video: pairs of frames per temporal block, each block labelled with its mean time."""
    import torch
    vp = proc.video_processor
    temporal = int(vp.temporal_patch_size)
    ts = [t for t, _ in clip]; fr = [im.convert("RGB") for _, im in clip]
    while len(fr) % temporal:
        fr.append(fr[-1]); ts.append(ts[-1])
    vt = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in fr])
    v = vp(videos=[vt], do_sample_frames=False, do_resize=False, return_tensors="pt")
    g = v["video_grid_thw"][0]
    per = int(g[1] * g[2]) // (int(vp.merge_size) ** 2)
    blocks = "".join(f"<{sum(ts[i:i + temporal]) / temporal:.1f} seconds><|vision_start|>" + "<|video_pad|>" * per + "<|vision_end|>"
                     for i in range(0, len(ts), temporal))
    chat = [{"role": "system", "content": F.WRITER_SYSTEM}, {"role": "user", "content": blocks + "\n" + prompt}]
    text = proc.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    x = dict(proc.tokenizer(text, return_tensors="pt"))
    x["pixel_values_videos"], x["video_grid_thw"] = v["pixel_values_videos"], v["video_grid_thw"]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ovo,ovo_fwd,sb,odv,ovb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--writer", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--videos", default="", help="JSON list of 'bench|video' keys (pilot); empty = all analysis-half videos")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--side", type=int, default=640, help="keyframe long side (px)")
    ap.add_argument("--writer-input", default="video", choices=["video", "keyframes"], help="video: the segment as a 2-fps clip (batch 1)")
    ap.add_argument("--video-side", type=int, default=448, help="writer clip long side (px)")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--keep-raw", action="store_true", help="store the writer's raw outputs (pilot inspection)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    import answer_variants as A
    import clm_check as C
    jobs = {}
    for part in args.bench.split(","):
        rows, vroot, _ = A.load_bench(part)
        for r in rows:
            if int(hashlib.md5(r["video"].encode()).hexdigest(), 16) % 2:
                continue  # analysis half only
            key = f"{part}|{r['video']}"
            j = jobs.setdefault(key, dict(key=key, bench=part, path=str(vroot / r["video"]), t_start=float(r.get("start") or 0), t_end=0.0))
            j["t_start"] = min(j["t_start"], float(r.get("start") or 0))
            j["t_end"] = max(j["t_end"], float(r["end"]))
    if args.videos:
        keep = set(json.load(open(args.videos)))
        jobs = {k: v for k, v in jobs.items() if k in keep}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    todo = sorted(jobs.values(), key=lambda j: j["key"])[args.shard::args.nshards]
    todo = [j for j in todo if not (out / (hashlib.md5(j["path"].encode()).hexdigest()[:16] + ".json")).exists()]
    print(f"{len(todo)} videos, {sum(j['t_end'] - j['t_start'] for j in todo) / 3600:.1f} h of video", flush=True)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = str(C.MODELS / args.writer)
    proc = AutoProcessor.from_pretrained(path)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()

    if args.writer_input == "video":
        args.batch = 1  # one clip per call; run several processes per GPU instead
    queue, active, t_all = list(todo), [], time.time()
    while queue or active:
        while queue and len(active) < args.batch:
            j = queue.pop(0)
            j.update(mem=F.Memory(), t=j["t_start"], reader=SeqReader(j["path"], j["t_start"]), calls=0, fails=0, tin=0, tout=0, raw=[],
                     seg_log=[])
            active.append(j)
        msgs, metas = [], []
        for j in active:
            t0 = j["t"]; t1 = min(t0 + F.SEG, j["t_end"])
            listed = j["mem"].listed()
            if args.writer_input == "video":  # the segment as a 2-fps clip, Qwen3-VL video path (as answer_variants.answer_video)
                clip = j["reader"].at(list(np.arange(t0 + 0.25, t1, 0.5)) or [t0])
                kts, chg = [t for t, _ in clip], 0.0
                x = video_inputs(proc, [(t, resize_long(im, args.video_side)) for t, im in clip], F.writer_prompt(t0, t1, kts, listed, video=True))
            else:
                probe = j["reader"].at(list(np.arange(t0 + 0.5, t1, 1.0)) or [t0])
                idx, chg = F.keyframes(probe)
                kf = [probe[i] for i in idx]
                kts = [t for t, _ in kf]
                content = []
                for t, im in kf:
                    content += [{"type": "text", "text": f"<{t:.1f} s>"}, {"type": "image", "image": resize_long(im, args.side)}]
                content.append({"type": "text", "text": F.writer_prompt(t0, t1, kts, listed)})
                msgs.append([{"role": "system", "content": [{"type": "text", "text": F.WRITER_SYSTEM}]}, {"role": "user", "content": content}])
            metas.append((j, t0, t1, kts, chg, 0))
        if args.writer_input != "video":
            x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True)
        x = {k: v.cuda() for k, v in x.items()}
        if "pixel_values_videos" in x:
            x["pixel_values_videos"] = x["pixel_values_videos"].to(model.dtype)
        with torch.inference_mode():
            y = model.generate(**x, max_new_tokens=args.max_new, do_sample=False)
        gen = y[:, x["input_ids"].shape[1]:]
        texts = proc.batch_decode(gen, skip_special_tokens=True)
        pad = proc.tokenizer.pad_token_id
        for b, ((j, t0, t1, kts, chg, npr), txt) in enumerate(zip(metas, texts)):
            d = F.parse_json(txt)
            j["calls"] += 1; j["fails"] += d is None
            j["tin"] += int((x["attention_mask"][b] > 0).sum()); j["tout"] += int((gen[b] != pad).sum())
            if d is not None:
                j["mem"].update(t0, t1, d)
            else:
                j["mem"].seg += 1
            j["seg_log"].append([round(t0, 2), round(t1, 2), kts, round(chg, 2)])
            if args.keep_raw:
                j["raw"].append(dict(t0=t0, t1=t1, out=txt))
            j["t"] = t1
        for j in [j for j in active if j["t"] >= j["t_end"] - 1e-6]:
            rec = dict(key=j["key"], bench=j["bench"], path=j["path"], t_start=j["t_start"], t_end=j["t_end"], writer=args.writer, seg=F.SEG,
                       side=args.side, calls=j["calls"], parse_fail=j["fails"], writer_in_tokens=j["tin"], writer_out_tokens=j["tout"],
                       segments=j["seg_log"], **j["mem"].dump())
            if args.keep_raw:
                rec["raw"] = j["raw"]
            json.dump(rec, open(out / (hashlib.md5(j["path"].encode()).hexdigest()[:16] + ".json"), "w"))
            j["reader"].cap.release()
            active.remove(j)
            print(f"{time.time() - t_all:7.0f}s done {j['key']} calls={j['calls']} fail={j['fails']} entities={len(j['mem'].entities)} "
                  f"events={len(j['mem'].events)}", flush=True)
    print("FOLIO_DONE", flush=True)


if __name__ == "__main__":
    main()
