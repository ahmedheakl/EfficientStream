"""Frame bank for question-time frame retrieval: one frame per second of each benchmark video, with a CLIP image embedding and a
sharpness score, as a live system would store them while watching (built offline here; retrieval only ever uses frames before the
question time).

Build (--bench sb | ovo, OVO covers the forward videos too): for each video, frames at t = 0, 1, 2, ... s up to its last question
time (frame index round(t * fps), decoded sequentially with OpenCV), CLIP ViT-H/14 (laion2B) image embeddings, L2-normalised, stored
as float16, and sharpness = variance of the Laplacian of the grey frame resized to 320 px wide (higher = sharper). One .npz per video
in out/frame_bank/<bench>/, plus the embedding time per frame (embed_ms_per_frame, batches of 64; a live system embeds one frame per second).

Selection: candidates are bank frames in [lo, hi), i.e. after the video start and before the recent window; the `pool` (32) most
similar to the question (cosine of CLIP embeddings) are scored and taken greedily, skipping any frame within `gap` (3) seconds of one
already taken, until k frames; returned in time order. Scores (rank_scores):
  clip   CLIP similarity only (no reranking)
  sharp  z(similarity) + 0.3 * z(log(1 + sharpness)) (select uses this)
  vlm    Qwen3-VL-Reranker-2B relevance of each candidate frame to the question (Reranker: yes-minus-no logit of the last token, as the
         model card's scripts/qwen3_vl_reranker.py; all candidates in one batch, frames at the frame-cache size)
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

CLIP = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
RERANKER = "Qwen/Qwen3-VL-Reranker-2B"
RERANK_SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                 'Note that the answer can only be "yes" or "no".')
RERANK_INSTRUCT = "Given a question about a live video, retrieve the video frames that show the visual evidence needed to answer it."
OUT = Path(__file__).resolve().parent / "out/frame_bank"
BANK_OF = {"ovo": "ovo", "ovo_fwd": "ovo", "sb": "sb", "streambench": "streambench", "odv": "odv", "ovb": "ovb"}  # answer_variants bench -> bank folder


def load_clip(device="cuda"):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP, dtype=torch.float16).to(device).eval()
    return model, CLIPProcessor.from_pretrained(CLIP)


def embed_text(model, proc, text):
    import torch
    x = proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(model.device)
    with torch.inference_mode():
        e = model.get_text_features(**x).float()
    return (e / e.norm(dim=-1, keepdim=True))[0].cpu().numpy()


def embed_images(model, proc, images):
    import torch
    x = proc(images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        e = model.get_image_features(pixel_values=x["pixel_values"].half()).float()
    return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)


def sharpness(frame_bgr):
    import cv2
    h, w = frame_bgr.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame_bgr, (320, max(1, round(h * 320 / w)))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def read_at(path, times):
    """Frames at the given times (s), frame index round(t * fps), as (t, PIL image)."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
            ok, fr = cap.read()
            if ok:
                out.append((float(t), Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
    finally:
        cap.release()
    return out


def candidates(bank, q, lo, hi, pool=32):
    """The `pool` bank frames in [lo, hi) most similar to the question, most similar first: (times, similarities, log(1 + sharpness))."""
    m = (bank["ts"] >= lo) & (bank["ts"] < hi)
    ts, sim, sh = bank["ts"][m], bank["emb"][m].astype(np.float32) @ q, np.log1p(bank["sharp"][m])
    top = np.argsort(-sim)[:pool]
    return ts[top], sim[top], sh[top]


def rank_scores(how, sim, sh, rerank=None, beta=0.3):
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    if how == "vlm":
        return np.asarray(rerank, np.float32)
    return z(sim) + beta * z(sh) if how == "sharp" and len(sim) > 1 else sim


def pick(ts, score, k, gap=3.0):
    """Indices of up to k candidates taken greedily by score, skipping any within `gap` s of one already taken; in time order."""
    chosen = []
    for i in np.argsort(-score, kind="stable"):
        if all(abs(ts[i] - ts[j]) >= gap for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda i: ts[i])


def spread(bank, lo, hi, k, gap=3.0):
    """Times (s) of up to k bank frames in [lo, hi), the frame nearest the centre of each of k equal segments (question-agnostic
    coverage of the earlier video), at least `gap` s apart, in time order."""
    ts = bank["ts"][(bank["ts"] >= lo) & (bank["ts"] < hi)]
    out = []
    for c in lo + (np.arange(k) + 0.5) * (hi - lo) / k if len(ts) else []:
        t = float(ts[np.abs(ts - c).argmin()])
        if all(abs(t - u) >= gap for u in out):
            out.append(t)
    return out


def select(bank, q, lo, hi, k, pool=32, beta=0.3, gap=3.0):
    """Times (s) and similarities of up to k frames in [lo, hi), CLIP candidates reranked with sharpness; see module docstring."""
    ts, sim, sh = candidates(bank, q, lo, hi, pool)
    idx = pick(ts, rank_scores("sharp", sim, sh, beta=beta), k, gap)
    return [float(ts[i]) for i in idx], [float(sim[i]) for i in idx]


class Reranker:
    """Qwen3-VL-Reranker-2B as a cross-encoder over (question, frame) pairs; score = (h_last . (w_yes - w_no)), sigmoid-monotone."""

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        lm = Qwen3VLForConditionalGeneration.from_pretrained(RERANKER, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        lm = lm.to(device).eval()
        self.model, self.proc = lm.model, AutoProcessor.from_pretrained(RERANKER, padding_side="left", use_fast=True)
        v = self.proc.tokenizer.get_vocab()
        self.w = (lm.lm_head.weight[v["yes"]] - lm.lm_head.weight[v["no"]]).detach().float()

    def score(self, question, images):
        import torch
        chats = [[{"role": "system", "content": [{"type": "text", "text": RERANK_SYSTEM}]},
                  {"role": "user", "content": [{"type": "text", "text": "<Instruct>: " + RERANK_INSTRUCT}, {"type": "text", "text": "<Query>:"},
                                               {"type": "text", "text": question}, {"type": "text", "text": "\n<Document>:"},
                                               {"type": "image", "image": im}]}] for im in images]
        text = [self.proc.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
        x = self.proc(text=text, images=list(images), padding=True, do_resize=False, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            h = self.model(**x).last_hidden_state[:, -1]
        return (h.float() @ self.w).cpu().numpy()


def load_bank(bench, video):
    d = np.load(OUT / bench / (video.replace("/", "__") + ".npz"))
    return {k: d[k] for k in ("ts", "emb", "sharp")}


def build(args):
    import cv2
    from PIL import Image
    import answer_variants as AV
    ends = collections.defaultdict(float)
    for b in [k for k, v in BANK_OF.items() if v == args.bench]:
        rows, vroot, _ = AV.load_bench(b)
        for r in rows:
            ends[r["video"]] = max(ends[r["video"]], float(r["end"]))
    out = OUT / args.bench
    out.mkdir(parents=True, exist_ok=True)
    model, proc = load_clip()
    stats = open(out / f"embed_ms_{args.shard:02d}.jsonl", "a")
    for vid in sorted(ends)[args.shard::args.nshards]:
        dst = out / (vid.replace("/", "__") + ".npz")
        if dst.exists():
            continue
        cap = cv2.VideoCapture(str(vroot / vid))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {round(t * fps): float(t) for t in range(int(ends[vid]) + 1) if round(t * fps) < n}
        ts, imgs, sharp, embs, i, spent = [], [], [], [], 0, 0.0

        def flush():
            nonlocal imgs, spent
            if imgs:
                t0 = time.perf_counter()
                embs.append(embed_images(model, proc, imgs))
                spent += time.perf_counter() - t0
                imgs = []
        while i <= max(want, default=-1):
            if i in want:
                ok, fr = cap.read()
                if not ok:
                    break
                ts.append(want[i])
                sharp.append(sharpness(fr))
                imgs.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                if len(imgs) == 64:
                    flush()
            elif not cap.grab():
                break
            i += 1
        flush()
        cap.release()
        ms = 1000 * spent / max(1, len(ts))
        np.savez(dst, ts=np.array(ts, np.float32), emb=np.concatenate(embs) if embs else np.zeros((0, 1024), np.float16),
                 sharp=np.array(sharp, np.float32))
        stats.write(json.dumps(dict(video=vid, frames=len(ts), embed_ms_per_frame=ms)) + "\n")
        stats.flush()
    print("BANK_DONE", args.shard, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["sb", "ovo", "streambench", "odv", "ovb"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    build(ap.parse_args())
