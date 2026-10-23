"""Answer every question of a benchmark under each memory method, greedy or sampled with seeds (one script for all benchmarks).

Benchmarks (--bench):
  ovo      OVO-Bench backward + real-time, 1,468 questions; memory = OneStreamer's caption files of run-id full8
  ovo_fwd  OVO-Bench forward (REC, SSR, CRR), 1,567 probes; memory = shared passes in out/ovo_forward (ovo_forward.py)
  sb       StreamingBench real-time, 2,500 questions; memory = shared timelines of the authors' run-id sb4
  odv      ODV-Bench, 6,348 questions, no memory (V5);  ovb  OVBench, 7,090 questions, no memory (V6)
  streambench  StreamBench (StreamChat), 1,838 open-ended questions; memory = one shared caption pass per video (streambench.py);
           answers up to 128 new tokens, scored later by the LLM judge (streambench.py --phase judge), so hit is null here
Methods (--variants; definitions in results/methods.json): full, consolidated, keep60, hybrid_qa, hybrid, none.
Decoding: greedy by default (as the paper); --seeds 0,1,2 samples instead with the authors' sampled setting (temperature 0.7, top-p 0.8,
top-k 20, from their OVO-Timing evaluation), seed fixed per (seed, question) so all methods of a question share the same random stream.
Each row records the answer-time memory tokens, the question-agnostic stored memory tokens and the tokens added at answer time.
Answer frames: source pixels by default (as the paper); --max-side 768,512,384 also answers with the 16 frames resized so the long
side is at most that many pixels (aspect ratio kept, both sides rounded to the 32-px patch grid), one row per (method, max_side), frames
decoded once per question.
--retrieve K keeps 16 answer frames but only the 16 - K most recent; the other K are earlier frames chosen from the frame bank
(frame_bank.py: CLIP similarity to the question, reranked with sharpness, at least 3 s apart, before the recent window); if fewer than
K qualify, all 16 are recent frames. Rows record the retrieved times, similarities and retrieval time (retrieve_ms).
--recent N keeps N recent frames with the K retrieved (e.g. --recent 16 --retrieve 8: 24 frames; then fewer than K qualifying frames
simply means fewer retrieved frames); --ret-side S resizes the retrieved frames to an S-px long side (default: as --max-side);
--select clip,sharp,vlm answers once per way of scoring the 32 CLIP candidates (frame_bank.rank_scores: CLIP similarity only, with
sharpness, or Qwen3-VL-Reranker-2B on the candidates at the --ret-side size); rows record select, embed_ms, select_ms (scoring and
picking, incl. rerank_ms) and retrieve_ms = embed_ms + select_ms (the bank is loaded from disk outside the timer).
--select spread instead takes the K bank frames nearest the centres of K equal segments of the earlier video (frame_bank.spread),
whatever the question asks; retrieved_sim then holds their CLIP similarities to the question.
--split dev|test answers only the questions of videos in that half (md5 of the video path even = dev, as results/failure_analysis.py).
--source-latest K keeps the K most recent frames at source pixels and resizes only the older recent frames to --max-side (rows record
source_latest).
--earlier-block puts the retrieved frames in their own <EARLIER_KEYFRAMES> block before the recent frames, described as past moments,
and adds a third evidence item naming that block to the system prompt (OneStreamer's answer() puts every frame in <RECENT_VISUAL_CONTEXT>,
described as the latest visual state); questions with no retrieved frame get the input of a run without it (rows record earlier_block).
--save-conf records the answer's confidence (greedy only): conf_first = probability of the first generated token (softmax over the
vocabulary), conf = for option-letter questions the largest option probability at that token (softmax over the option letters), else
conf_first; tok_logp = log-probability of each generated token (results/HYBRID_RULES.md, G1).
--model NAME answers with another model under path_to/models (e.g. Qwen3-VL-8B-Instruct) with the same prompts and the same
OneStreamer-4B caption memory text; rows record model when it is not OneStreamer-4B. InternVL* models run through
internvl_engine.InternVLEngine (frames at source pixels tiled, resized frames one 448-px tile).
Rerunning with the same --out resumes: questions with all their rows are kept, the rest are answered again.
"""
import argparse
import collections
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

import clm_check as C
import folio as FO
import consolidate_eval as K

SB_RUN = C.EVAL / "outputs/OneStreamer-4B/streamingbench/sb4/memory/streamingbench"
SB_ROOT = C.ROOT / "path_to/datasets/StreamingBench"
FWD = Path(__file__).resolve().parent / "out/ovo_forward"


def resize_frames(frames, side):
    """Cap the long side at `side` px, keeping the aspect ratio; both sides rounded to the 32-px patch grid, so no padding is added."""
    from PIL import Image
    out = []
    for ts, im in frames:
        w, h = im.size
        s = min(1.0, side / max(w, h))
        r = im.resize((max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)), Image.BICUBIC)
        r.info["resized"] = 1  # read by internvl_engine (one tile for resized frames); unused by the Qwen engine
        out.append((ts, r))
    return out


EARLIER_ITEM = ("3. Keyframes from earlier in the video, before the recent frames, in <EARLIER_KEYFRAMES>. They show past moments, not the "
                "current state; use them only for what happened earlier.\n")


def capidx_select(fb, q_img, cap_sim, ev, lo, hi, beta=0.3):
    """VT: every bank frame in [lo, hi) scored by z(CLIP image similarity) + z(best text similarity of a caption covering it, +-1 s)
    + beta z(log sharpness); frames no caption covers get the caption term 0. -> (times, image similarities, scores)."""
    m = (fb["ts"] >= lo) & (fb["ts"] < hi)
    ts = fb["ts"][m]
    if not len(ts):
        return ts, np.zeros(0, np.float32), np.zeros(0, np.float32)
    img = fb["emb"][m].astype(np.float32) @ q_img
    sh = np.log1p(fb["sharp"][m])
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    caps = [e for e in ev if e["kind"] in ("response", "summary") and e["text"]]
    cs = np.full(len(ts), np.nan, np.float32)
    if caps:
        for e, v in zip(caps, cap_sim([C.body_of(e["text"]) for e in caps])):
            w = (ts >= float(e["start"]) - 1) & (ts <= float(e["end"]) + 1)
            cs[w] = np.fmax(cs[w], v)
    ok = np.isfinite(cs)
    cz = np.zeros(len(ts), np.float32)
    if ok.sum() > 1:
        cz[ok] = (cs[ok] - cs[ok].mean()) / (cs[ok].std() + 1e-6)
    return ts, img, z(img) + cz + beta * z(sh)


def motion_window(frames, n=8, side=320, min_energy=2.0, min_share=0.4, center=False):
    """MZ (results/HYBRID_RULES.md): the half-width x half-height window with the most ego-motion-compensated frame difference over the
    last n recent frames (oldest first). Returns (box in source pixels or None, stats). center=True (CZ control): zoom on the same
    questions, but always on the centred window."""
    import cv2
    import numpy as np
    fr = frames[-n:]
    if len(fr) < 2:
        return None, dict(zoomed=0)
    W, H = fr[0][1].size
    s = side / max(W, H)
    w, h = max(4, round(W * s)), max(4, round(H * s))
    g = [cv2.resize(cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA) for _, im in fr]
    orb, bf = cv2.ORB_create(500), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    acc, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for a, b in zip(g[:-1], g[1:]):  # warp the earlier frame onto the later one, difference over the overlap
        Hm = None
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is not None and db is not None and len(ka) >= 8 and len(kb) >= 8:
            m = bf.match(da, db)
            if len(m) >= 8:
                Hm, _ = cv2.findHomography(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]), cv2.RANSAC, 3.0)
        if Hm is None:
            Hm = np.eye(3)
        va = cv2.warpPerspective(np.ones_like(a, np.float32), Hm, (w, h))
        acc += np.abs(cv2.warpPerspective(a, Hm, (w, h)).astype(np.float32) - b.astype(np.float32)) * va
        cnt += va
    E = cv2.blur(acc / np.maximum(cnt, 1), (9, 9))
    wh, ww = h // 2, w // 2
    I = np.pad(E, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    S = I[wh:, ww:] - I[:-wh, ww:] - I[wh:, :-ww] + I[:-wh, :-ww]  # S[y, x] = energy of E[y:y + wh, x:x + ww]
    y, x = np.unravel_index(int(np.argmax(S)), S.shape)
    tot, mean = float(E.sum()), float(E.mean())
    share = float(S[y, x]) / tot if tot > 0 else 0.0
    ok = mean >= min_energy and share >= min_share
    if center:
        y, x = (h - wh) // 2, (w - ww) // 2
    box = (round(x / s), round(y / s), min(W, round((x + ww) / s)), min(H, round((y + wh) / s)))
    st = dict(zoomed=int(ok), energy=round(mean, 3), share=round(share, 4), box=list(box) if ok else None)
    return (box if ok else None), st


def zoom_clip(frames, box, side=768):
    """MZ: the box cropped from each frame at source pixels, resized so the long side is `side` (32-px grid)."""
    from PIL import Image
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    s = side / max(cw, ch)
    size = (max(32, round(cw * s / 32) * 32), max(32, round(ch * s / 32) * 32))
    return [(t, im.crop(box).resize(size, Image.BICUBIC)) for t, im in frames]


ZOOM_TEXT = "\nZoomed view (2x) of the region with the most motion in the last 4 seconds:\n"


def bounded_store_times(times, budget):
    """EXP6: the times kept by EfficientStream's BoundedStore (efficientstream/frames.py) after admitting `times` in order: over budget, the
    frame whose removal leaves the smallest gap between its neighbours goes; the first and the newest frame are never evicted."""
    kept = []
    for t in times:
        kept.append(t)
        if len(kept) > budget:
            hi = len(kept) - 1
            if hi <= 1:
                kept.pop(1)
            else:
                gaps = [kept[j + 1] - kept[j - 1] for j in range(1, hi)]
                kept.pop(1 + int(np.argmin(gaps)))
    return kept


def change_store_frames(reader, times, budget, side):
    """ACT: admit the frames at `times` in order (resized to `side`); over budget, drop the stored frame that differs least from the frame
    kept before it (mean absolute difference of 64x36 grayscale thumbnails); the first and the newest frame are never dropped."""
    from folio_mem import resize_long
    kept = []
    for t in times:
        got = reader.at([t])
        if not got:
            continue
        im = resize_long(got[0][1], side)
        kept.append((t, im, np.asarray(im.convert("L").resize((64, 36)), dtype=np.float32)))
        if len(kept) > budget:
            change = [np.abs(kept[j][2] - kept[j - 1][2]).mean() for j in range(1, len(kept) - 1)]
            kept.pop(1 + int(np.argmin(change)))
    return [(t, im) for t, im, _ in kept]


def past_block(eng, past, video_index_offset=0):
    """EXP6 / CV: the past as one low-res video for the memory slot; returns (text, video processor output)."""
    import torch
    vp = eng.processor.video_processor
    temporal = int(vp.temporal_patch_size)
    pts = [t for t, _ in past]; pfr = [im.convert("RGB") for _, im in past]
    while len(pfr) % temporal:
        pfr.append(pfr[-1]); pts.append(pts[-1])
    v = vp(videos=[torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in pfr])], do_sample_frames=False, do_resize=False,
           return_tensors="pt")
    g = v["video_grid_thw"][0]
    per = int(g[1] * g[2]) // (int(vp.merge_size) ** 2)
    step = (past[-1][0] - past[0][0]) / max(1, len(past) - 1)
    text = (f"<LONG_TERM_VIDEO_MEMORY>\nA low-resolution video of the stream before the recent frames, from {past[0][0]:.0f} s to "
            f"{past[-1][0]:.0f} s, one frame every {step:.1f} s, ordered from oldest to newest.\n"
            + "".join(f"<{sum(pts[i:i + temporal]) / temporal:.1f} seconds><|vision_start|>" + "<|video_pad|>" * per + "<|vision_end|>"
                      for i in range(0, len(pts), temporal)) + "</LONG_TERM_VIDEO_MEMORY>")
    return text, v


def answer_stills(eng, system, memory_text, recent, question_text, max_tokens, past=None, keep_memory=False):
    """EXP6: OneStreamerEngine.answer's prompt (recent frames as stills at source pixels, padded to the patch grid), built token by token
    so that the memory slot can hold the past as a video; past=None gives the plain 16-recent prompt through the same path."""
    import torch
    ipr = eng.processor.image_processor
    ims = [eng._pad_to_patch_grid(im.convert("RGB")) for _, im in recent]
    ip = ipr(images=ims, do_resize=False, return_tensors="pt")
    m2 = int(ipr.merge_size) ** 2
    stills = "".join(f"<{t:.3f}s><|vision_start|>" + "<|image_pad|>" * (int(g.prod()) // m2) + "<|vision_end|>"
                     for (t, _), g in zip(recent, ip["image_grid_thw"]))
    mem = memory_text or "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>"
    v = None
    if past:
        pm, v = past_block(eng, past)
        mem = memory_text + pm if keep_memory and memory_text else pm  # KV-BUILD: caption memory, then the past video
    user = (mem + "<RECENT_VISUAL_CONTEXT>\nThe following frames are ordered from oldest to newest. They retain their decoded source pixels; "
            "only minimal border padding may be added to align them to the model patch grid." + stills + "</RECENT_VISUAL_CONTEXT>\n" + question_text)
    tok = eng.processor.tokenizer
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False,
                                     add_generation_prompt=True)
    x = tok(prompt, return_tensors="pt")
    x["pixel_values"], x["image_grid_thw"] = ip["pixel_values"], ip["image_grid_thw"]
    if v is not None:
        x["pixel_values_videos"], x["video_grid_thw"] = v["pixel_values_videos"], v["video_grid_thw"]
    ids_v, ids_i = tok.convert_tokens_to_ids("<|video_pad|>"), tok.convert_tokens_to_ids("<|image_pad|>")
    eng.last_input_metrics = {"input_tokens": int(x["input_ids"].shape[-1]), "visual_tokens": int((x["input_ids"] == ids_v).sum() + (x["input_ids"] == ids_i).sum())}
    x = {k: t.to(eng.device) for k, t in x.items()}
    for k in ("pixel_values_videos", "pixel_values"):
        if k in x: x[k] = x[k].to(eng.model.dtype)
    with torch.inference_mode():
        y = eng.model.generate(**x, max_new_tokens=max_tokens, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, use_cache=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return eng._strip_end_tokens(eng.processor.batch_decode(y[:, x["input_ids"].shape[-1]:], skip_special_tokens=False)[0])


def answer_video(eng, system, memory_text, recent, question_text, max_tokens, last_still=None, zoom=None, past=None):
    """MV (results/HYBRID_RULES.md): the recent frames as one video (pairs of frames per temporal block, each block labelled with its
    mean time), same prompt blocks as OneStreamerEngine.answer otherwise. Frames must already share one size."""
    import numpy as np
    import torch
    vp = eng.processor.video_processor
    temporal = int(vp.temporal_patch_size)
    ts = [t for t, _ in recent]; fr = [im.convert("RGB") for _, im in recent]
    while len(fr) % temporal:
        fr.append(fr[-1]); ts.append(ts[-1])
    vt = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in fr])
    vids = [vt]
    if past:  # CV: the stream before the recent window as one low-resolution video, first (it sits in the memory slot)
        pts = [t for t, _ in past]; pfr = [im.convert("RGB") for _, im in past]
        while len(pfr) % temporal:
            pfr.append(pfr[-1]); pts.append(pts[-1])
        vids.insert(0, torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in pfr]))
    gi = 1 if past else 0
    if zoom:  # MZ: the zoom clip as a second video
        zts = [t for t, _ in zoom]; zfr = [im.convert("RGB") for _, im in zoom]
        while len(zfr) % temporal:
            zfr.append(zfr[-1]); zts.append(zts[-1])
        vids.append(torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in zfr]))
    v = vp(videos=vids, do_sample_frames=False, do_resize=False, return_tensors="pt")
    grid = v["video_grid_thw"][gi]
    per_block = int(grid[1] * grid[2]) // (int(vp.merge_size) ** 2)
    blocks = "".join(f"<{sum(ts[i:i + temporal]) / temporal:.1f} seconds><|vision_start|>" + "<|video_pad|>" * per_block + "<|vision_end|>"
                     for i in range(0, len(ts), temporal))
    if zoom:
        zg = v["video_grid_thw"][gi + 1]
        zper = int(zg[1] * zg[2]) // (int(vp.merge_size) ** 2)
        blocks += ZOOM_TEXT + "".join(f"<{sum(zts[i:i + temporal]) / temporal:.1f} seconds><|vision_start|>" + "<|video_pad|>" * zper
                                      + "<|vision_end|>" for i in range(0, len(zts), temporal))
    mem = memory_text or "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>"
    if past:
        pg = v["video_grid_thw"][0]
        pper = int(pg[1] * pg[2]) // (int(vp.merge_size) ** 2)
        step = (past[-1][0] - past[0][0]) / max(1, len(past) - 1)
        mem = (f"<LONG_TERM_VIDEO_MEMORY>\nA low-resolution video of the stream before the recent frames, from {past[0][0]:.0f} s to "
               f"{past[-1][0]:.0f} s, one frame every {step:.1f} s, ordered from oldest to newest.\n"
               + "".join(f"<{sum(pts[i:i + temporal]) / temporal:.1f} seconds><|vision_start|>" + "<|video_pad|>" * pper + "<|vision_end|>"
                         for i in range(0, len(pts), temporal)) + "</LONG_TERM_VIDEO_MEMORY>")
    still = ""
    if last_still is not None:  # MV2: the newest frame again, as one still at source pixels (padded to the patch grid as in engine.answer)
        im = eng._pad_to_patch_grid(last_still[1].convert("RGB"))
        ip = eng.processor.image_processor(images=[im], do_resize=False, return_tensors="pt")
        n_img = int(ip["image_grid_thw"][0].prod()) // (int(eng.processor.image_processor.merge_size) ** 2)
        still = (f"\nThe newest frame at full resolution:\n<{last_still[0]:.3f}s><|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>")
    fps_txt = "per second" if len(ts) < 2 or abs(ts[1] - ts[0] - 1) < 1e-3 else f"every {ts[1] - ts[0]:.1f} s"
    user = (mem + "<RECENT_VISUAL_CONTEXT>\nThe following video clip shows the most recent frames, ordered from oldest to newest, one frame "
            + fps_txt + "; they have been resized while preserving aspect ratio.\n" + blocks + still + "</RECENT_VISUAL_CONTEXT>\n" + question_text)
    chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    tok = eng.processor.tokenizer
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    x = tok(prompt, return_tensors="pt")
    x["pixel_values_videos"] = v["pixel_values_videos"]; x["video_grid_thw"] = v["video_grid_thw"]
    if last_still is not None:
        x["pixel_values"] = ip["pixel_values"]; x["image_grid_thw"] = ip["image_grid_thw"]
    vid = tok.convert_tokens_to_ids("<|video_pad|>")
    img = tok.convert_tokens_to_ids("<|image_pad|>")
    eng.last_input_metrics = {"input_tokens": int(x["input_ids"].shape[-1]), "visual_tokens": int((x["input_ids"] == vid).sum() + (x["input_ids"] == img).sum())}
    x = {k: t.to(eng.device) for k, t in x.items()}
    for k in ("pixel_values_videos", "pixel_values"):
        if k in x: x[k] = x[k].to(eng.model.dtype)
    with torch.inference_mode():
        y = eng.model.generate(**x, max_new_tokens=max_tokens, do_sample=False, temperature=1.0, top_p=1.0, top_k=0, use_cache=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return eng._strip_end_tokens(eng.processor.batch_decode(y[:, x["input_ids"].shape[-1]:], skip_special_tokens=False)[0])


SF_PROMPTS = {  # SF2: block descriptions (fast, slow); timestamps are always kept
    "base": ("Low-resolution frames of the most recent seconds, every 0.5 s, oldest to newest, for following motion and changes.",
             "Detailed recent frames, oldest to newest; the newest keeps its source pixels, the others are resized preserving aspect ratio."),
    "none": ("", ""),
    "rich": ("Low-resolution frames covering the last {span:.0f} seconds, one every {step:.2f} s, oldest to newest. Use them to follow movement, "
             "direction, speed, distance changes, counts and the timing of actions.",
             "Detailed frames of the same period, oldest to newest; the newest frame is the current moment at full resolution. Use them for "
             "small details, text, attributes and object identity.")}


def answer_slowfast(eng, system, memory_text, frames, question_text, max_tokens, slow_every=4, fast_side=256, slow_side=768, prompt="base",
                    fast_last=None):
    """SF (results/HYBRID_RULES.md): frames = the recent window at 2 fps (oldest first). Fast path: every frame at fast_side px in a
    <FAST_MOTION_CONTEXT> block; slow path: every slow_every-th frame counted back from the newest at slow_side px, the newest at source
    pixels, in <RECENT_VISUAL_CONTEXT>. All as stills (engine.generate_chat, no resizing by the processor)."""
    fast = resize_frames(frames[-fast_last:] if fast_last else frames, fast_side)
    idx = sorted(set(range(len(frames) - 1, -1, -slow_every)))
    span = fast[-1][0] - fast[0][0] + (fast[1][0] - fast[0][0] if len(fast) > 1 else 0)
    step = (fast[1][0] - fast[0][0]) if len(fast) > 1 else 0
    pf, ps = (x.format(span=span, step=step) for x in SF_PROMPTS[prompt])
    slow = resize_frames([frames[i] for i in idx[:-1]], slow_side) + [frames[idx[-1]]]
    content = [{"type": "text", "text": memory_text or "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>"},
               {"type": "text", "text": "<FAST_MOTION_CONTEXT>\n" + pf}]
    images = []
    for t, im in fast:
        content += [{"type": "text", "text": f"<{t:.1f}s>"}, {"type": "image", "image": im}]; images.append(im)
    content.append({"type": "text", "text": "</FAST_MOTION_CONTEXT>\n<RECENT_VISUAL_CONTEXT>\n" + ps})
    for t, im in slow:
        content += [{"type": "text", "text": f"<{t:.3f}s>"}, {"type": "image", "image": im}]; images.append(im)
    content.append({"type": "text", "text": "</RECENT_VISUAL_CONTEXT>\n" + question_text})
    chat = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
    return eng.generate_chat(chat=chat, images=images, max_tokens=max_tokens, temperature=0.0, do_resize=False)


def answer_with_earlier(eng, system, memory_text, earlier, recent, question_text, max_tokens, recent_resize_policy):
    """OneStreamerEngine.answer with the earlier (retrieved) frames in their own block before the recent frames, and the system prompt
    naming that block; everything else as in answer()."""
    anchor = "Use them for the latest visual state.\n"
    assert system.count(anchor) == 1
    system = system.replace(anchor, anchor + EARLIER_ITEM)
    content = [{"type": "text", "text": memory_text or "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>"}]
    content.append({"type": "text", "text": "<EARLIER_KEYFRAMES>\nKeyframes from earlier in the video, before the recent frames, ordered "
                                            "from oldest to newest and shown at a small size. They show past moments, not the current state."})
    for t, im in earlier:
        content += [{"type": "text", "text": f"<{t:.3f}s>"}, {"type": "image", "image": im}]
    content.append({"type": "text", "text": "</EARLIER_KEYFRAMES>"})
    content.append({"type": "text", "text": "<RECENT_VISUAL_CONTEXT>\nThe following frames are ordered from oldest to newest. " + (
        "They retain their decoded source pixels; only minimal border padding may be added to align them to the model patch grid."
        if recent_resize_policy == "source_pixels" else "They have been resized while preserving aspect ratio.")})
    for t, im in recent:
        content += [{"type": "text", "text": f"<{t:.3f}s>"}, {"type": "image", "image": im}]
    content.append({"type": "text", "text": "</RECENT_VISUAL_CONTEXT>\n" + question_text})
    chat = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
    return eng.generate_chat(chat=chat, images=[im for _, im in earlier + recent], max_tokens=max_tokens, temperature=0.0,
                             do_resize=False)


NEW_DATA = Path("/data/bulk0/users/ahmed_heakl/onestreamer_data")  # benchmarks added for V5 / V6 (HYBRID_RULES.md)
VLM_JSON = C.ROOT / "OneStreamer/Eval/VLMEvalKit/json_data"


NB_MEM = Path(__file__).resolve().parent / "out/newbench_mem"


def nb_boundary(r):
    """M1: the question's last full second on the caption grid (start + floor(end - start))."""
    import math
    s = float(r["start"])
    return s + math.floor(float(r["end"]) - s + 1e-6)


def nb_key(r):
    """M1: one shared caption pass per (video, clip start)."""
    import hashlib
    return hashlib.md5(f'{r["video"]}|{float(r["start"]):.3f}'.encode()).hexdigest()


def nb_memory_fn(bench):
    """M1: memory events finished at the question's boundary; none in its first second."""
    from onestreamer.caption_optimized import boundary_key
    cache = {}

    def mem(r):
        b = nb_boundary(r)
        if b <= float(r["start"]):
            return []
        p = NB_MEM / bench / f"{nb_key(r)}.json"
        if p not in cache:
            cache.clear()
            cache[p] = json.load(open(p))
        bo = cache[p].get("m1", {}).get("boundary_of")
        if bo is not None:  # M1 amendment: boundary capped at the video's end
            b = bo.get(r["sample_id"])
            if b is None:
                return []
        return cache[p]["boundary_results"][boundary_key(b)]["events"]
    return mem


def odv_rows():
    """ODV-Bench (V5): the paper's packaged annotation; options = the candidates in order (A-D), answer = the matching letter, as
    vlmeval/dataset/odvbench.py. No caption memory exists for it."""
    rows = []
    for i, r in enumerate(json.load(open(VLM_JSON / "odvbench/ODVbench_bbox1000.json"))):
        cand = [str(x) for x in r["candidates"]]
        rows.append(dict(sample_id=f"odv-{i:05d}", index=i, task=str(r.get("task", "")), subtask=str(r["subtask"]), video=r["video"],
                         start=float(r.get("start") or 0.0), end=float(r["end"]), question=str(r["question"]),
                         options=dict(zip("ABCD", cand)), answer="ABCD"[cand.index(str(r["answer"]))], answer_text=str(r["answer"]),
                         eval_mode="mcq_letter"))
    return rows


def ovb_rows():
    """OVBench (V6): the paper's packaged annotation; options "A. text" -> {"A": "text"}; subtask = sub_answer_type (16). Videos from
    OVBench/_manifest.json (onestreamer_data/prep_ovb.py); AVA_frame_4fps_new rows read the AVA_RAW movie with the paper's window,
    start = max(question time - 120 s, 900 s) (HYBRID_RULES.md, V6 amendment). No caption memory exists for it."""
    man = json.load(open(NEW_DATA / "OVBench/_manifest.json"))
    rows = []
    for i, r in enumerate(json.load(open(VLM_JSON / "ovbench/ovbench_split_with_fps_ava_raw_frame_processed_128_bbox1000.json"))):
        opts = {o.split(".", 1)[0].strip(): o.split(".", 1)[1].strip() for o in r["options"]}
        end = float(r["end"])
        start = max(end - 120.0, 900.0) if r["video_id"].startswith("AVA_frame_4fps_new/") else float(r["start"])
        rows.append(dict(sample_id=f"ovb-{i:05d}", index=i, task=r["answer_type"], subtask=r["sub_answer_type"], video=man[r["video_id"]],
                         start=start, end=end, question=str(r["question"]), options=opts, answer=str(r["answer"]).strip(),
                         answer_text=opts.get(str(r["answer"]).strip(), ""), eval_mode="mcq_letter", ovb_id=r["id"]))
    return rows


def load_bench(bench):
    """Returns (rows, video_root, memory_fn) where memory_fn(row) -> list of memory events available at the row's end."""
    from onestreamer.caption_optimized import boundary_key
    if bench == "ovo":
        rows = json.load(open(C.FULL8 / "manifest.json"))["samples"]
        cap = C.FULL8 / "memory/captions/OneStreamer-4B"
        return rows, C.VIDEO_ROOT, lambda r: json.load(open(cap / f"{r['sample_id']}.json"))["events"]
    if bench == "ovo_fwd":
        rows = json.load(open(FWD / "forward_rows.json"))
        return rows, C.VIDEO_ROOT, lambda r: json.load(open(FWD / "memory" / f"item-{r['item']}.json"))["boundary_results"][
            boundary_key(r["end"])]["events"]
    if bench == "ovb":
        return ovb_rows(), NEW_DATA / "OVBench", nb_memory_fn("ovb")
    if bench == "odv":
        return odv_rows(), NEW_DATA / "ODV-Bench", nb_memory_fn("odv")
    if bench == "streambench":
        import streambench as SBN
        return json.load(open(SBN.OUT / "rows.json")), SBN.ROOT, SBN.memory_fn()
    rows = json.load(open(SB_RUN / "manifest.json"))["samples"]
    by_video = {}
    for p in glob.glob(str(SB_RUN / "memory/shared_timelines/OneStreamer-4B/*.json")):
        d = json.load(open(p))
        by_video[d["video"].split("StreamingBench/")[-1]] = p
    cache = {}

    def mem(r):
        p = by_video[r["video"]]
        if p not in cache:
            cache.clear()
            cache[p] = json.load(open(p))
        return cache[p]["boundary_results"][boundary_key(float(r["end"]))]["events"]
    return rows, SB_ROOT, mem


def memories(ev, qtime, question, tok, render, render_plain, want=None, cap_sim=None, cap_sim_opt=None, sv_caps=None):
    """Answer-time memory text per method, plus question-agnostic stored tokens and answer-time added tokens."""
    n = lambda t: len(tok.encode(t, add_special_tokens=False)) if t else 0
    hy_qa = K.merge_repeats(K.consolidate(ev, keep_after=qtime - 60))
    hy = K.hybrid(ev, qtime, question)
    out = {}
    for v, events, stored in (("full", ev, ev), ("consolidated", K.consolidate(ev), K.consolidate(ev)),
                              ("keep60", K.consolidate(ev, keep_after=qtime - 60), K.consolidate(ev, keep_after=qtime - 60)),
                              ("hybrid_qa", sorted(hy_qa, key=lambda e: (float(e["end"]), e["kind"])), hy_qa), ("hybrid", hy, hy_qa)):
        text = render(events)
        st = n(render_plain(stored))
        out[v] = (text, n(text), st, max(0, n(text) - st))
    out["none"] = ("", 0, 0, 0)
    hs = sorted(hy_qa, key=lambda e: (float(e["end"]), e["kind"]))
    for k in (1, 2, 4, 8):  # VT: the k most recent compacted entries
        if want is None or f"last{k}" in want:
            x = hs[-k:]
            text = render(x)
            out[f"last{k}"] = (text, n(text), n(render_plain(x)), 0)
    caps = sorted((e for e in ev if e["kind"] == "response"), key=lambda e: float(e["end"]))
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):  # CD: the k most recent of OneStreamer's own captions, no summaries
        if want is not None and f"raw{k}" in want:
            x = caps[-k:]
            text = render(x)
            out[f"raw{k}"] = (text, n(text), n(render_plain(x)), 0)
    for pre, cs in (("raw", caps), ("sv", sv_caps or [])):  # CQ: the 64 most recent captions of a captioner, or all of them; no summaries
        for k in ("64", "all"):
            if want is not None and pre + k in want and pre + k not in out:
                x = cs if k == "all" else cs[-64:]
                text = render(x)
                out[pre + k] = (text, n(text), n(render_plain(x)), 0)
    for name, sim in (("ret8", cap_sim), ("ret8opt", cap_sim_opt)):  # VT / RO: the 8 compacted entries most similar to the question
        if want is None or name not in want:                          # (RO: question + answer options), CLIP text-text, in time order
            continue
        x = hs
        if len(hs) > 8 and sim is not None:
            sc = sim([C.body_of(e["text"]) for e in hs])
            x = [hs[i] for i in sorted(np.argsort(-sc)[:8])]
        text = render(x)
        out[name] = (text, n(text), n(render_plain(x)), 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["ovo", "ovo_fwd", "sb", "streambench", "odv", "ovb"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="full,consolidated,keep60,hybrid_qa,hybrid,none")
    ap.add_argument("--seeds", default="", help="comma list; empty = greedy")
    ap.add_argument("--max-side", default="", help="comma list of long-side caps (px) for the answer frames; empty = source pixels")
    ap.add_argument("--retrieve", type=int, default=0, help="answer frames retrieved from earlier in the video (of 16); 0 = all recent")
    ap.add_argument("--recent", type=int, default=0, help="recent frames kept with --retrieve; 0 = 16 - retrieve")
    ap.add_argument("--ret-side", type=int, default=0, help="long side (px) of the retrieved frames; 0 = as --max-side")
    ap.add_argument("--select", default="sharp", help="comma list of candidate scorings with --retrieve: clip, sharp, vlm, spread")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all", help="only the questions of videos in this half")
    ap.add_argument("--source-latest", type=int, default=0, help="most recent frames kept at source pixels when resizing")
    ap.add_argument("--earlier-block", action="store_true", help="retrieved frames in their own block, marked as past moments")
    ap.add_argument("--save-conf", action="store_true", help="record answer confidence (conf, conf_first, tok_logp); greedy only")
    ap.add_argument("--model", default="OneStreamer-4B", help="answering model under path_to/models")
    ap.add_argument("--ids", default="", help="CQ: only the questions listed in this JSON file (a list, or a dict with a questions key)")
    ap.add_argument("--sv-dir", default="", help="CQ: folder of StreamingVLM caption passes (sv_caption.py) for the sv64 / svall variants")
    ap.add_argument("--sf-prompt", default="base", choices=["base", "none", "rich"], help="SF2: block descriptions")
    ap.add_argument("--sf-fast-side", type=int, default=256, help="SF2: fast-frame long side (px)")
    ap.add_argument("--sf-slow-side", type=int, default=768, help="SF2: slow-frame long side (px)")
    ap.add_argument("--sf-slow-every", type=int, default=4, help="SF2: slow frame every k-th recent frame, counted back from the newest")
    ap.add_argument("--sf-fast-last", type=int, default=0, help="SF2: fast path uses only the last N recent frames (0 = all)")
    ap.add_argument("--slowfast", action="store_true", help="SF: slow detailed stills + fast low-res stills (use --recent-n 32 --recent-fps 2)")
    ap.add_argument("--recent-video-last-source", action="store_true", help="MV2: with --recent-video, add the newest frame as a source-pixel still")
    ap.add_argument("--recent-n", type=int, default=16, help="MV2: recent frames without retrieval")
    ap.add_argument("--recent-fps", type=float, default=1.0, help="MV2: sampling rate of the recent frames without retrieval")
    ap.add_argument("--past-fps", type=float, default=0.0, help="CV: add the stream before the recent window as a low-res video at this rate")
    ap.add_argument("--past-side", type=int, default=320, help="CV: long side (px) of the past video")
    ap.add_argument("--past-max", type=int, default=256, help="CV: at most this many past frames (uniformly subsampled)")
    ap.add_argument("--past-evict", choices=["time", "change"], default="time",
                    help="EXP6 store eviction: time = BoundedStore (closest neighbours in time); ACT change = the frame that changed least")
    ap.add_argument("--past-window", type=float, default=0.0, help="STRESS: the past covers only this many seconds before the first recent frame (0 = all)")
    ap.add_argument("--past-store", type=int, default=0, help="EXP6: keep the past frames with EfficientStream's BoundedStore of this budget")
    ap.add_argument("--past-keep-memory", action="store_true", help="KV-BUILD: keep the caption memory before the past video")
    ap.add_argument("--stills-past", action="store_true", help="EXP6: recent frames as stills at source pixels (answer_stills), past as video")
    ap.add_argument("--mem-dir", default="", help="EXP3 for his models: per-question memory events (hismem/events/<model>) instead of OneStreamer's")
    ap.add_argument("--timing", action="store_true", help="record the time to the first generated token and to the end of generation per answer")
    ap.add_argument("--st-dir", default="", help="ST: folder of FOLIO-style memories (folio_mem.py) for the folio variant")
    ap.add_argument("--motion-zoom", default="", choices=["", "motion", "center"], help="MZ: add a 2x zoom clip of the most-moving window (center = CZ control)")
    ap.add_argument("--recent-video", action="store_true", help="MV: the recent frames as one video (needs --max-side, no --source-latest)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    selects = args.select.split(",") if args.retrieve else [None]
    recent = args.recent or 16 - args.retrieve
    if args.retrieve:
        import frame_bank as FB
        clip = FB.load_clip()
        FB.embed_text(*clip, "warm-up")  # first CUDA call is slow; keep it out of retrieve_ms
        if "vlm" in selects:
            from PIL import Image
            reranker = FB.Reranker()
            reranker.score("warm-up", [Image.new("RGB", (384, 224))] * 2)
    sides = [int(x) for x in args.max_side.split(",")] if args.max_side else [0]
    use_text = bool({"ret8", "ret8opt"} & set(args.variants.split(","))) or "capidx" in (selects or [])
    if use_text and not args.retrieve:
        import frame_bank as FB
        clip = FB.load_clip()
    _tcache = {}
    _svc = {}

    _stc = {}

    def st_mem_of(r):  # ST: the FOLIO-style memory of the row's video
        import hashlib
        key = hashlib.md5(str(vroot / r["video"]).encode()).hexdigest()[:16]
        if key not in _stc:
            _stc.clear()
            _stc[key] = json.load(open(Path(args.st_dir) / f"{key}.json"))
        return _stc[key]

    def sv_caps_of(r):  # CQ: StreamingVLM captions that end by the question time, in time order
        import hashlib
        key = hashlib.md5(str(vroot / r["video"]).encode()).hexdigest()[:16]
        if key not in _svc:
            _svc.clear()
            _svc[key] = json.load(open(Path(args.sv_dir) / f"{key}.json"))["events"]
        return sorted((e for e in _svc[key] if float(e["end"]) <= float(r["end"]) + 1e-6), key=lambda e: float(e["end"]))

    def text_embs(texts):  # VT: CLIP text embeddings, cached
        import torch
        new = [t for t in dict.fromkeys(texts) if t not in _tcache]
        for i in range(0, len(new), 128):
            x = clip[1](text=new[i:i + 128], return_tensors="pt", padding=True, truncation=True, max_length=77).to(clip[0].device)
            with torch.inference_mode():
                e = clip[0].get_text_features(**x).float()
            _tcache.update(zip(new[i:i + 128], (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()))
        return np.stack([_tcache[t] for t in texts])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import transformers
    from onestreamer.engine import OneStreamerEngine
    from onestreamer.memory import MemoryBank, MemoryEvent
    from onestreamer.video import read_recent_frames
    from onestreamer.prompts import MEMORY_QA_SYSTEM, MEMORY_DIRECT_QA_SYSTEM
    from onestreamer.ovobench import build_question_text, score_prediction
    from onestreamer.schema import Sample
    rows, vroot, mem = load_bench(args.bench)
    if args.ids:  # CQ: sanity subset
        keep = json.load(open(args.ids))
        keep = set(keep["questions"] if isinstance(keep, dict) else keep)
        rows = [r for r in rows if r["sample_id"] in keep]
    if args.split != "all":
        import hashlib
        rows = [r for r in rows if (int(hashlib.md5(r["video"].encode()).hexdigest(), 16) % 2 == 0) == (args.split == "dev")]
    Engine = OneStreamerEngine
    if args.model.startswith("InternVL"):
        from internvl_engine import InternVLEngine as Engine
    eng = Engine(model_path=str(C.MODELS / args.model), dtype="bfloat16", attn_implementation=os.environ.get("VR_ATTN", "flash_attention_2"),
                 min_pixels=3136, max_pixels=100352)
    tok = eng.processor.tokenizer
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [None]
    if args.seeds:
        eng.model.generation_config.top_p, eng.model.generation_config.top_k = 0.8, 20
        orig = eng.generate_chat
        eng.generate_chat = lambda **kw: orig(**{**kw, "temperature": 0.7})
    last = {}  # --save-conf: logits of the first generated token and the log-probability of every generated token
    if args.save_conf:
        import torch
        assert not args.seeds
        generate = eng.model.generate

        def generate_and_keep(**kw):
            out = generate(**kw, output_logits=True, return_dict_in_generate=True)
            lg = torch.stack(out.logits)[:, 0].float()
            ids = out.sequences[0, kw["input_ids"].shape[-1]:][:len(lg)]
            last.update(first=lg[0].cpu(), tok_logp=torch.log_softmax(lg, -1).gather(1, ids[:, None])[:, 0].tolist())
            return out.sequences
        eng.model.generate = generate_and_keep
    tm = {}  # --timing: when the first token's logits exist and when generation ends (CUDA synchronised)
    if args.timing:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        class _First(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if "first" not in tm:
                    torch.cuda.synchronize(); tm["first"] = time.perf_counter()
                return scores
        generate_t = eng.model.generate

        def generate_timed(**kw):
            lp = kw.pop("logits_processor", None) or LogitsProcessorList()
            lp.append(_First())
            out = generate_t(**kw, logits_processor=lp)
            torch.cuda.synchronize(); tm["end"] = time.perf_counter()
            return out
        eng.model.generate = generate_timed
    path = out / f"results_{args.shard:02d}.jsonl"
    want = len(args.variants.split(",")) * len(seeds) * len(sides) * len(selects)
    kept = collections.defaultdict(list)
    if path.exists():  # resume after an interruption: keep questions with every (method, seed) row, redo the rest
        for line in open(path):
            try:
                kept[json.loads(line)["sample_id"]].append(line.rstrip("\n") + "\n")
            except (json.JSONDecodeError, KeyError):
                pass
    done = {q for q, lines in kept.items() if len(lines) == want}
    with open(path, "w") as f:
        f.writelines(line for q in sorted(done) for line in kept[q])
    print(f"shard {args.shard}: resuming with {len(done)} questions done", flush=True)
    with open(path, "a") as f:
        for i, r in enumerate(rows[args.shard::args.nshards]):
            if r["sample_id"] in done:
                continue
            sample = Sample.from_dict({k: v for k, v in r.items() if k not in ("item", "domain", "category", "time")})
            if args.mem_dir:  # EXP3 for his models: his PHCM memory at this question
                _p = Path(args.mem_dir) / f"{r['sample_id']}.json"
                ev = [e for e in json.load(open(_p))["events"] if e["kind"] in ("response", "summary") and e["text"]] if _p.exists() else []
            else:
                ev = [e for e in mem(r) if e["kind"] in ("response", "summary") and e["text"]]
            cap_sim = None
            if use_text:
                qv_txt = FB.embed_text(*clip, sample.question)
                cap_sim = lambda texts, qv=qv_txt: text_embs(texts) @ qv if texts else np.zeros(0, np.float32)
            cap_sim_opt = None
            if use_text and "ret8opt" in args.variants.split(","):  # RO: query = question + answer options (as sharpopt)
                qv_opt = FB.embed_text(*clip, (sample.question + " " + " ".join(str(v) for v in (r.get("options") or {}).values())).strip())
                cap_sim_opt = lambda texts, qv=qv_opt: text_embs(texts) @ qv if texts else np.zeros(0, np.float32)
            ret = {None: ({}, [])}  # select -> (row fields, retrieved (t, image) at source pixels)
            if args.retrieve:
                frames = read_recent_frames(vroot / sample.video, sample.start, sample.end, recent_frames=recent, sample_fps=1.0)
                fb = FB.load_bank(FB.BANK_OF[args.bench], sample.video)
                t0 = time.perf_counter()
                q = FB.embed_text(*clip, sample.question)
                embed_ms = 1000 * (time.perf_counter() - t0)
                t0 = time.perf_counter()
                ts, sim, sh = FB.candidates(fb, q, float(sample.start), frames[0][0] - 1.0)
                cand_ms = 1000 * (time.perf_counter() - t0)
                imgs = {}
                if "vlm" in selects and len(ts):
                    imgs = dict(FB.read_at(vroot / sample.video, sorted(float(t) for t in ts)))
                ret = {}
                ok = [i for i in range(len(ts)) if float(ts[i]) in imgs]  # the frame cache holds the candidates at the rerank size already
                small = [im for _, im in resize_frames([(0, imgs[float(ts[i])]) for i in ok], args.ret_side or sides[0] or 384)]
                for sel in selects:
                    t0, extra = time.perf_counter(), {}
                    ts_s, sim_s = ts, sim
                    if sel == "vlm":
                        t1 = time.perf_counter()
                        rr = reranker.score(sample.question, small) if ok else []
                        extra["rerank_ms"] = 1000 * (time.perf_counter() - t1)
                        score = np.full(len(ts), -np.inf, np.float32)
                        score[ok] = rr
                        idx = [i for i in FB.pick(ts, score, args.retrieve) if np.isfinite(score[i])]
                    elif sel == "sharpopt":  # VT: thumbnail query = question + answer options
                        q2 = FB.embed_text(*clip, (sample.question + " " + " ".join(str(v) for v in (r.get("options") or {}).values())).strip())
                        ts_s, sim_s, sh_s = FB.candidates(fb, q2, float(sample.start), frames[0][0] - 1.0)
                        idx = FB.pick(ts_s, FB.rank_scores("sharp", sim_s, sh_s), args.retrieve)
                    elif sel in ("pad", "crop3", "tile7", "sg2", "sg2tile"):  # RT: stronger retrieval banks (frame_bank2)
                        import frame_bank2 as FB2
                        ts_s, sim_s, sh_s = FB2.candidates(sel, FB.BANK_OF[args.bench], sample.video, sample.question, float(sample.start),
                                                           frames[0][0] - 1.0)
                        idx = FB.pick(ts_s, FB.rank_scores("sharp", sim_s, sh_s), args.retrieve)
                    elif sel == "capidx":  # VT: frames indexed by the captions written around them
                        ts_s, sim_s, sc_s = capidx_select(fb, q, cap_sim, ev, float(sample.start), frames[0][0] - 1.0)
                        idx = FB.pick(ts_s, sc_s, args.retrieve)
                    elif sel != "spread":
                        idx = FB.pick(ts, FB.rank_scores(sel, sim, sh), args.retrieve)
                    if sel == "spread":
                        times = FB.spread(fb, float(sample.start), frames[0][0] - 1.0, args.retrieve)
                        sel_ms = 1000 * (time.perf_counter() - t0)
                        on = np.isin(fb["ts"], times)
                        sims = [float(x) for x in fb["emb"][on].astype(np.float32) @ q]
                    else:
                        sel_ms = cand_ms + 1000 * (time.perf_counter() - t0)
                        times, sims = [float(ts_s[i]) for i in idx], [float(sim_s[i]) for i in idx]
                    if len(times) < args.retrieve and recent + args.retrieve <= 16:  # too little video: 16 recent frames, as without retrieval
                        times, sims = [], []
                    got = [(t, imgs[t]) for t in times if t in imgs]
                    got += FB.read_at(vroot / sample.video, [t for t in times if t not in imgs])
                    ret[sel] = (dict(select=sel, retrieved_ts=times, retrieved_sim=sims, embed_ms=embed_ms, select_ms=sel_ms,
                                     retrieve_ms=embed_ms + sel_ms, **extra), sorted(got, key=lambda x: x[0]))
                if any(not v[1] for v in ret.values()) and recent < 16:
                    frames16 = read_recent_frames(vroot / sample.video, sample.start, sample.end, recent_frames=16, sample_fps=1.0)
            else:
                frames = read_recent_frames(vroot / sample.video, sample.start, sample.end, recent_frames=args.recent_n, sample_fps=args.recent_fps)
            bank = lambda x: MemoryBank(MemoryEvent.from_dict(e) for e in x)
            render = lambda x: bank(x).render_with_budget(mode="hierarchical", question=sample.question, tokenizer=tok, max_tokens=12000)[0]
            render_plain = lambda x: bank(x).render("hierarchical") if x else ""
            sv_caps = sv_caps_of(r) if any(v.startswith("sv") for v in args.variants.split(",")) else None
            texts = memories(ev, float(sample.end), sample.question, tok, render, render_plain, want=set(args.variants.split(",")), cap_sim=cap_sim, cap_sim_opt=cap_sim_opt, sv_caps=sv_caps)
            if "folio" in args.variants.split(","):  # ST
                ft, st_info = FO.render(st_mem_of(r), float(sample.start), float(sample.end), sample.question, r.get("options"))
                nft = len(tok.encode(ft, add_special_tokens=False))
                texts["folio"] = (ft, nft, nft, 0)
            system = MEMORY_QA_SYSTEM if sample.eval_mode == "mcq_letter" else MEMORY_DIRECT_QA_SYSTEM
            size = lambda fr, side: resize_frames(fr, side) if side else fr
            L = args.source_latest
            size_recent = lambda fr, side: size(fr[:-L], side) + fr[-L:] if L and side else size(fr, side)
            sized, n_early = {}, {}
            for sel in selects:
                got = ret[sel][1]
                for side in sides:
                    if args.retrieve and not got and recent < 16:
                        sized[sel, side], n_early[sel, side] = size_recent(frames16, side), 0
                    else:
                        sized[sel, side], n_early[sel, side] = size(got, args.ret_side or side) + size_recent(frames, side), len(got)
            for seed, sel, side, v in ((a, s_, b, c) for a in seeds for s_ in selects for b in sides for c in args.variants.split(",")):
                fr = sized[sel, side]
                text, n_ans, n_stored, n_added = texts[v]
                if seed is not None:
                    transformers.set_seed(seed * 1_000_003 + int(r["index"]))
                kw = dict(system=system, memory_text=text, question_text=build_question_text(sample),
                          max_tokens=128 if args.bench == "streambench" else 16, recent_resize_policy="resized" if side else "source_pixels")
                k = n_early[sel, side] if args.earlier_block else 0
                past = None
                if args.past_fps:  # CV / EXP6: question start up to the first recent frame
                    t_w, t_s = float(frames[0][0]), float(sample.start)
                    if args.past_window:
                        t_s = max(t_s, t_w - args.past_window)
                    times = [float(x) for x in np.arange(t_s, t_w - 1e-6, 1.0 / args.past_fps)]
                    if args.past_store and args.past_evict == "change":
                        pass  # ACT: the store decides from the pixels below
                    elif args.past_store:
                        times = bounded_store_times(times, args.past_store)
                    elif len(times) > args.past_max:
                        times = [times[i] for i in np.linspace(0, len(times) - 1, args.past_max).round().astype(int)]
                    if len(times) >= 2:
                        from folio_mem import SeqReader, resize_long
                        rd = SeqReader(vroot / sample.video, t_s)
                        if args.past_store and args.past_evict == "change":
                            past = change_store_frames(rd, times, args.past_store, args.past_side)
                        else:
                            past = [(t, resize_long(im, args.past_side)) for t, im in rd.at(times)]
                        rd.cap.release()
                if args.timing:  # from the question (frames, past frames and memory text ready) to the first token / the end of the answer
                    tm.clear(); torch.cuda.synchronize(); t_q = time.perf_counter()
                if args.stills_past:  # EXP6 (past may be None: the plain 16-recent prompt through the same path)
                    pred = answer_stills(eng, kw["system"], kw["memory_text"], fr, kw["question_text"], kw["max_tokens"], past=past,
                                         keep_memory=args.past_keep_memory)
                elif args.slowfast:  # SF
                    pred = answer_slowfast(eng, kw["system"], kw["memory_text"], frames, kw["question_text"], kw["max_tokens"],
                                           slow_every=args.sf_slow_every, fast_side=args.sf_fast_side, slow_side=args.sf_slow_side,
                                           prompt=args.sf_prompt, fast_last=args.sf_fast_last or None)
                elif args.recent_video:  # MV
                    zoom = None
                    if args.motion_zoom:  # MZ
                        box, mz = motion_window(frames, center=args.motion_zoom == "center")
                        zoom = zoom_clip(frames[-8:], box) if box else None
                    pred = answer_video(eng, kw["system"], kw["memory_text"], fr, kw["question_text"], kw["max_tokens"],
                                        last_still=frames[-1] if args.recent_video_last_source else None, zoom=zoom, past=past)
                else:
                    pred = answer_with_earlier(eng, earlier=fr[:k], recent=fr[k:], **kw) if k else eng.answer(recent_frames=fr, **kw)
                if args.timing:
                    t_first_ms, t_gen_ms = 1000 * (tm["first"] - t_q), 1000 * (tm["end"] - t_q)
                parsed, hit = (pred.strip(), None) if args.bench == "streambench" else score_prediction(sample, pred)
                rec = dict(sample_id=r["sample_id"], subtask=r["subtask"], variant=v, prediction=pred, parsed=parsed, hit=hit,
                           memory_tokens=n_ans, stored_memory_tokens=n_stored, answer_time_added_tokens=n_added,
                           input_tokens=eng.last_input_metrics["input_tokens"], visual_tokens=eng.last_input_metrics["visual_tokens"])
                if v.startswith("raw"):  # CD: captions available at the question (raw{k} supplies min(k, this))
                    rec["n_cap_avail"] = sum(e["kind"] == "response" for e in ev)
                if v.startswith("sv"):  # CQ: StreamingVLM captions available at the question
                    rec["n_cap_avail"] = len(sv_caps)
                if seed is not None:
                    rec["seed"] = seed
                if side:
                    rec["max_side"] = side
                if L:
                    rec["source_latest"] = L
                if args.earlier_block:
                    rec["earlier_block"] = 1
                if args.recent_video:
                    rec["recent_video"] = 1 + int(args.recent_video_last_source)
                if args.motion_zoom:
                    rec["mz"] = dict(mode=args.motion_zoom, **mz)
                if v == "folio":
                    rec["st"] = st_info
                if args.timing:
                    rec["t_first_ms"], rec["t_gen_ms"] = round(t_first_ms, 1), round(t_gen_ms, 1)
                if args.past_fps or args.stills_past:
                    rec["past"] = dict(n=len(past or []), t0=past[0][0] if past else None, t1=past[-1][0] if past else None,
                                       evict=args.past_evict)
                if args.slowfast:
                    rec["slowfast"] = dict(prompt=args.sf_prompt, fast_side=args.sf_fast_side, slow_side=args.sf_slow_side, slow_every=args.sf_slow_every,
                                           fast_last=args.sf_fast_last, n=args.recent_n, fps=args.recent_fps)
                if args.recent_n != 16 or args.recent_fps != 1.0:
                    rec["recent_n"], rec["recent_fps"] = args.recent_n, args.recent_fps
                if args.model != "OneStreamer-4B":
                    rec["model"] = args.model
                if args.save_conf:
                    rec["conf_first"] = round(float(torch.softmax(last["first"], 0).max()), 5)
                    rec["conf"] = rec["conf_first"]
                    _p = torch.softmax(last["first"].float(), 0)
                    rec["conf_second"] = round(float(torch.topk(_p, 2).values[1]), 5)  # VT
                    rec["entropy_first"] = round(float(-(_p * torch.log(_p.clamp_min(1e-12))).sum()), 5)
                    if sample.eval_mode == "mcq_letter":
                        p = torch.softmax(last["first"][[tok.convert_tokens_to_ids(x) for x in sample.options]], 0)
                        rec["p_opts"] = {x: round(float(v), 5) for x, v in zip(sample.options, p)}
                        rec["conf"] = round(float(p.max()), 5)
                    rec["tok_logp"] = [round(x, 4) for x in last["tok_logp"]]
                if args.retrieve:
                    rec.update(ret[sel][0], recent=recent, n_retrieved=len(ret[sel][0]["retrieved_ts"]))
                    if args.ret_side:
                        rec["ret_side"] = args.ret_side
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"shard {args.shard}: {i + 1}", flush=True)
    print("ANSWER_DONE", args.shard, flush=True)


if __name__ == "__main__":
    main()
