#!/usr/bin/env python3
"""Run the PyTorch EfficientStream pipeline on a video and save what the iOS port must reproduce, per question:
token ids, the input images (the letterboxed thumbnails and the latest frame after its JPEG round trip), every
vision view in the processor's layout, the vision features of each view, and the answer.

    python ios/tools/dump_reference.py video.mp4 --model ahmedheakl/efficientstream --out ref --device cpu --at 60 200

Then compare the Swift port (EfficientStreamKit's es-verify):
    xcodebuild -scheme es-verify -destination 'platform=macOS' -configuration Release -derivedDataPath dd build
    dd/Build/Products/Release/es-verify ios/models ref

Use --device cpu for a reference: on Apple's MPS backend, sliced views copied to the GPU gave wrong vision
features in our runs (the package's own question path, which concatenates the views first, was not affected).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors.numpy import save_file

from efficientstream import EfficientStream
from efficientstream.frames import fit_long_side, format_time, jpeg_roundtrip, letterbox
from efficientstream.index import top_k_in_time_order
from efficientstream.prompt import question_block
from efficientstream.video import iter_frames

QUESTIONS = [
    ("What is the large animal doing right now?", None),
    ("What colour is the main character?", ["A. white or grey", "B. green", "C. red", "D. blue"]),
    ("Where does this scene take place?", ["A. in a meadow or forest", "B. in a city street", "C. underwater",
                                           "D. in space"]),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--model", default="ahmedheakl/efficientstream")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--at", type=float, nargs="+", default=[60.0], help="question times (s): asked at the first "
                   "decoded frame at or after each time")
    p.add_argument("--questions", help="JSON file [[question, [options] or null], ...] (default: built-in)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qs = json.load(open(a.questions)) if a.questions else QUESTIONS

    es = EfficientStream.from_pretrained(a.model, device=a.device)
    s, cfg, A = es.session(), es.config, es.answerer
    meta = {"cases": []}
    pending = sorted(a.at)
    for t, frame in iter_frames(a.video, fps=2.0, end=pending[-1] + 1.0):
        s.add_frame(t, frame)
        if not pending or t + 1e-6 < pending[0]:
            continue
        at = pending.pop(0)
        now, latest = s.latest
        for qi, (q, opts) in enumerate(qs):
            tag = f"t{int(at)}_q{qi}"
            cand = s.store.before(now)
            qe = es.index.embed_text([q])[0]
            picked = [cand[i] for i in top_k_in_time_order(np.stack([f.embedding for f in cand]), qe, cfg.retrieve_k)]
            parts, images = [], []
            for f in picked:
                parts += [{"type": "text", "text": f"frame time[{format_time(f.time, cfg.timestamps)}]: "},
                          {"type": "image"}, {"type": "text", "text": "\n"}]
                images.append((letterbox(f.image, cfg.thumb_side), cfg.thumb_side, 1))
            parts += [{"type": "text", "text": f"frame time[{format_time(now, cfg.timestamps)}] (now): "},
                      {"type": "image"}, {"type": "text", "text": "\n"}]
            hi = jpeg_roundtrip(fit_long_side(latest, cfg.latest_long_side), cfg.latest_jpeg_quality)
            images.append((hi, cfg.thumb_side, cfg.latest_max_slices))
            parts.append({"type": "text", "text": question_block(q, opts)})
            ids, views = A.encode(A.build_prompt(parts), images)
            feats = []
            with torch.no_grad():
                for px, hw in views:
                    out = A.model.model.get_image_features(
                        px.contiguous().unsqueeze(0).to(a.device), torch.tensor([hw], dtype=torch.int32, device=a.device),
                        downsample_mode="16x").pooler_output
                    feats.append(out[0].float().cpu().numpy())
            r = s.ask(q, opts)
            dump = {"ids": np.array(ids, np.int32), "question_emb": qe.astype(np.float32),
                    "vision_feats": np.concatenate(feats, 0).astype(np.float32)}
            for vi, (px, _) in enumerate(views):
                dump[f"view{vi}"] = px.contiguous().float().numpy()
            for i, (im, _, _) in enumerate(images):
                dump[f"image{i}"] = np.asarray(im, np.uint8)
            save_file(dump, os.path.join(a.out, tag + ".safetensors"))
            meta["cases"].append(dict(tag=tag, now=now, question=q, options=opts, view_hw=[list(hw) for _, hw in views],
                                      answer=r.text, probs=r.probs, frame_times=r.frame_times,
                                      prompt_tokens=r.prompt_tokens, seconds=r.seconds))
            json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
            print(tag, json.dumps(dict(answer=r.text, probs=r.probs, tokens=r.prompt_tokens,
                                       seconds=round(r.seconds, 2))), flush=True)
        if not pending:
            break


if __name__ == "__main__":
    main()
