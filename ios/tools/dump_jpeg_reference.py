#!/usr/bin/env python3
"""Pillow JPEG round trips (the reference's jpeg_roundtrip) of video frames and noise at awkward sizes, for
es-verify jpeg."""
import argparse, io
import numpy as np
from PIL import Image
from safetensors.numpy import save_file
from efficientstream.video import iter_frames

p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("--out", required=True); p.add_argument("--quality", type=int, default=88)
a = p.parse_args()
sizes = [(672, 384), (384, 672), (504, 896), (896, 504), (448, 448), (448, 256), (853, 480), (481, 271),
         (17, 9), (16, 16), (1, 1), (2, 3), (33, 65)]
frames = [fr for _, fr in iter_frames(a.video, fps=0.05, end=300)]
rng = np.random.default_rng(0)
d = {}
for i, (w, h) in enumerate(sizes):
    src = Image.fromarray(frames[i % len(frames)]).resize((w, h), Image.BICUBIC) if i % 3 else \
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO(); src.save(buf, format="JPEG", quality=a.quality); buf.seek(0)
    d[f"in{i}"] = np.asarray(src, np.uint8)
    d[f"out{i}"] = np.asarray(Image.open(buf).convert("RGB"), np.uint8)
save_file(d, a.out)
print(len(sizes), "images")
