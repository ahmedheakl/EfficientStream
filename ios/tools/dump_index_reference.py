#!/usr/bin/env python3
"""Reference data for the iOS frame index (MobileCLIP2-S0 + the store's image handling): for frames of a video, the
raw frame, the store-size resize, the stored frame (after the JPEG round trip), open_clip's preprocessing of it, and
its embedding; for a list of questions, the token ids and text embeddings.

    python ios/tools/dump_index_reference.py video.mp4 --encoder <weights>/encoder --out ref/index.safetensors
    es-verify index ios/models ref/index.safetensors
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file

from efficientstream.frames import grid_size, jpeg_roundtrip
from efficientstream.index import FrameIndex
from efficientstream.video import iter_frames

QUESTIONS = ["What is the large animal doing right now?", "What colour is the main character?",
             "Where does this scene take place?", "Is there a butterfly?", "How many squirrels are there?",
             "What did the rabbit just pick up?", "Describe the weather.", "Who throws the fruit at the rabbit?  ",
             "What's happening at 3:15?", "Is it night or day?", "Was a bird visible earlier in the stream?",
             "ARE THERE ANY TREES??"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--encoder", required=True, help="the weights' encoder/ folder")
    p.add_argument("--out", required=True, help="output .safetensors (a .json with the times and questions is "
                   "written next to it)")
    p.add_argument("--fps", type=float, default=0.1)
    p.add_argument("--end", type=float, default=400.0)
    a = p.parse_args()
    idx = FrameIndex(a.encoder, device="cpu", dtype=torch.float32, round_fp16=True)
    d, times = {}, []
    for t, fr in iter_frames(a.video, fps=a.fps, end=a.end):
        im = Image.fromarray(fr)
        resized = im.resize(grid_size(im.width, im.height, 262144), Image.BICUBIC)
        stored = jpeg_roundtrip(resized, 88)
        i = len(times)
        times.append(t)
        d[f"raw{i}"] = fr
        d[f"resized{i}"] = np.asarray(resized, np.uint8)
        d[f"stored{i}"] = np.asarray(stored, np.uint8)
        d[f"pre{i}"] = idx.preprocess(stored).numpy()
        d[f"emb{i}"] = idx.embed_images([stored])[0]
    for j, q in enumerate(QUESTIONS):
        d[f"qids{j}"] = idx.tokenizer([q])[0].numpy().astype(np.int32)
        d[f"qemb{j}"] = idx.embed_text([q])[0]
    save_file(d, a.out)
    json.dump({"times": times, "questions": QUESTIONS}, open(a.out.replace(".safetensors", ".json"), "w"))
    print(len(times), "frames,", len(QUESTIONS), "questions")


if __name__ == "__main__":
    main()
