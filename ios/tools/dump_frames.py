#!/usr/bin/env python3
"""The decoded frames the reference session sees (efficientstream.video.iter_frames), for es-verify session."""
import argparse, json
import numpy as np
from safetensors.numpy import save_file
from efficientstream.video import iter_frames

p = argparse.ArgumentParser()
p.add_argument("video"); p.add_argument("--out", required=True)
p.add_argument("--fps", type=float, default=2.0); p.add_argument("--end", type=float, required=True)
a = p.parse_args()
d, times = {}, []
for t, fr in iter_frames(a.video, fps=a.fps, end=a.end):
    d[f"f{len(times)}"] = np.ascontiguousarray(fr)
    times.append(t)
save_file(d, a.out)
json.dump(times, open(a.out.replace(".safetensors", ".json"), "w"))
print(len(times), "frames, last at", times[-1])
