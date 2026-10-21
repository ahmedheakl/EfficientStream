"""Measures how far the recording lags the phone's ESDEMO / ESQDEMO log times (screen-to-Mac latency plus the clock
difference between the phone and the Mac): movie time = (log time - first frame wall time) + offset. The clock
difference changes between sessions, so measure it for each take.

The keyboard is up while each question (or watch prompt) is typed and starts closing when it is sent (or the watch is
started). This finds the keyboard in the recording (the bottom of the screen matching the frame 1.2 s into the first
typing) and compares the moments it starts closing with the logged sends. Put the printed offset in the scenes file
("offset").

python sync_offset.py take.mov rec.log console.log
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

FPS, SW = 60, 120


def frames(path, fps, ss=None, n=None):
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    cmd += ["-i", path, "-vf", f"fps={fps},scale={SW}:-2,format=gray"]
    if n:
        cmd += ["-frames:v", str(n)]
    raw = subprocess.run(cmd + ["-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    w, h = map(int, subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         path], check=True, capture_output=True, text=True).stdout.split(","))
    return np.frombuffer(raw, np.uint8).reshape(-1, round(SW * h / w / 2) * 2, SW).astype(np.float32)


def main(take, rec_log, console_log):
    rec0 = float(re.search(r"first frame wall ([\d.]+)", Path(rec_log).read_text()).group(1))
    events = [(float(m.group(1)) - rec0, m.group(2)) for m in
              (re.match(r".*ES(?:Q)?DEMO ([\d.]+) (\w+)", line) for line in Path(console_log).read_text().splitlines())
              if m]
    types = [t for t, k in events if k in ("type", "ask")]
    sends = [t for t, k in events if k in ("send", "start")]
    template = frames(take, 1, ss=types[0] + 1.2, n=1)[0]
    y0, y1 = int(template.shape[0] * 0.66), int(template.shape[0] * 0.95)
    diff = np.abs(frames(take, FPS)[:, y0:y1] - template[y0:y1]).mean(axis=(1, 2))
    up = diff < 12
    closing = [i / FPS for i in range(1, len(up)) if up[i - 1] and not up[i]]
    offsets = [min(closing, key=lambda c: abs(c - s)) - s for s in sends]
    print("keyboard starts closing:", [round(c, 2) for c in closing])
    print("sends (log time)       :", [round(s, 2) for s in sends])
    print(f"offset {np.median(offsets):.3f} s (per question: {[round(o, 3) for o in offsets]})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
