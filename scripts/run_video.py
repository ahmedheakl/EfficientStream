#!/usr/bin/env python3
"""Stream a video file into EfficientStream and ask a question at a given moment.

  python scripts/run_video.py video.mp4 --at 95 --question "What did the person put on the table?" \
      --option "A. a cup" --option "B. a phone" --option "C. a book" --option "D. keys"
  python scripts/run_video.py video.mp4 --at 95 --question "What is the person cooking?"      # open-ended

Prints the answer, the moments shown to the model, and the time spent on the stream and on the question (video
decoding is excluded from both)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from efficientstream import EfficientStream  # noqa: E402
from efficientstream.video import iter_frames  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--question", required=True)
    p.add_argument("--option", action="append", help='multiple choice, repeat per option: "A. ...", "B. ..."')
    p.add_argument("--at", type=float, default=None, help="question time in seconds (default: the end of the video)")
    p.add_argument("--model", default="ahmedheakl/efficientstream", help="Hugging Face repo or local folder")
    p.add_argument("--device", default="cuda")
    p.add_argument("--decode-fps", type=float, default=2.0, help="frames fed to the session per second (>= 0.5)")
    a = p.parse_args()

    es = EfficientStream.from_pretrained(a.model, device=a.device)
    s = es.session()
    stream_s, n = 0.0, 0
    for t, frame in iter_frames(a.video, fps=a.decode_fps, end=a.at):
        t0 = time.perf_counter()
        s.add_frame(t, frame)                     # decoding happened in the iterator, outside the timer
        stream_s += time.perf_counter() - t0
        n += 1
    r = s.ask(a.question, a.option)
    print(json.dumps(dict(answer=r.text, probs=r.probs, now=round(r.now, 2), shown=[round(x, 1) for x in r.frame_times],
                          prompt_tokens=r.prompt_tokens, question_seconds=round(r.seconds, 3),
                          stream_seconds=round(stream_s, 3), frames_fed=n, stored=len(s.store)), indent=1))


if __name__ == "__main__":
    main()
