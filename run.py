"""Stream a video through the streaming Qwen3.5-0.8B and ask questions at given seconds.

    python run.py video.mp4 --questions "30:What is on the table?" "75:Where did I put the cup?"

Each second prints the stream's answer (</Silence>, or a caption) and its cost; each question prints the answer, what
it was asked on (the seconds in the cache, the compacted captions, the retrieved frames' times) and its latency.
"""
import argparse
import math
import os

import torch

from streaming_qwen import FrameIndex, Model, StreamingQwen


def seconds(video, fps):
    """(second, frames, times) for each second of the video at `fps`: the first decoded frame of each 1/fps slot."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30
    cur, frames, times, last, i = 0, [], [], -1, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / vfps
        i += 1
        slot = math.floor(t * fps + 1e-7)
        if slot <= last:
            continue
        last = slot
        if math.floor(t + 1e-7) > cur and frames:
            yield cur, frames, times
            frames, times = [], []
        cur = math.floor(t + 1e-7)
        frames.append(Image.fromarray(fr[:, :, ::-1]))
        times.append(t)
    cap.release()
    if frames:
        yield cur, frames, times


def download(repo, folder):
    if os.path.isdir(repo):
        return os.path.join(repo, folder)
    from huggingface_hub import snapshot_download
    return os.path.join(snapshot_download(repo, allow_patterns=[folder + "/*"]), folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--questions", nargs="*", default=[], help='"seconds:question"')
    ap.add_argument("--repo", default="ahmedheakl/efficientstream", help="Hugging Face repository or local folder")
    ap.add_argument("--fps", type=int, default=4, help="frames per second (the model was trained at 4)")
    ap.add_argument("--window", type=int, default=32, help="seconds kept in the cache")
    ap.add_argument("--evict-every", type=int, default=8, help="seconds cut at once")
    ap.add_argument("--retrieved", type=int, default=4, help="earlier frames shown with a question")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    model = Model(download(a.repo, "stream-qwen3.5-0.8b-r4"), a.device)
    index = FrameIndex(download(a.repo, "encoder"), a.device)
    sq = StreamingQwen(model, index, a.window, a.evict_every, a.retrieved)
    qs = sorted((float(q.split(":", 1)[0]), q.split(":", 1)[1]) for q in a.questions)

    def ask_until(t):
        while qs and qs[0][0] <= t:
            at, q = qs.pop(0)
            r = sq.ask(q)
            if r is None:
                print(f"  Q@{at:g}s (no video yet) {q}", flush=True)
                continue
            print(f"  Q@{at:g}s {q}\n    -> {r['answer']}\n    cache {r['first']}-{r['last']}s ({r['context']} tokens), "
                  f"{r['captions']} captions (of {r['captions_before_compaction']}), frames at "
                  f"{[round(t, 1) for t in r['retrieved']]}; {r['prompt_tokens']} prompt tokens, first words "
                  f"{r['ttft']:.3f}s, full answer {r['total']:.3f}s", flush=True)

    for s, frames, times in seconds(a.video, a.fps):
        ask_until(s)
        u = sq.feed(s, frames, times)
        print(f"{s:5d}s {u['total']:.3f}s (vision {u['vision']:.3f}, prefill {u['prefill']:.3f} / {u['prefill_tokens']} "
              f"tok, decode {u['decode']:.3f} / {u['decode_tokens']} tok) cache {u['context']} tok  {u['answer']}",
              flush=True)
    ask_until(math.inf)


if __name__ == "__main__":
    main()
