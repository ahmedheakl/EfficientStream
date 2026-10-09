# EfficientStream (mobile branch)

Streaming video question answering with a small, fixed budget: a bounded store of past frames, a MobileCLIP2-S0 frame
index, and **one forward pass** of MiniCPM-V-4.6 per question. No captioner runs while the video streams.

Weights: [`ahmedheakl/efficientstream`](https://huggingface.co/ahmedheakl/efficientstream)
(`backbone/` = MiniCPM-V-4.6, `encoder/` = MobileCLIP2-S0, `efficientstream_config.json` = the settings below).

## How it works

```
while the video plays (every 2 s):   frame -> resize (<= 262k px) -> MobileCLIP2-S0 image embedding -> bounded store (<= 83 frames)
when a question is asked:            question -> MobileCLIP2-S0 text embedding -> top-16 stored frames by cosine (shown in time order)
                                     prompt = 16 thumbnails (448 px, letterboxed, time-stamped) + the latest frame (896 px, "(now)") + question
                                     MiniCPM-V-4.6, one prefill (~1,700 tokens); multiple choice: the option letter from the logits,
                                     open-ended: greedy answer after a forced "<answer>"
```

The store keeps the first and the newest frame and, when full, drops the frame in the densest region, so it always
spans the whole stream. Per frame, the stream side costs one MobileCLIP2-S0 image embedding every 2 s and nothing
else; the language model only runs when a question is asked.

## Quick start

```bash
pip install -r requirements.txt          # Python >= 3.10, PyTorch >= 2.4
python scripts/run_video.py video.mp4 --at 95 \
    --question "What did the person put on the table?" \
    --option "A. a cup" --option "B. a phone" --option "C. a book" --option "D. keys"
```

```python
from efficientstream import EfficientStream

es = EfficientStream.from_pretrained("ahmedheakl/efficientstream", device="cuda")   # or a local folder
session = es.session()
for t, frame in frames:                     # (seconds, RGB uint8 H x W x 3 or PIL image), in time order, >= 0.5 fps
    session.add_frame(t, frame)
r = session.ask("What did the person put on the table?", ["A. a cup", "B. a phone", "C. a book", "D. keys"])
print(r.text, r.probs, r.frame_times, r.seconds)
r = session.ask("What is the person cooking?")     # open-ended
```

`session.add_store_frame(t, image)` and `session.set_latest(t, image)` feed the two halves separately (for replaying a
pre-sampled store, or a camera pipeline that delivers a preview stream and a full-resolution "now" frame).

## Results (evaluated configuration, multiple choice by option-letter logits, one run each)

| Benchmark | Accuracy | Frames per question | Prompt tokens per question |
|---|---|---|---|
| OVO-Bench (real-time + backward, 1,468 questions) | 53.27 | 16.1 | 1,696 |
| StreamingBench Real-Time (2,495 questions) | 70.94 | 15.1 | 1,581 |
| StreamBench (1,838 open-ended questions, LLM judge, mean of 6 classes) | 72.94 | 16.0 | 1,683 |

Same pipeline with CLIP ViT-L/14 instead of MobileCLIP2-S0 as the frame index: 52.79 / 70.53 / 71.86 (5.7 times more
encoder parameters, no measurable accuracy difference). Measured on AMD MI210 GPUs; the time to the first answer token
at question time was about 0.5 s (modelled benchmark mean).

Notes for reading these numbers:
- The multiple-choice benchmarks favour the present moment: the same model answering from the latest frame plus the 3
  newest thumbnails (4 frames, with a prompt sentence that allows asking for more context) scores 58.24 / 75.61, but
  51.67 on StreamBench. This configuration (16 retrieved frames)
  is the one that keeps history and wins on StreamBench.
- Multiple-choice answers are read from the logits of the option letters at the end of the prefill (no text is
  generated); open-ended answers are generated.

## Code

| File | What |
|---|---|
| `efficientstream/config.py` | every setting, with the evaluated values as defaults |
| `efficientstream/frames.py` | resizing, letterboxing, time stamps, the bounded store (no GPU) |
| `efficientstream/index.py` | MobileCLIP2-S0 frame / question embeddings, top-k retrieval |
| `efficientstream/model.py` | MiniCPM-V-4.6: mixed-scale image assembly, one prefill, letter read-out or greedy answer |
| `efficientstream/session.py` | `EfficientStream` (loading) and `StreamSession` (`add_frame`, `ask`) |
| `efficientstream/video.py` | frames from a video file (OpenCV) |
| `scripts/run_video.py` | command-line demo |
| `tests/test_cpu.py` | unit tests that need no weights: `python -m pytest tests/` |
| `docs/MOBILE.md` | what it takes to run this on an iPhone |

The package was checked against the evaluated benchmark rows (fed the evaluated stored frames): the same answer on
60 of 60 multiple-choice and 29 of 30 open-ended questions, the same prompt length on all 90; see "Verification" in
`docs/MOBILE.md`.

## Licenses

Code: see the repository's license. MiniCPM-V-4.6 (backbone): Apache-2.0 (OpenBMB). MobileCLIP2-S0 (encoder): Apple Machine Learning Research
Model License, **research use only**; its license is in `encoder/LICENSE` of the weights repository.
