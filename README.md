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
| `proactive/stream_qwen.py` | the proactive streaming model (below), self-contained |

The package was checked against the evaluated benchmark rows (fed the evaluated stored frames): the same answer on
60 of 60 multiple-choice and 29 of 30 open-ended questions, the same prompt length on all 90; see "Verification" in
`docs/MOBILE.md`.

## Proactive streaming model (Qwen3.5-0.8B)

`proactive/stream_qwen.py` runs our streaming Qwen3.5-0.8B (weights: `stream-qwen3.5-0.8b-r4/` in
[ahmedheakl/efficientstream](https://huggingface.co/ahmedheakl/efficientstream); `--sub stream-qwen3.5-0.8b` loads the
earlier run-2 weights). It is Qwen3.5-0.8B fine-tuned with OneStreamer's recipe, our training run 4: state tokens
`</Silence>`, `</Standby>`, `</Response>` and, for the caption memory, `</Observe>`, `</Summary>`; the 969,843
OneStreamer-1M records whose media we could obtain plus 70,000 LLaVA-Video QA; 5,415 steps over 1,039,843 windows,
vision tower frozen (run 2: 1,136 steps over 200,000 windows, without the caption-memory data). It decides by itself
when to speak:

```bash
pip install "transformers>=5.0" opencv-python   # flash-linear-attention recommended on GPU
python proactive/stream_qwen.py video.mp4 --prompt "Tell me when the person picks something up."
```

- `stream(video, prompt)`: one update per second (4 frames per second, last 32 seconds in context), each answered
  `</Silence>`, `</Standby>` or `</Response>` + text, the layout it was trained on. Each second's frames go through the
  vision encoder once (cached while they stay in the window): same answers, about 11x faster than re-encoding.
- `ask(frames, question, options)`: a question on the 16 most recent frames.
- `ask_phcm(video, t, question, options)` / `--question`: OneStreamer's caption memory (PHCM), run-4 weights only. While
  the video plays, the model writes `</Observe>` details and `</Summary>` segment summaries (4 frames per second, a
  32-update window, at most 128 new tokens per update, as in the OneStreamer release); at second `t` it answers from
  that memory plus the 16 most recent frames. `--memory` prints the memory. Unlike the release's caption stream, which
  was written for Qwen3-VL, the runner passes `mm_token_type_ids`: without them Qwen3.5 gives the video tokens plain
  text positions instead of their multimodal RoPE positions, unlike in training. StreamingBench Real-Time sample_1:

```bash
python proactive/stream_qwen.py video.mp4 --time 82 --question "What logo is seen on the jersey of the player covering his mouth?" \
  --options "A. Apple." "B. Google." "C. Adidas." "D. Spotify."
# [1-2s] observe: [1s] A 'SUBSCRIBE' button appears on a blue and red background.
# [3-4s] observe: [3s] Players from both teams walk onto the field.
# [12-13s] observe: [12s] The scoreboard shows '0-0' and '0:00'.
# [21-22s] observe: [21s] The scoreboard updates to '2-0' and '1:00'.
# [32-33s] observe: [32s] The scoreboard updates to '3-0' and '1:00'.     (then 12 more times up to 79 s)
# answer: D.
```

- `caption(video)` / `--caption`: live captions, every 4 seconds one sentence about the last 8 seconds. By default it
  uses the captioning weights `stream-qwen3.5-0.8b-caption/` (the same model trained further on 12,000 one-sentence
  captions of 8-second clips written by Qwen3-VL-4B, mixed with 6,000 of its original training records):

```bash
python proactive/stream_qwen.py video.mp4 --caption
#    4s  The video opens with the logo for "The Life Center Chiropractic" and introduces Dr. Jeremy Brook as the host.
#   16s  A shirtless man in blue pants performs a series of slackingline movements in front of a house, with the text
#        "Slacklining for Optimal Balance" displayed on screen.
```

The runner uses flash attention when it is installed: with `sdpa`, Qwen3.5's vision encoder returned wrong features on
our AMD GPUs (every video got the same description).

Checked on one GPU from the uploaded run-4 weights (the PHCM example above). Scores with the OneStreamer release's
evaluation, every question:

| Weights | Setting | OVO-Bench | StreamingBench RT | StreamBench | ODV-Bench |
|---|---|---|---|---|---|
| run 4 (`stream-qwen3.5-0.8b-r4/`) | 16 recent frames | 63.68 | 80.68 | 48.97 | 55.32 |
| run 4 | PHCM, release's caption stream | 63.30 | 80.08 | 49.73 | - |
| run 2 (`stream-qwen3.5-0.8b/`) | 16 recent frames | 62.53 | 78.40 | 53.61 | 47.39 |

The PHCM row is the release's caption stream with `mm_token_type_ids` passed to `generate()` (as `ask_phcm` does):
without them, transformers 5 gives the video tokens text positions. Measured 2026-10-20; against 16 recent frames the
caption memory changes the scores by -0.38 / -0.60 / +0.76. Our Qwen3-VL-4B
trained the same way (run 4): 70.77 / 85.97 / 54.88 with 16 recent frames, 71.27 / 86.17 / 56.32 with PHCM. The caption memory often repeats itself, as in the example. Details on the model cards
(`stream-qwen3.5-0.8b-r4/README.md` and `stream-qwen3.5-0.8b/README.md` in the weights repository).

### Streaming MiniCPM-V-4.6, for comparison

`proactive/stream_minicpm.py` runs MiniCPM-V-4.6 trained with the same data and schedule (weights:
`stream-minicpm-v4.6/` in the same repository). Same calls as the Qwen runner (`stream()`, `ask()`, `caption()`), same
prompts and frame sampling; each frame is one 448-pixel image (66 tokens for a 16:9 frame) after a time label:

```bash
pip install "transformers>=5.17" opencv-python
python proactive/stream_minicpm.py video.mp4 --prompt "Tell me when the man starts walking on the line."
python proactive/stream_minicpm.py video.mp4 --caption
```

| Model (OneStreamer recipe, same data) | Parameters | OVO-Bench | StreamingBench RT | StreamBench |
|---|---|---|---|---|
| Qwen3.5-0.8B (`stream_qwen.py`, run-2 weights) | 1.11B | 62.53 | 78.40 | 53.61 |
| MiniCPM-V-4.6 (`stream_minicpm.py`) | 1.55B | 60.77 | 74.35 | 42.10 |

Checked on one GPU from the uploaded weights: all three calls give well-formed answers. Each frame goes through the
vision encoder once while it stays in the window (3.7x faster); in bfloat16 this shifts the features by a few percent,
as any change in how many frames are encoded together does, so an answer near a tie can differ (`cache=False` turns it
off).

## Licenses

Code: see the repository's license. MiniCPM-V-4.6 (backbone): Apache-2.0 (OpenBMB). Qwen3.5-0.8B (proactive model): Apache-2.0 (Qwen). MobileCLIP2-S0 (encoder): Apple Machine Learning Research
Model License, **research use only**; its license is in `encoder/LICENSE` of the weights repository.
