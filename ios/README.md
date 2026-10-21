# EfficientStream on iPhone

An iOS app for streaming video question answering: point the camera (or play a video) and ask a question at any
moment. It runs the evaluated EfficientStream pipeline on the phone (MiniCPM-V-4.6 in bfloat16 on MLX, the
MobileCLIP2-S0 frame index on Core ML), with nothing sent off the device. The app is derived from the
[Mobile-O](https://github.com/Amshaker/Mobile-O) iOS app.

## Layout

| Path | What |
|---|---|
| `EfficientStreamKit/` | Swift package: the pipeline (store, index, prompt, MiniCPM-V-4.6, session) + `es-verify` |
| `app/` | the Xcode project (`EfficientStream.xcodeproj`), SwiftUI app |
| `tools/export_ios.py` | weights -> `ios/models/` (MLX safetensors for MiniCPM-V-4.6, Core ML for MobileCLIP2-S0) |
| `tools/export_qwen.py` | the streaming Qwen3.5-0.8B (`stream-qwen3.5-0.8b/`) -> `ios/models/qwen/` (MLX safetensors) |
| `tools/install_models.sh` | copies `ios/models/` into the app on a connected iPhone |
| `tools/dump_*.py` | PyTorch reference dumps for `es-verify` |
| `bench/` | the 100-question streaming test set, the proactive latency benchmark, the StreamBench subset, their tools, and the phone's results |

| Swift file | Python counterpart |
|---|---|
| `Config.swift` | `config.py` (reads the exported `efficientstream_config.json`) |
| `FrameStore.swift` | `frames.py` `BoundedStore`, `index.py` `top_k_in_time_order` |
| `ImageOps.swift` | `frames.py` resizing (Pillow's resampler, ported with its fixed-point arithmetic), letterbox |
| `JPEGRoundTrip.swift` | `frames.py` `jpeg_roundtrip` (libjpeg-turbo's lossy steps, ported) |
| `FrameIndex.swift` | `index.py` (open_clip preprocessing and `SimpleTokenizer`, ported) |
| `Prompt.swift` | `prompt.py` + the MiniCPM-V-4.6 chat template |
| `ImageProcessor.swift` | transformers' `MiniCPMV4_6ImageProcessor` (slicing, views, placeholders) |
| `MiniCPMV46.swift` | transformers' `modeling_minicpmv4_6` + `qwen3_5` (vision tower, window merger, merger, gated-delta LM) |
| `Answerer.swift` | `model.py` (one prefill; option letter from the logits, or greedy decoding after `<answer>`) |
| `StreamSession.swift` | `session.py` |
| `Qwen35VL.swift` | transformers' `modeling_qwen3_5` (vision tower, M-RoPE; the text model is MiniCPM's, the same Qwen3.5) |
| `StreamQwen.swift` | `proactive/stream_qwen.py` `ask` (frames sized and patched as the Qwen processors do) |
| `QwenStreamer.swift` | `proactive/stream_qwen.py` `stream` (the proactive mode and the caption stream; the streaming KV cache) |
| `QwenVerify.swift` | checks the port against mlx-swift-lm's MLXVLM Qwen35 on the phone |

## What runs when

- **While the feed runs:** every 2 s one frame is resized to store size, JPEG round-tripped, embedded by
  MobileCLIP2-S0 and stored (at most 83, densest region evicted). On the phone, the stored frame's thumbnail
  (448 px letterbox, one view, 64 tokens) also goes through the MiniCPM vision tower once, when it enters the store;
  the reference computes exactly these features at question time, so this only moves work off the question.
- **A question:** MobileCLIP2-S0 text embedding, top-16 stored frames (in time order), the latest frame at 896 px
  (up to 9 views) through the vision tower, one prefill of about 1,700 tokens. Multiple choice: the letter from the
  logits. Open-ended: greedy decoding, streamed to the screen.

## Checked against the PyTorch package

`EfficientStreamKit/Sources/es-verify` runs the Swift code on a Mac against dumps of the PyTorch package
(bfloat16 on CPU; Big Buck Bunny, 854x480; 3 questions at 60 s and at 200 s; `tools/dump_*.py` make the dumps):

| Check | Result |
|---|---|
| Whole session from the decoded frames (store, eviction, JPEG, index, retrieval, latest frame, answer) | the same 16 retrieved moments in 6/6 questions; the same answer in 5/6 |
| Answers from the reference's inputs | multiple choice 4/4 identical (letters and probabilities); open-ended 1/2 word for word, the other identical for its first 25 words |
| Prompt token ids (whole prompt, ~1,714 tokens) | identical, 6/6 |
| Vision views (grids; pixels after the processor's resizes) | same grids; pixels within 1 of torchvision's |
| Vision features (same view pixels) | cosine 0.998-0.999 mean; on the few low tokens, PyTorch's bfloat16 is as far from float32 as MLX's |
| JPEG round trip (13 sizes, 1x1 to 896x504) | bit-exact with Pillow (libjpeg-turbo 3.1) |
| Store resize (Pillow bicubic) from raw frames | bit-exact |
| MobileCLIP2-S0: tokenizer (12 questions), preprocessing, embeddings | identical ids; identical inputs; cosine >= 0.9999 (Core ML float32) |

The one open-ended difference is the kind the package's own GPU check found (29/30 identical): bfloat16 arithmetic
differs between PyTorch and MLX, and greedy decoding diverges after many tokens.

Timings on an M4 Mac (Release build): a stored frame 0.33 s (resize, JPEG, MobileCLIP2-S0, the thumbnail through
the vision tower); a question 2.4-2.8 s to the answer (multiple choice) or to the first words (the latest frame's
three views through the vision tower 0.9 s, the prefill of 1,714 tokens 1.4 s).

## Measured on an iPhone 17 Pro

The 100-question streaming test set (`bench/`), run on the phone (2026-10-10). The videos were decoded in order on
the phone and each question was asked when the stream reached its time. The phone ran continuously while charging
over USB, so most questions ran with the phone hot ("fair" or "serious" thermal state), which slows it by about 30%.

| Stage (mean of 100 questions, 0.5 fps) | Seconds |
|---|---|
| Question embedding (MobileCLIP2-S0) + top-16 | 0.014 |
| Thumbnails (features computed when the frame was stored) | 0.000 |
| Latest frame: resize + JPEG / vision tower | 0.014 / 0.52 |
| Prefill (1,609 tokens on average) | 0.76 |
| **Question -> first words** | **1.31** (1.05 with the phone cool) |
| Decode (48 tokens on average, 37.8 tok/s) | 1.23 |
| **Question -> full answer** | **2.55** |

- **Each stored frame:** 0.18 s (resize + JPEG 0.01, MobileCLIP2-S0 0.014, thumbnail vision tower 0.15), so even
  1 frame per second runs in real time.
- **Latents at question time instead of when stored** (measured on the same questions): +0.43 s for the frame
  embeddings (47 frames on average, up to 1.1 s with a full store) and +2.5 s for the 16 thumbnails, so the first
  words would come at 4.3 s instead of 1.3 s.
- **Memory:** 3.3 GB after a question, peak 4.6 GB.

Answers. Open-ended answers were graded by hand against the reference: correct when the reference's key fact is
stated and not contradicted. This is not the paper's LLM judge. Multiple choice uses the option-letter logits, as
evaluated.

| | 0.5 fps open-ended | 1 fps open-ended | 0.5 fps MC | 1 fps MC |
|---|---|---|---|---|
| StreamBench (50) | 35 | 34 | - | - |
| OVO-Bench (25) | 12 | 11 | 17 | 16 |
| StreamingBench (25) | 15 | 19 | 18 | 19 |
| All | 62 / 100 | 64 / 100 | 35 / 50 | 35 / 50 |

Indexing every 1 s doubles the stored frames (104 against 52 by question time, on average) for +2 open-ended
answers (6 newly right, 4 newly wrong): within the noise of 100 questions. It costs about 0.15 s more to the first
words (more candidate frames fill all 16 thumbnail slots: 1,698 against 1,609 prompt tokens; both runs compared
with the phone hot: 1.55 against 1.39 s).

The per-question lines (answers, timings, memory, thermal state) are in `bench/results/results.jsonl`, and the
grades in `bench/results/grades.json`. `python bench/report.py bench/results/results.jsonl bench/results/grades.json`
prints the tables.

### Running the test set on the phone

```bash
# once: annotations + videos (ES_BENCH = a work folder; see make_testset.py for the annotation files it reads)
ES_BENCH=~/es-bench python ios/bench/fetch.py
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
  --domain-identifier <bundle id> --source ios/bench/manifest.json --destination Documents/bench/manifest.json
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
  --domain-identifier <bundle id> --source ~/es-bench/videos --destination Documents/bench/videos
# run 0.5 fps then 1 fps; progress prints as ESBENCH lines; a rerun resumes where it stopped
xcrun devicectl device process launch --console --device <udid> <bundle id> -- -esbench 0.5,1
xcrun devicectl device copy from --device <udid> --domain-type appDataContainer \
  --domain-identifier <bundle id> --source Documents/bench/results.jsonl --destination results.jsonl
```

### Recording a demo video

`-esdemo <script>` runs a scripted demo in the normal stream screen: each segment's video plays (fast-forwarded
between questions), slows to real time before each question, types the question into the question bar and sends it;
the answers are the model's, live. `demo/` has the script used for the demo videos (`demo_script.json`, test-set
videos), a recorder for the phone's screen, and the editor that cuts the recording into a 1920x1080 video (the
waiting sped up and labelled, questions and answers in real time, mirrored in large type).

```bash
swiftc -O ios/demo/record_iphone.swift -o record_iphone
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
  --domain-identifier <bundle id> --source ios/demo/demo_script.json --destination Documents/demo/demo_script.json
./record_iphone ~/es-demo/take3.mov ~/es-demo/stop > ~/es-demo/rec3.log &
xcrun devicectl device process launch --console --device <udid> <bundle id> -- -esdemo demo/demo_script.json \
  | tee ~/es-demo/console3.log    # ends with "ESDEMO ... DONE"; then: touch ~/es-demo/stop
python ios/demo/sync_offset.py ~/es-demo/take3.mov ~/es-demo/rec3.log ~/es-demo/console3.log  # -> "offset" in scenes
ES_DEMO=~/es-demo python ios/demo/make_demo.py ios/demo/scenes_60s.json demo_60s.mp4
```

The showcase video (`make_showcase.py`, `showcase.json`) puts the two models side by side, one section per moment a
question can come: at the beginning (the streaming Qwen3.5-0.8B's proactive mode: `-model qwen -esqwendemo`, a
segment's `prompt`), in the middle (its questions about the last 16 seconds: a segment's `questions`, typed so that
they are sent at video time `at` while the video plays) and at the end (EfficientStream's memory: `-esdemo`, with a
segment's `rate` for the fast-forward). Both demo modes log the playback position four times a second, so the editor
shows the source video large, at the position the phone is playing, beside the phone's screen; questions and answers
come from the logs, and so do the frames a memory question recalled (the stored frames closest to it, `recalled`).
The phone's recording plays in real time, except stretches an example's `ff` fast-forwards (the toaster's 23 silent
seconds at 6x), which are labelled on the video and the phone; a source the app itself played at 2x or 4x is labelled
too. Each take has its own `offset` (sync_offset.py; it includes the phone-to-Mac clock difference, so measure it per
session). The example videos are from YouTube (in `showcase.json`, by id).

```bash
./record_iphone ~/es-demo/take_q.mov ~/es-demo/stop_q > ~/es-demo/rec_q.log &
xcrun devicectl device process launch --console --device <udid> <bundle id> -- -model qwen \
  -esqwendemo demo/showcase_q.json > ~/es-demo/console_q.log    # then: touch ~/es-demo/stop_q
# the same for -model efficientStream -esdemo demo/showcase_m.json (take_m), then:
ES_DEMO=~/es-demo python ios/demo/make_showcase.py ios/demo/showcase.json showcase.mp4
```

## The streaming Qwen3.5-0.8B on the phone

`proactive/stream_qwen.py`'s model (`stream-qwen3.5-0.8b/` in ahmedheakl/efficientstream) runs on the phone with both
of the runner's modes. Its language model is the same Qwen3.5 as MiniCPM-V-4.6's, so the Swift port is shared; new
are Qwen3.5's vision tower and the M-RoPE positions of the vision tokens (`Qwen35VL.swift`).

- `ask` (`StreamQwen.swift`): a question on the 16 most recent frames, one per second; each frame goes through the
  vision tower when it arrives, so a question costs the prompt's prefill and the answer.
- `stream` (`QwenStreamer.swift`): one update per second (4 frames as two 2-frame blocks), answered `</Silence>`,
  `</Standby>` or `</Response>` + text. The caches are kept between seconds: a new second appends its turn, the same
  tokens the runner's re-rendered prompt adds.

In the app: ⋯ > Use Streaming Qwen3.5-0.8B (questions about the last 16 seconds; ⋯ > Watch for... is the proactive
mode); ⋯ > Use EfficientStream switches back. One model is loaded at a time.

| Check (there is no PyTorch reference: the models are not run on the Mac) | Result |
|---|---|
| Against mlx-swift-lm's MLXVLM Qwen35 on the phone (an independent MLX port, loading the original checkpoint; the same token ids and patch pixels), 3 prompts | first-token logits: cosine 0.9994-0.9997, KL 0.0014-0.0016, the same top-10 tokens (near-ties swapped); greedy answers identical until a near-tie flips ("Do it!" / "Do It!") |
| Prompts against the checkpoint's chat template | identical: `ask` open-ended and multiple choice, the `stream` window, a second appended to the cache |

### Questions (`ask`): accuracy and latency

The 100 questions above, on the phone (2026-10-11), the phone hot for most of them ("fair" or "serious"). 16 frames,
one per second up to the question time, sized for the vision tower in two ways:
- `pad640`: the decoded frame scaled down to at most 640 x 360 and padded with black to a multiple of 32, as the
  runner pads (240 tokens a frame at 16:9);
- `resized100k`: Qwen's smart_resize to at most 100,352 pixels, the size of the streaming windows the model was
  trained on (91 tokens a frame at 16:9).

| Stage (mean of 100 questions, seconds) | pad640 | resized100k | EfficientStream (above) |
|---|---|---|---|
| Each frame when it arrives: resize/pad + patches / vision tower | 0.006 / 0.061 | 0.005 / 0.021 | 0.18 per stored frame |
| Prompt tokens (of which visual) | 4,041 (3,649) | 1,793 (1,401) | 1,609 |
| Prefill | 1.89 (2,283 tok/s) | 0.71 (2,688 tok/s) | 0.76 |
| **Question -> first words** | **1.89** | **0.71** | **1.31** |
| Decode (11 tokens on average) | 0.31 (29 tok/s) | 0.29 (31 tok/s) | 1.23 (48 tokens) |
| **Question -> full answer** | **2.20** | **1.00** | **2.55** |
| Multiple choice: question -> letter | 1.70 | 0.63 | - |
| Memory after a question | 2.6 GB | 2.4 GB | 3.3 GB |

| Correct answers | pad640 | resized100k | EfficientStream 0.5 fps |
|---|---|---|---|
| StreamBench open-ended (50) | 31 | 28 | 35 |
| OVO-Bench open-ended (25) | 11 | 10 | 12 |
| StreamingBench open-ended (25) | 14 | 13 | 15 |
| **All open-ended (100)** | **56** | **51** | **62** |
| OVO-Bench multiple choice (25) | 16 | 17 | 17 |
| StreamingBench multiple choice (25) | 20 | 21 | 18 |
| **All multiple choice (50)** | **36** | **38** | **35** |

Inside the language model (`-esqwenprofile`: a 16-frame question on the kitchen video; each part evaluated on its own,
which removes the GPU's overlap between parts, so the shares matter more than the sums):

| Part | Prefill, 4,148 tokens (pad640) | Prefill, 1,812 tokens (resized100k) | Per decoded token |
|---|---|---|---|
| Gated-delta layers (18) | 0.85 s (58%) | 0.56 s (59%) | 10-14 ms |
| MLPs (24) | 0.43 s (29%) | 0.29 s (30%) | 13-18 ms |
| Full-attention layers (6) | 0.19 s (13%) | 0.09 s (10%) | 4.6-5.0 ms |
| Output head (248,320-token vocabulary) | 0.011 s | 0.010 s | 7.3-7.8 ms |
| Embeddings + vision features placed | 0.002 s | 0.001 s | - |

Open-ended answers graded by hand as above. The answers are short (11 tokens on average; the runner's prompt asks for
only the requested answer). With the last 16 seconds and no memory ("None for this ablation" in the runner's prompt),
questions about earlier moments fail (where did I put the toolbox, where did I pick the plate, counting over the
video), which EfficientStream's retrieval answers; on the current moment (the multiple-choice questions) it does as
well or better.

### The proactive mode (`stream`)

The kitchen video (`379fc8fe52.mp4`, 1280 x 960, frames resized to 352 x 256: 88 tokens a block, 2 blocks a second),
"Tell me when the person picks up a plate.", the first 60 seconds, one update per second (2026-10-12, phone hot):

| One update (mean, seconds) | Silent second, first 32 s | Silent second, after 32 s: runner's window (recomputed) | Silent second, after 32 s: context kept |
|---|---|---|---|
| 4 frames: resize + patches | 0.027 | 0.031 | 0.029 |
| Vision tower (2 blocks) | 0.054 | 0.051 | 0.065 |
| Prefill | 0.137 (215 tokens) | 3.859 (7,475 tokens) | 0.224 (216 tokens) |
| Answer (`</Silence>`, 1 token) | 0.036 | 0.039 | 0.043 |
| **Total** | **0.25** | **3.98** | **0.36** |
| Context | 4.2k tokens | 7.5k tokens | 10.7k tokens |

- A response (`</Response>` + about 40 tokens) takes 1.1-1.3 s more for the text. Both ways responded at the same
  seconds with the same words: 9 s ("The person is holding a yellow plate..."), 35 s ("...a transparent glass bowl"),
  49 s ("...a green plate").
- Real time needs an update under a second: appending each second does it (0.25-0.53 s up to 13.6k tokens of
  context); the runner's 32-second window, recomputed every second once full, takes 3.9 s. Keeping the window but
  recomputing it every few seconds (dropping several seconds at once), or a longer kept context, are the options.
- In the app (live camera, `-eswatch` / `-esask` launch checks): 0.25 s per second with nothing skipped, and a
  question asked meanwhile answered in 1.7 s to the first words; with the runner's window the updates fell behind
  after 32 s (7-13 s each, seconds skipped). The app therefore keeps appending and never recomputes (next section).

### The streaming KV cache

Both of the app's streams (the proactive mode's watch prompt and the PHCM caption stream) keep one KV cache each for
the whole video, built as the video plays (`QwenStreamer.swift`):

- **Append only.** Each second appends its turn: the second's time-labelled 2-frame blocks (8 frames at 8 fps, 416 x
  224, 91 tokens a block), then the model's answer. Frames and the model's own words interleave in one cache; an
  update prefills only the new second (about 420 tokens), never the window.
- **Evict without recomputing.** Every 8 seconds the turns older than the last 32 seconds are cut out of the 6
  full-attention layers' caches (`Context.evict(8)`; the first turn, with the instruction, stays). Positions are not
  renumbered (the M-RoPE gap is left as is), and the 18 gated-delta layers keep their recurrent state over everything
  seen, so nothing is prefilled again.
- **Questions fork the stream.** A question takes the caption stream's latest caches (copy-on-write), adds the seconds
  not yet in them, the captions of seconds already evicted (as text) and the question, and prefills only that; the
  stream goes on unchanged.
- **Frame compression (`Context.compress(n)`, an ablation).** Frames stay for the newest seconds only: after each
  update the second before the newest keeps its n frames that the newest decision attended to most (its attention,
  summed over the attention layers), the second before that loses the rest; all text stays.

Measured on the phone (2026-10-21; iPhone 17 Pro, plugged in, mostly "fair" or "serious" thermal state):

**Proactive latency** (`bench/proactive/`): 200 items from OVO-Bench's forward active responding tasks (48 CRR:
the clue that answers a question appears; 47 REC: an activity is repeated; 105 SSR: a COIN step starts), each a clip
from 16 s before the event to 12 s after it, the watch prompt given at the start, played in real time. Latency is from
the event on screen to the response's last token; a response counts if it comes from 1 s before to 10 s after the
event.

| | Watch stream + caption stream | Watch stream only |
|---|---|---|
| Responded in time | 138 / 200 (69%) | 137 / 200 (68%) |
| Responded before the event too | 125 | 124 |
| Latency, mean / median / p90 | 3.65 / 2.81 / 6.54 s | 3.56 / 2.56 / 7.33 s |
| ... waiting for the second to end (median) | 1.00 s | 1.00 s |
| ... processing: vision, prefill, decoding (median) | 1.65 s | 1.41 s |
| Silent second (median) | 0.35 s | 0.32 s |

By task (watch + captions): CRR 4/48 (it answers right after the prompt, before the clue), REC 39/47, SSR 95/105.
A response is about 38 tokens (about 1 s of decoding).

**StreamBench subset** (`bench/streambench_subset/`): 50 videos (16 Ego, 6 Movie, 28 WebVideo; 158 minutes), 200
open-ended questions balanced over the six types, the caption stream at 8 fps, each question asked at its time on the
stream's cache; graded by hand against the references (`grading_rubric.md`; grades in
`bench/results/streambench_subset_grades.json`).

| | `evict8` (the app) | `compress4` | `compress` (2 frames) |
|---|---|---|---|
| SF / SM / LM | 24 / 15 / 15 of 34 / 34 / 33 | 22 / 12 / 10 | 21 / 12 / 9 |
| OS / KG / CI | 11 / 16 / 19 of 33 / 34 / 32 | 4 / 19 / 15 | 3 / 19 / 12 |
| **Correct** | **100 / 200 (50.0%)** | 82 / 200 (41.0%) | 76 / 200 (38.0%) |
| One second of video, median (silent / with a caption) | 0.56 s (0.55 / 1.32) | 0.49 s (0.48 / 1.15) | 0.45 s (0.44 / 1.07) |
| Question -> first words / full answer, median | 0.23 / 0.74 s | 0.20 / 0.65 s | 0.20 / 0.64 s |
| Context at a question, median | 15.2k tokens | 3.2k tokens | 3.0k tokens |

Compression saves 15-20% per second and 5x the context, but costs 9-12 points, mostly on questions about where
something was and about earlier moments; the attention-based choice nearly always keeps the second's first frames
(the first block in 82% of seconds with `compress`, among the two kept in 92% with `compress4`).

```bash
# proactive: items, clips and the phone's batch scripts, then both configurations, then the scores
ES_BENCH=~/es-bench python ios/bench/proactive/make_items.py      # needs $ES_BENCH/raw/ovo.json (OVO-Bench)
ES_BENCH=~/es-bench python ios/bench/proactive/fetch_clips.py
ES_BENCH=~/es-bench python ios/bench/proactive/make_scripts.py
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer --domain-identifier <bundle id> \
  --source ~/es-bench/proactive_bench/clips --destination Documents/pb/clips    # and scripts -> Documents/pb/scripts
UDID=<udid> BUNDLE=<bundle id> ES_BENCH=~/es-bench ios/bench/proactive/run_phone.sh
ES_BENCH=~/es-bench python ios/bench/proactive/score.py both watch
# StreamBench subset: needs $ES_BENCH/raw/sb (StreamBench v0.3); videos -> Documents/sb/videos, manifest.json -> Documents/sb
ES_BENCH=~/es-bench python ios/bench/streambench_subset/make_subset.py
ES_BENCH=~/es-bench python ios/bench/streambench_subset/fetch.py
UDID=<udid> BUNDLE=<bundle id> ES_BENCH=~/es-bench ios/bench/streambench_subset/run_phone.sh
ES_BENCH=~/es-bench python ios/bench/streambench_subset/report.py compress compress4 evict8
```

### Running it

```bash
hf download ahmedheakl/efficientstream --include "stream-qwen3.5-0.8b/*" --local-dir ~/es-qwen
python ios/tools/export_qwen.py --src ~/es-qwen/stream-qwen3.5-0.8b --out ios/models/qwen   # numpy only
xcrun devicectl device copy to --device <udid> --domain-type appDataContainer \
  --domain-identifier <bundle id> --source ios/models/qwen --destination Documents/models/qwen
# the test set (as -esbench): frame sizes pad640, resized100k, pad (decoded pixels), padN, resizedN;
# -esbenches OVO-Bench,StreamingBench for a subset; results in Documents/bench/qwen_results.jsonl
xcrun devicectl device process launch --console --device <udid> <bundle id> -- -esqwen pad640,resized100k
# the proactive mode on a test-set video, timed per second (Documents/bench/qwen_stream.jsonl)
xcrun devicectl device process launch --console --device <udid> <bundle id> -- \
  -esqwenstream "379fc8fe52.mp4,60,slidingWindow;379fc8fe52.mp4,60,growing" \
  -esprompt "Tell me when the person picks up a plate."
# against MLXVLM's Qwen35 (Documents/models/qwen-ref: the checkpoint's config.json and model.safetensors)
xcrun devicectl device process launch --console --device <udid> <bundle id> -- -esqwenverify
python ios/bench/report_qwen.py ios/bench/results/qwen_results.jsonl ios/bench/results/qwen_grades.json \
  ios/bench/results/results.jsonl ios/bench/results/grades.json
```

## Run it on an iPhone

Requirements: a Mac with Xcode 27 (with the Metal Toolchain component), an iPhone 15 Pro or newer (8 GB RAM; the
app holds about 3 GB of weights), Python 3.10+ for the export.

1. Export the models (once; downloads the weights from Hugging Face, about 3 GB; the repo is private, so log in
   with `hf auth login` first):
   ```bash
   pip install -r requirements.txt coremltools
   pip install -e .
   python ios/tools/export_ios.py --weights ahmedheakl/efficientstream --out ios/models
   ```
2. Build and install the app: open `ios/app/EfficientStream.xcodeproj`, set your team and a unique bundle id in
   `ios/app/Configuration/Build.xcconfig` (or in Signing & Capabilities), select your iPhone and press Run.
   The scheme runs the Release configuration: the image code is slow without optimization.
3. Copy the models to the phone (the app shows a setup screen until they are there):
   ```bash
   ios/tools/install_models.sh
   ```
   then tap **Check again** in the app.

## Credits

The app is built on the [Mobile-O](https://github.com/Amshaker/Mobile-O) iOS app. Models: MiniCPM-V-4.6 (OpenBMB),
MobileCLIP2-S0 (Apple).
