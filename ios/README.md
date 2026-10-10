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
| `tools/install_models.sh` | copies `ios/models/` into the app on a connected iPhone |
| `tools/dump_*.py` | PyTorch reference dumps for `es-verify` |

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
three views through the vision tower 0.9 s, the prefill of 1,714 tokens 1.4 s). Not yet measured on an iPhone.

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

## Licenses

The app (`ios/app/`) is derived from Mobile-O's iOS app, which is licensed CC BY-NC-SA 4.0, so `ios/app/` is
CC BY-NC-SA 4.0 as well (attribution to Mobile-O; non-commercial; share alike). MiniCPM-V-4.6: Apache-2.0.
MobileCLIP2-S0: Apple Machine Learning Research Model License (research use only).
