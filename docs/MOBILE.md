# Running EfficientStream on an iPhone: what it takes

This package runs the evaluated pipeline in PyTorch. Nothing here has been run on a phone yet; this note lists what
the pipeline needs, so that a port can be planned.

## The two models

| | MiniCPM-V-4.6 (`backbone/`) | MobileCLIP2-S0 (`encoder/`) |
|---|---|---|
| Role | answers, once per question | embeds one stored frame every 2 s, and the question |
| Parameters | 1.30 B (vision tower + resampler 0.55 B, language model 0.75 B) | 74.8 M (image and text towers) |
| Weights here | bfloat16 safetensors, 2.5 GB | open_clip safetensors, 0.3 GB |
| Language model | 24 layers, hidden size 1,024: 18 linear-attention (gated delta) + 6 full-attention layers, vocabulary 248,094 | - |
| Input per call | about 1,700 tokens: 16 thumbnails x 64 + the latest frame (up to 9 vision views, about 192 tokens) + text | one 256 x 256 image, or the question |

## Work per stream second and per question

- **Streaming:** every 2 s one frame is resized (at most 262,144 pixels), stored, and embedded once by MobileCLIP2-S0
  (one 256 x 256 image). No language model runs while the video plays. The store holds at most 83 frames (about 4 MB
  as JPEG) and their 512-dimensional embeddings (83 KB in float16).
- **A question:** one MobileCLIP2-S0 text embedding; a dot product over at most 83 vectors; 17 images through the
  MiniCPM vision tower (16 thumbnails of one view each + the latest frame); one prefill of about 1,700 tokens.
  Multiple choice reads the option letter from the logits of that prefill (no decoding); open-ended answers decode
  up to 128 tokens.
- On an AMD MI210 GPU the question side took about 0.5 s to the first answer token.

## Porting notes

- **MobileCLIP2-S0** is Apple's mobile image-text model; converting its image and text towers to Core ML is the
  straightforward part. Keep its own preprocessing (resize 256, centre crop 256, mean 0 / std 1) and normalise the
  embeddings. Retrieval quality was checked against CLIP ViT-L/14 at the same settings: no measurable difference.
- **MiniCPM-V-4.6** is the hard part: its language model mixes gated-delta (linear-attention) and full-attention
  layers (`qwen3_5_text`), so a phone runtime (Core ML, MLX, llama.cpp, ExecuTorch) must support that layer type, and
  the vision tower must accept several images of different sizes in one prompt. Check runtime support for this exact
  architecture before converting; it is the main risk of the port.
- **Precision:** evaluated in bfloat16. 4-bit or 8-bit weights would be needed on a phone (1.3 B parameters is
  about 0.7 GB at 4 bits); the accuracy after quantisation has not been measured.
- **Cheaper views if needed:** the thumbnails dominate the prompt (16 x 64 tokens); `retrieve_k` and `thumb_side` in
  `StreamConfig` change the view, and any change should be re-evaluated. For reference, a different evaluated view, the
  latest frame + the 3 newest thumbnails with a prompt sentence that allows asking for more context (not implemented
  in this package), scored higher on the multiple-choice benchmarks (58.24 / 75.61) and lower on StreamBench (51.67).
- **The latest frame** is the most important input for "now" questions. A camera pipeline can feed a low-rate preview
  stream to `add_store_frame` and a full-resolution still to `set_latest` just before `ask`.

## Verification

The package was fed the exact inputs of the evaluated benchmark runs (the stored 0.5 fps frames of each stream up to
the question and the frame at the question time) and compared with the stored rows of the evaluated configuration,
30 random questions per benchmark, on an AMD MI210 (`es_verify.py` in the research repository, 2026-10-09):

| Benchmark | Same answer | Same prompt length | Largest difference of an option probability |
|---|---|---|---|
| OVO-Bench (multiple choice) | 30 / 30 | 30 / 30 | 0.042 |
| StreamingBench Real-Time (multiple choice) | 30 / 30 | 30 / 30 | 0.092 |
| StreamBench (open-ended) | 29 / 30 | 30 / 30 | - |

The one open-ended difference is one word of a two-sentence answer ("Looking through the workshop" against "Looking at
the workshop"); the probability differences most likely come from computing the frame embeddings one by one instead of
in batches and from bfloat16 arithmetic. A question took 0.54-0.69 s on average (multiple choice; the first call of a
process takes a few seconds more) and 1.8 s on average (open-ended, decoding included) on one MI210, with the frames
already in the store.
