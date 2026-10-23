# Benchmark harness: the streaming Qwen3.5-0.8B

The code that loads the streaming Qwen3.5-0.8B (`proactive/stream_qwen.py`) and runs it on OVO-Bench, StreamingBench (real-time),
StreamBench, ODV-Bench and OVBench. Everything except `export_rows.py` and `score_es.py` is copied unchanged from the analysis server
(`~/onestreamer_repro/clm`, 2026-10-23). Those two were written for this folder from the commands the runs used, and reproduce them
exactly (see "Check" below).

## Files

| File | What |
|---|---|
| `es_eval.py` | loads the 0.8B (`--sub`, default `stream-qwen3.5-0.8b-r4`) and answers every question of one benchmark in one setting (below); one JSON line per question |
| `export_rows.py` | the question rows `es_eval.py` reads: `--split dev` (the analysis half, by video) or `--split all`, plus `--rest <run dir>` |
| `frame_bank.py` | one frame per second of every benchmark video with a CLIP ViT-H/14 embedding and a sharpness score, and the retrieval rule: the 32 frames most similar to the question, scored z(similarity) + 0.3 z(log sharpness), taken at least 3 s apart |
| `answer_variants.py` | the OneStreamer-4B / Qwen3-VL harness: `load_bench` is the benchmark loader, and its `--retrieve` run on OneStreamer-4B made the frame picks that `s16ret` reuses |
| `streambench.py` | StreamBench rows and the judge (Meta-Llama-3-8B-Instruct, greedy): `--phase judge --src <run dir>/streambench` |
| `score_es.py` | per-benchmark scores of two settings on the same questions, with a paired bootstrap |
| `clm_check.py`, `consolidate_eval.py`, `folio.py`, `folio_mem.py`, `frame_bank2.py`, `internvl_engine.py` | modules the files above import |
| `run_*.sh` | the launchers as run (8 GPUs; each header says what it ran) |

## Settings (`es_eval.py --setting`)

| Setting | What `ask()` gets |
|---|---|
| `s16` | the 16 most recent frames, 1 fps, source pixels, as still images (OneStreamer's "recent frames" setting) |
| `s16ret` | the 8 frames retrieved for the question (CLIP + sharpness), 384 px, before the 16 recent |
| `s16vm` | the past before the 16 recent (1 fps into a 128-frame store, 320 px) as one video, before the 16 recent |
| `v2fl` | the last 16 s at 2 fps as one video, then the newest frame again as a still at source pixels |
| `phcm`, `phcmc` | the PHCM caption memory (`--mem-dir`, events written while the video plays), plus compaction for `phcmc`, with the 16 recent |
| `phcmret`, `phcmvm`, `phcmcret`, `phcmretvm` | the PHCM memory with the retrieved frames and/or the 128 s before the recent frames as one video |

`s16ret` does not retrieve by itself: it reads the frame times picked for each question by the OneStreamer-4B retrieval run
(`--ret-from out/exp5`). The picks depend only on the question and the frame bank, not on the answering model.

## Server layout (paths are hard-coded)

- `~/onestreamer_repro/clm/`: this folder; run everything from here, outputs go to `out/`.
- `~/onestreamer_repro/OneStreamer/`: the OneStreamer release (`MCG-NJU/OneStreamer`); `clm_check.py` puts its `onestreamer` package
  (benchmark loaders, `score_prediction`) on the path.
- `~/onestreamer_repro/path_to/datasets/` and `~/onestreamer_repro/path_to/models/`: benchmark videos and weights; the 0.8B is
  `path_to/models/efficientstream/<sub>`.
- `~/onestreamer_repro/EfficientStream/proactive/stream_qwen.py`: this repository's runner (branch `stream-qwen3.5-phcm`, commit
  `4c4da37`, for the numbers below). `s16vm` loads this branch's runner from `~/onestreamer_repro/es_vr/proactive/stream_qwen.py`.
- Two environments: `malow` (transformers 5) for `es_eval.py`, `onestreamer-eval` (the OneStreamer release's) for the rest.

## Steps (the retrieval test)

```bash
python export_rows.py --split all                     # out/es/rows_all_<bench>.json
python frame_bank.py --bench sb                       # banks: sb, ovo, streambench, odv, ovb (run_frame_bank.sh, run_bank_new.sh, run_streambench.sh)
python answer_variants.py --bench sb --out out/exp5/sb --save-conf --variants none \
    --retrieve 8 --recent 16 --ret-side 384 --select sharp   # OneStreamer-4B; its rows hold the frame picks (run_slides.sh)
python es_eval.py --bench sb --setting s16 --rows out/es/rows_all_sb.json --out out/es/s16          # malow env; add --shard k --nshards n
python es_eval.py --bench sb --setting s16ret --rows out/es/rows_all_sb.json --out out/es/s16ret
python streambench.py --phase judge --src out/es/s16ret/streambench   # StreamBench only
python score_es.py --arm s16ret --ref s16
```

Benchmark names: `ovo` + `ovo_fwd` = OVO-Bench (real-time and backward / forward), `sb` = StreamingBench real-time,
`streambench` = StreamBench, `odv` = ODV-Bench, `ovb` = OVBench.

## Check (2026-10-23)

`export_rows.py` gives the same rows as the 11 files the runs read (`--split dev`: ovo, ovo_fwd, sb, odv, ovb; `--split all`: the same
and streambench). `score_es.py --arm s16ret --ref s16` on those runs (run-4 weights, every question) gives the recorded scores:

| Benchmark | `s16` | `s16ret` | Difference [95% interval] | Questions |
|---|---|---|---|---|
| OVO-Bench | 63.83 | 64.06 | +0.24 [-0.59, +1.04] | 3,035 |
| StreamingBench (real-time) | 80.72 | 80.44 | -0.28 [-0.96, +0.40] | 2,500 |
| StreamBench | 49.89 | 54.30 | +4.41 [+2.83, +6.04] | 1,838 |
| ODV-Bench | 73.25 | 73.13 | -0.13 [-0.35, +0.11] | 6,348 |
| OVBench | 59.50 | 58.95 | -0.55 [-1.16, +0.06] | 7,090 |
