#!/usr/bin/env bash
# Frame bank for retrieval (frame_bank.py): StreamingBench and OVO videos, 20 shards each, 8 processes per GPU on GPUs 1-5 next to the
# resolution sweep (CLIP ViT-H/14 fp16 uses ~3 GB). GPU 0 stays free; GPUs 6-7 are left to the latency runs.
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
pids=()
for b in sb ovo; do
  for k in $(seq 0 19); do
    CUDA_VISIBLE_DEVICES=$((1 + k % 5)) python frame_bank.py --bench $b --shard $k --nshards 20 > out/frame_bank.$b.$k.log 2>&1 &
    pids+=($!)
  done
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
for b in sb ovo; do echo "$b npz $(ls out/frame_bank/$b/*.npz | wc -l)"; done
[ $fail = 0 ] && echo BANK_ALL_DONE || echo FAIL_BANK
