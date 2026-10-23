#!/usr/bin/env bash
# Frame banks for V5 / V6 (frame_bank.py --bench odv | ovb): 10 shards, 2 per GPU on GPUs 3-7, next to the V3 answer runs. GPU 0 stays free.
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
b=$1; pids=()
for k in $(seq 0 9); do
  CUDA_VISIBLE_DEVICES=$((3 + k % 5)) python frame_bank.py --bench $b --shard $k --nshards 10 > out/frame_bank.$b.$k.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
echo "$b npz $(ls out/frame_bank/$b/*.npz | wc -l)"
[ $fail = 0 ] && echo "BANK_DONE $b" || echo "FAIL_BANK $b"
