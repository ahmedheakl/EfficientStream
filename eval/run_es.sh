#!/usr/bin/env bash
# ES steps 1-2 (results/HYBRID_RULES.md): his Qwen3.5-0.8B (run 4) with s16 and v2fl on the analysis halves. GPUs 1-7, 2 workers per
# setting per GPU (beside the ST 32B pilot), GPU 0 stays free.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
log() { echo "$(date -u +%H:%M:%S) $*"; }
pids=()
for s in s16 v2fl; do k=0
  for g in 1 2 3 4 5 6 7; do for w in 0 1; do
    ( for part in sb ovo ovo_fwd odv ovb; do
        CUDA_VISIBLE_DEVICES=$g ~/conda_envs/malow/bin/python es_eval.py --bench $part --setting $s --out out/es/$s --shard $k --nshards 14 || exit 1
      done ) > out/es/$s.s$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done; done
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
for s in s16 v2fl; do log "$s rows $(cat out/es/$s/*/results_*.jsonl | wc -l)"; done
log "ES_DONE fail=$fail"
