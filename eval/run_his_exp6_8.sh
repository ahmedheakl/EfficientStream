#!/usr/bin/env bash
# EXP6 on his models (results/HYBRID_RULES.md, EXP6-HIS): after the his-model queue, Qwen3.5-0.8B (es_eval s16vm) and Qwen3-VL-4B
# (answer_variants, the EXP6 flags) on all five benchmarks, then the StreamBench judge. GPUs 0-7 (the user freed GPU 0, 2026-10-21).
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
log() { echo "$(date -u +%H:%M:%S) $*"; }
q08() {  # part
  local part=$1 o=out/es/s16vm; local pids=() k=0
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g ~/conda_envs/malow/bin/python es_eval.py --bench $part --setting s16vm --rows out/es/rows_all_$part.json --out $o --shard $k --nshards 8 > $o.$part.h$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "q08 s16vm $part rows $(cat $o/$part/results_*.jsonl 2>/dev/null | wc -l) fail=$fail"
}
m4() {  # part
  local part=$1 o=out/m4/exp6/$1; mkdir -p out/m4/exp6; local pids=() k=0
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g python answer_variants.py --bench $part --out $o --shard $k --nshards 8 --save-conf --model Qwen3VL-4B-r4 --variants none --stills-past --past-fps 1 --past-side 320 --past-store 128 > $o.s$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "m4 exp6 $part rows $(cat $o/results_*.jsonl 2>/dev/null | wc -l) fail=$fail"
}
judge() {
  local pids=() k=0
  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k python streambench.py --phase judge --src $1 --shard $k --nshards 8 > $1.judge$k.log 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  log "judge $1 answers $(cat $1/results_*.jsonl | wc -l) judged $(cat $1/judged_*.jsonl 2>/dev/null | wc -l)"
}
until grep -q HIS_MODELS_DONE out/his_models_driver.log; do sleep 120; done
log "start"
for part in sb ovo ovo_fwd streambench odv ovb; do q08 $part; done
for part in sb ovo ovo_fwd streambench odv ovb; do m4 $part; done
for d in out/es/s16vm/streambench out/m4/exp6/streambench; do judge $d; done
log "HIS_EXP6_DONE"
