#!/usr/bin/env bash
# EXP5 for his models (results/HYBRID_RULES.md): Qwen3-VL-4B r4 (our harness) then Qwen3.5-0.8B r4 (his ask), full OVO / SB-RT / StreamBench,
# 16 recent vs + 8 retrieved; then the StreamBench judges. Starts after EXP6's core runs. GPUs 1-7 (1 worker each, beside the ODV / OVBench
# runs), GPU 0 stays free.
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
log() { echo "$(date -u +%H:%M:%S) $*"; }
RET="--retrieve 8 --recent 16 --ret-side 384 --select sharp"
until grep -q EXP6_CORE_DONE out/slides3_driver.log; do sleep 120; done
log "start"
m4() {  # tag part extra...
  local tag=$1 part=$2; shift 2; local o=out/m4/$tag/$part; mkdir -p out/m4/$tag; local pids=() k=0
  for g in 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g python answer_variants.py --bench $part --out $o --shard $k --nshards 7 --save-conf --variants none --model Qwen3VL-4B-r4 "$@" > $o.s$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "m4 $tag $part rows $(cat $o/results_*.jsonl 2>/dev/null | wc -l) fail=$fail"
}
q08() {  # setting part rows
  local s=$1 part=$2 rows=$3 o=out/es/$1; local pids=() k=0
  for g in 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g ~/conda_envs/malow/bin/python es_eval.py --bench $part --setting $s --rows $rows --out $o --shard $k --nshards 7 > $o.$part.full$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "q08 $s $part rows $(cat $o/$part/results_*.jsonl 2>/dev/null | wc -l) fail=$fail"
}
judge() {  # answer dir
  local pids=()
  for k in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=$((k + 2)) python streambench.py --phase judge --src $1 --shard $k --nshards 6 > $1.judge$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  log "judge $1 answers $(cat $1/results_*.jsonl | wc -l) judged $(cat $1/judged_*.jsonl 2>/dev/null | wc -l)"
}
for part in sb ovo ovo_fwd streambench; do m4 base $part; m4 ret $part $RET; done
for part in sb ovo ovo_fwd streambench; do
  q08 s16 $part out/es/rows_rest_$part.json
  q08 s16ret $part out/es/rows_all_$part.json
done
for d in out/m4/base/streambench out/m4/ret/streambench out/es/s16/streambench out/es/s16ret/streambench; do judge $d; done
log "MODELS_DONE"
