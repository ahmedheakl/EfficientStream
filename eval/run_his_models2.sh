#!/usr/bin/env bash
# The rest of run_his_models.sh on all 8 GPUs (the user freed GPU 0, 2026-10-21): starts beside the running Qwen3.5-0.8B s16ret odv step (7 shards, old driver);
# the remaining steps unchanged except 8 shards on GPUs 0-7. Logs to out/his_models_driver.log (run_his_exp6.sh waits for HIS_MODELS_DONE).
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
log() { echo "$(date -u +%H:%M:%S) $*"; }
RET="--retrieve 8 --recent 16 --ret-side 384 --select sharp"
GPUS="0 1 2 3 4 5 6 7"
q08() {  # setting part rows extra...
  local s=$1 part=$2 rows=$3; shift 3; local o=out/es/$s; local pids=() k=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g ~/conda_envs/malow/bin/python es_eval.py --bench $part --setting $s --rows $rows --out $o --shard $k --nshards 8 "$@" > $o.$part.h$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "q08 $s $part rows $(cat $o/$part/results_*.jsonl 2>/dev/null | wc -l) fail=$fail"
}
m4() {  # tag part extra...
  local tag=$1 part=$2; shift 2; local o=out/m4/$tag/$part; mkdir -p out/m4/$tag; local pids=() k=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g python answer_variants.py --bench $part --out $o --shard $k --nshards 8 --save-conf --model Qwen3VL-4B-r4 "$@" > $o.s$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "m4 $tag $part rows $(cat $o/results_*.jsonl 2>/dev/null | wc -l) fail=$fail"
}
judge() {
  local pids=() k=0
  for g in $GPUS; do CUDA_VISIBLE_DEVICES=$g python streambench.py --phase judge --src $1 --shard $k --nshards 8 > $1.judge$k.log 2>&1 & pids+=($!); k=$((k + 1)); done
  for p in "${pids[@]}"; do wait $p; done
  log "judge $1 answers $(cat $1/results_*.jsonl | wc -l) judged $(cat $1/judged_*.jsonl 2>/dev/null | wc -l)"
}
M4=../hismem/events/OneStreamer-repro4


q08 s16 ovb out/es/rows_rest_ovb.json; q08 s16ret ovb out/es/rows_all_ovb.json
log "Q08_DONE"
for part in sb ovo ovo_fwd; do m4 exp3 $part --variants none,full,hybrid_qa --mem-dir $M4; done
for part in streambench odv ovb; do m4 base $part --variants none; done
for part in sb ovo ovo_fwd streambench odv ovb; do m4 ret $part --variants none $RET; done
for d in out/m4/base/streambench out/m4/ret/streambench; do judge $d; done
log "HIS_MODELS_DONE"
