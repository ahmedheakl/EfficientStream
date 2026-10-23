#!/usr/bin/env bash
# Slide priority (results/HYBRID_RULES.md EXP3 / EXP5): (A) EXP5 answers on OVO, SB-RT, StreamBench; (B) StreamBench judge on GPUs 2-7
# while the timing pass runs alone on GPU 1; (C) resume EXP3 ODV/OVBench, then EXP5 ODV/OVBench. GPU 0 stays free.
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
log() { echo "$(date -u +%H:%M:%S) $*"; }
RET="--retrieve 8 --recent 16 --ret-side 384 --select sharp"
exp5() {  # part
  local part=$1 o=out/exp5/$1; mkdir -p out/exp5; local pids=() k=0
  for g in 1 2 3 4 5 6 7; do for w in 0 1; do
    CUDA_VISIBLE_DEVICES=$g python answer_variants.py --bench $part --out $o --shard $k --nshards 14 --save-conf --variants none $RET > $o.s$k.log 2>&1 &
    pids+=($!); k=$((k + 1))
  done; done
  local fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  log "exp5 $part rows $(cat $o/results_*.jsonl | wc -l) fail=$fail"
}
for part in sb ovo ovo_fwd streambench; do exp5 $part; done
log "A_DONE"
pids=()
for k in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$((k + 2)) python streambench.py --phase judge --src out/exp5/streambench --shard $k --nshards 6 > out/exp5/streambench.judge$k.log 2>&1 &
  pids+=($!)
done
for part in ovo sb; do
  CUDA_VISIBLE_DEVICES=1 python answer_variants.py --bench $part --out out/timing/base/$part --ids out/timing/ids.json --variants none,full,hybrid_qa --timing > out/timing/base_$part.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python answer_variants.py --bench $part --out out/timing/ret/$part --ids out/timing/ids.json --variants none $RET --timing > out/timing/ret_$part.log 2>&1
  log "timing $part base $(cat out/timing/base/$part/results_*.jsonl | wc -l) ret $(cat out/timing/ret/$part/results_*.jsonl | wc -l)"
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
log "judge exp5 streambench answers $(cat out/exp5/streambench/results_*.jsonl | wc -l) judged $(cat out/exp5/streambench/judged_*.jsonl | wc -l) fail=$fail"
log "SLIDES_CORE_DONE"
./run_exp3.sh >> out/exp3_driver.log 2>&1
for part in odv ovb; do exp5 $part; done
log "SLIDES_ALL_DONE"
