#!/usr/bin/env bash
# StreamBench (10 Oct; streambench.py): rows, frame bank (GPUs 1-5, light, next to retrieval v3), then after retrieval v3's accuracy part
# the caption pass (15 workers on GPUs 1-5, plus 4 on GPUs 6-7 once the v3 latency run is done), then answers on GPUs 1-5 (15 shards):
# source pixels; 768 px; 768 px + up to 8 retrieved at 384 px (sharp, vlm); the same with the 2 latest frames at source pixels.
# full, hybrid_qa, none. The judge runs separately once the LLaMA-3-8B-Instruct access is granted. GPU 0 stays free.
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda_envs/onestreamer-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd ~/onestreamer_repro/clm
S=out/streambench
mkdir -p $S out/frame_bank/streambench
log() { echo "$(date -u +%H:%M:%S) $*"; }
waitall() { local fail=0 p; for p in "$@"; do wait $p || fail=1; done; return $fail; }
python streambench.py --phase build || { log FAIL_BUILD; exit 1; }
pids=(); for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((k + 1)) python frame_bank.py --bench streambench --shard $k --nshards 5 > out/frame_bank/streambench.$k.log 2>&1 &
  pids+=($!); done
waitall "${pids[@]}" || { log FAIL_BANK; exit 1; }
log "bank $(ls out/frame_bank/streambench/*.npz | wc -l) videos"
until grep -q -E "ovo_fwd rows|FAIL" out/ret3.driver.log; do sleep 60; done
log "caption start"
rm -f $S/memory/*.lock
cap() { CUDA_VISIBLE_DEVICES=$1 python streambench.py --phase caption --shard $2 > $S/caption.$2.log 2>&1; }
pids=(); k=0; for g in 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5; do cap $g $k & pids+=($!); k=$((k + 1)); done
(until grep -q -E "LAT_RET3_DONE|FAIL" out/latency_ret3.driver.log 2>/dev/null; do sleep 60; done
 hp=(); for g in 6 6 7 7; do cap $g $k & hp+=($!); k=$((k + 1)); done; waitall "${hp[@]}") &
pids+=($!)
waitall "${pids[@]}" || { log FAIL_CAPTION; exit 1; }
n=$(ls $S/memory/item-*.json | wc -l); log "caption videos $n"
[ "$n" = 307 ] || { log FAIL_CAPTION_COUNT; exit 1; }
runN() {  # outdir extra-args...
  local o=$1; shift; local -a g=(1 1 1 2 2 2 3 3 3 4 4 4 5 5 5); local pids=() k
  for k in "${!g[@]}"; do
    CUDA_VISIBLE_DEVICES=${g[$k]} python answer_variants.py --bench streambench --out $o --shard $k --nshards ${#g[@]} \
      --variants full,hybrid_qa,none "$@" > $o.shard$k.log 2>&1 &
    pids+=($!)
  done
  waitall "${pids[@]}"
}
for spec in "ans_source:3:" "ans_768:3:--max-side 768" \
            "ans_ret2:6:--max-side 768 --ret-side 384 --recent 16 --retrieve 8 --select sharp,vlm" \
            "ans_ret3:6:--max-side 768 --ret-side 384 --recent 16 --retrieve 8 --select sharp,vlm --source-latest 2"; do
  name=${spec%%:*}; rest=${spec#*:}; per=${rest%%:*}; extra=${rest#*:}
  log "answer $name"
  runN $S/$name $extra || { log "FAIL_$name"; exit 1; }
  n=$(cat $S/$name/results_*.jsonl | wc -l); log "$name rows $n"
  [ "$n" = "$((1838 * per))" ] || { log "FAIL_${name}_COUNT"; exit 1; }
done
log SBN_ANS_DONE
