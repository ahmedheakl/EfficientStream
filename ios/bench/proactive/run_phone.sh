#!/bin/zsh
# Runs the 8 batches of the proactive benchmark on the phone for each configuration, one launch per batch, and copies
# each batch's log back (logs/<config>_b<batch>.log). Skips batches whose log already ends with DONE.
U=${UDID:?the iPhone's UDID}
B=${BUNDLE:?the app's bundle id}
P=${ES_BENCH:-es-bench}/proactive_bench
mkdir -p $P/logs
for cfg in ${=CONFIGS:-both watch}; do
  extra=()
  [[ $cfg == watch ]] && extra=(-esnocaptions)
  for b in 0 1 2 3 4 5 6 7; do
    log=$P/logs/${cfg}_b$b.log
    if [[ -f $log ]] && grep -q " DONE" $log; then continue; fi
    echo "$(date +%T) $cfg batch $b"
    xcrun devicectl device process launch --terminate-existing --device $U $B -- -model qwen \
      -esqwendemo pb/scripts/pb_b$b.json -esfps 8 $extra > /dev/null 2>&1 || { sleep 30; continue; }
    sleep 60
    size=-1; still=0
    for i in $(seq 1 120); do
      xcrun devicectl device copy from --device $U --domain-type appDataContainer --domain-identifier $B \
        --source Documents/demo/last_demo.log --destination $log > /dev/null 2>&1
      grep -q " DONE" $log && break
      # no progress for 3 minutes (the phone was disconnected or the app stopped): run the batch again
      now=$(wc -c < $log 2>/dev/null || echo 0)
      if [[ $now -eq $size ]]; then still=$((still + 1)); else still=0; fi
      size=$now
      if [[ $still -ge 6 ]]; then
        echo "$(date +%T) $cfg batch $b stalled, relaunching"
        xcrun devicectl device process launch --terminate-existing --device $U $B -- -model qwen \
          -esqwendemo pb/scripts/pb_b$b.json -esfps 8 $extra > /dev/null 2>&1
        still=0; sleep 60
      fi
      sleep 30
    done
    echo "$(date +%T) $cfg batch $b: $(grep -c ' segment ' $log) segments, $(tail -1 $log | cut -c1-60)"
  done
done
echo FINISHED
