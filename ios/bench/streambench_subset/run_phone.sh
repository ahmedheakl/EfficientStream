#!/bin/zsh
# Runs the StreamBench subset on the phone for each context policy (QwenStreamBench: -esqwensb manifest.json
# -escontext <ctx>), copying Documents/sb/results_<ctx>.jsonl back every minute. A run resumes where it stopped, so it
# is relaunched if the results stop growing for 15 minutes (the phone was disconnected or the app stopped).
U=${UDID:?the iPhone's UDID}
B=${BUNDLE:?the app's bundle id}
S=${ES_BENCH:-es-bench}/sb_subset
for ctx in ${=CONTEXTS:-compress compress4 evict8}; do
  out=$S/results_$ctx.jsonl
  last=-1; stuck=0
  launch() {
    xcrun devicectl device process launch --terminate-existing --device $U $B -- -model qwen \
      -esqwensb manifest.json -escontext $ctx -esfps 8 > /dev/null 2>&1
    echo "$(date +%T) launched $ctx"
  }
  launch
  while true; do
    sleep 60
    xcrun devicectl device copy from --device $U --domain-type appDataContainer --domain-identifier $B \
      --source Documents/sb/results_$ctx.jsonl --destination $out > /dev/null 2>&1
    n=$(grep -c '"kind":"video"' $out 2>/dev/null); n=${n:-0}
    lines=$(wc -l < $out 2>/dev/null); lines=${lines:-0}
    [[ $n -ge 50 ]] && break
    if [[ $lines -eq $last ]]; then stuck=$((stuck + 1)); else stuck=0; fi
    last=$lines
    if [[ $stuck -ge 15 ]]; then echo "$(date +%T) $ctx stuck at $n videos, relaunching"; launch; stuck=0; fi
  done
  echo "$(date +%T) $ctx done: $n videos"
done
