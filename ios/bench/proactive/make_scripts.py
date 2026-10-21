"""The phone's batch scripts for the proactive benchmark (QwenDemoModel `-esqwendemo`): 8 batches of 25 items, each
item a segment that plays its clip once with the watch prompt set at the start (instant, not typed). Writes
scripts/pb_b<batch>.json; copy clips/ and scripts/ to the app's Documents/pb/."""
import json, os
P = os.environ.get('ES_BENCH', 'es-bench').rstrip('/') + '/proactive_bench/'
items = json.load(open(P + 'items.json'))
os.makedirs(P + 'scripts', exist_ok=True)
for b in range(8):
    segs = [dict(video=f"pb/clips/pb_{it['id']:03d}.mp4", seconds=round(it['duration'] - 0.3, 2), prompt=it['prompt'],
                 instant=True, hold=0.5, id=str(it['id'])) for it in items[25 * b: 25 * (b + 1)]]
    json.dump(dict(segments=segs), open(P + f'scripts/pb_b{b}.json', 'w'), indent=1)
