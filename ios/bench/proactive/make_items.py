"""200 proactive items from OVO-Bench's forward active responding tasks, each with the moment the event first appears:
48 CRR (the clue that answers the question: clue_time), 47 REC (the first repetition of the activity: start_times[0]),
105 SSR (a step of a COIN tutorial: its start_time). Each item is a clip from PRE s before the event (or the video start)
to POST s after it; the watch prompt is given at the clip start. Events in the first 4 s of a video are left out (the
model needs a few seconds of "nothing yet")."""
import json, os, random, hashlib
random.seed(0)
B = os.environ.get('ES_BENCH', 'es-bench').rstrip('/') + '/'
PRE, POST = 16.0, 12.0
ovo = json.load(open(B + 'raw/ovo.json'))
items = []

def add(task, x, onset, prompt, ref):
    start = max(0.0, onset - PRE)
    items.append(dict(task=task, ovo_id=x['id'], video='ovobench/' + x['video'], onset=float(onset), start=start,
                      end=float(onset + POST), prompt=prompt, reference=ref))

for x in [x for x in ovo if x['task'] == 'CRR']:
    if x['clue_time'] >= 4:
        q = x['question'].strip().rstrip('?')
        add('CRR', x, x['clue_time'], f"Tell me as soon as you can answer this: {q}?", x['answer'])

for x in [x for x in ovo if x['task'] == 'REC']:
    if x['start_times'][0] >= 4:
        add('REC', x, x['start_times'][0], f"Tell me when someone is {x['activity']}.", x['activity'])

# SSR: steps that start >= 4 s and >= 3 s after the previous step; round robin over the tutorials until 105
pool = []
for x in [x for x in ovo if x['task'] == 'SSR']:
    steps = [(i, s, t) for i, (s, t) in enumerate(zip(x['all_steps'], x['start_time']))
             if t >= 4 and (i == 0 or t - x['start_time'][i - 1] >= 3)]
    random.shuffle(steps)
    pool.append((x, steps))
random.shuffle(pool)
n = 0
while n < 105:
    for x, steps in pool:
        if steps and n < 105:
            i, s, t = steps.pop()
            add('SSR', x, t, f"Tell me when they {s}.", s)
            n += 1

for k, it in enumerate(items):
    it['id'] = k
    it['clip'] = f"pb_{k:03d}.mp4"
print(len(items), {t: sum(1 for i in items if i['task'] == t) for t in ('CRR', 'REC', 'SSR')})
json.dump(items, open(B + 'proactive_bench/items.json', 'w'), indent=1)
