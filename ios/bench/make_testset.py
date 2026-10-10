"""Pick the 100-question open-ended test set: 50 StreamBench (native open-ended), 25 OVO-Bench (real-time + backward,
answer text as reference), 25 StreamingBench Real-Time (correct option text as reference). Questions at <= 300 s."""
import ast, csv, glob, json, random, re, collections
random.seed(0)
import os
R = os.path.join(os.environ.get('ES_BENCH', 'es-bench'), 'raw/')
sizes = {k.split('|')[1]: v for k, v in json.load(open(R + 'sizes.json')).items()}
out = []

# StreamBench: whole videos (all their questions <= 300 s), balanced over Ego / Movie / WebVideo, until 50
sb = json.load(open(R + 'sb/streaming_bench_v0.3.json'))
pool = {c: [v for v in sb if v['info']['class_1'] == c and max(b['time'] for b in v['breakpoint']) <= 300]
        for c in ['Ego', 'Movie', 'WebVideo']}
for c in pool: random.shuffle(pool[c])
n, ci = 0, 0
while n < 50:
    c = ['Ego', 'Movie', 'WebVideo'][ci % 3]; ci += 1
    v = pool[c].pop()
    for b in sorted(v['breakpoint'], key=lambda b: b['time']):
        if n == 50: break
        out.append(dict(bench='StreamBench', task=b['class'], video=f"streambench/{c}/{v['info']['video_path']}",
                        time=float(b['time']), question=b['question'], answer=b['answer'], options=None, gt=None))
        n += 1

# OVO-Bench real-time + backward (no HLD: its answer is "Unable to answer"), 3 per task, small videos
ovo = json.load(open(R + 'ovo.json'))
tasks = ['EPM', 'ASI', 'OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
used = set()
for i, t in enumerate(tasks):
    cand = [x for x in ovo if x['task'] == t and x['realtime'] <= 300 and 'videos/' + x['video'] in sizes
            and sizes['videos/' + x['video']] < 400e6 and x['video'] not in used]
    random.shuffle(cand)
    for x in cand[:4 if i == 0 else 3]:
        used.add(x['video'])
        out.append(dict(bench='OVO-Bench', task=t, video='ovobench/' + x['video'], time=float(x['realtime']),
                        question=x['question'].strip(), answer=x['answer'], options=x['options'], gt=x['gt']))

# StreamingBench Real-Time: 10 tasks, questions that make sense without options, few videos
rows = {}
for p in glob.glob(R + 'sbench/*.csv'):
    if p.endswith('OP_TR.csv'): continue
    for r in csv.DictReader(open(p)): rows[r['question_id']] = r
def secs(s):
    v = [int(p) for p in s.split(':')]
    return sum(a * 60 ** i for i, a in enumerate(reversed(v)))
bad = re.compile(r'which of the following|following|option|true|false|correct', re.I)
cand = []
for r in rows.values():
    vid = 'sample_' + r['question_id'].split('_sample_')[1].rsplit('_', 1)[0] + '_real.mp4'
    if 'videos/' + vid not in sizes or sizes['videos/' + vid] > 200e6: continue
    if secs(r['time_stamp']) > 300 or bad.search(r['question']): continue
    cand.append((r, vid))
by_task = collections.defaultdict(list)
for r, v in cand: by_task[r['task_type']].append((r, v))
for t in by_task: random.shuffle(by_task[t])
tt = sorted(by_task)
k = 0
while sum(o['bench'] == 'StreamingBench' for o in out) < 25:
    t = tt[k % len(tt)]; k += 1
    if not by_task[t]: continue
    r, vid = by_task[t].pop()
    opts = [re.sub(r'^[A-D]\.\s*', '', o) for o in ast.literal_eval(r['options'])]
    gt = 'ABCD'.index(r['answer'].strip())
    out.append(dict(bench='StreamingBench', task=r['task_type'], video='streamingbench/' + vid,
                    time=float(secs(r['time_stamp'])), question=r['question'].strip(), answer=opts[gt],
                    options=opts, gt=gt))
for i, o in enumerate(out): o['id'] = i
json.dump(out, open(os.path.join(os.environ.get('ES_BENCH', 'es-bench'), 'testset.json'), 'w'), indent=1)
print(len(out), collections.Counter(o['bench'] for o in out), len({o['video'] for o in out}), 'videos')
print(collections.Counter((o['bench'], o['task']) for o in out))
