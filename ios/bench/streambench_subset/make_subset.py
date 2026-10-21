"""A StreamBench (v0.3) subset for the phone: 50 videos, 200 questions. Videos are drawn in proportion to the three
sources (Ego / Movie / WebVideo), then 4 questions per video picked so the six question types stay balanced (the
type with the fewest picks so far goes first; ties go to the earlier question, which keeps videos short)."""
import json, os, random, collections
random.seed(0)
B = os.environ.get('ES_BENCH', 'es-bench').rstrip('/') + '/'
sb = json.load(open(B + 'raw/sb/streaming_bench_v0.3.json'))
share = {'Ego': 16, 'Movie': 6, 'WebVideo': 28}
videos = []
for c, n in share.items():
    pool = [v for v in sb if v['info']['class_1'] == c and len(v['breakpoint']) >= 4]
    random.shuffle(pool)
    videos += pool[:n]
random.shuffle(videos)
count = collections.Counter()
out, qid = [], 0
for v in videos:
    bps = sorted(v['breakpoint'], key=lambda b: b['time'])
    pick = []
    for _ in range(4):
        rest = [b for b in bps if b not in pick]
        b = min(rest, key=lambda b: (count[b['class']], b['time']))
        pick.append(b)
        count[b['class']] += 1
    qs = []
    for b in sorted(pick, key=lambda b: b['time']):
        qs.append(dict(id=qid, time=float(b['time']), question=b['question'], answer=b['answer'], **{'class': b['class']}))
        qid += 1
    c = v['info']['class_1']
    out.append(dict(source=f"streambench/{c}/{v['info']['video_path']}", category=c, questions=qs))
print(len(out), 'videos', qid, 'questions', dict(count), 'video seconds', sum(max(q['time'] for q in v['questions']) for v in out))
json.dump(out, open(B + 'sb_subset/subset.json', 'w'), indent=1)
