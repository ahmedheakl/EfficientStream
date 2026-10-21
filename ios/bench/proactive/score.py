"""Proactive latency from the phone's demo logs (logs/<config>_b<batch>.log, one segment per item).

For each item the event appears at clip time e (items.json "event"). The phone plays the clip in real time and logs
its playback position 4 times a second, so W(t), the wall time at which video time t was on screen, is interpolated
from those samples. An update line is logged when the model's answer for second k is complete (its last token).

  hit:      the first shown response (</Response>, not a suppressed repeat) for a second k with
            floor(e) - 1 <= k <= e + WINDOW (the annotations are whole seconds; 1 s of slack before)
  latency:  W_done(response) - W(e): from the event on screen to the response's last token
            = waiting (video time from e to the end of second k, when its last frame has been shown: k + 1 - e)
            + processing (W_done - W(k + 1): queueing behind earlier seconds + vision + prefill + decode)
  early:    any response for a second k < floor(e) - 1 (before the event)

python score.py [both watch]  -> results.json and a Markdown summary on stdout
"""
import json, math, os, re, statistics as st, sys
from pathlib import Path

P = Path(os.environ.get('ES_BENCH', 'es-bench')) / 'proactive_bench'
WINDOW = 10.0
items = {str(i['id']): i for i in json.load(open(P / 'items.json'))}


def parse(path):
    segs, cur = [], None
    for line in Path(path).read_text().splitlines():
        m = re.match(r'ESQDEMO ([\d.]+) (\w+) ?(.*)', line)
        if not m:
            continue
        w, kind, rest = float(m.group(1)), m.group(2), m.group(3)
        if kind == 'segment':
            cur = dict(id=re.search(r' id (\S+)', rest).group(1), thermal=int(re.search(r'thermal (\d)', rest).group(1)),
                       pos=[], updates=[], repeats=[], start=w)
            segs.append(cur)
        elif cur is None:
            continue
        elif kind == 'pos':
            p, _, playing = rest.split()
            if playing == '1':
                cur['pos'].append((w, float(p)))
        elif kind == 'update':
            k, total, ntok, ans = rest.split(' ', 3)
            cur['updates'].append(dict(w=w, second=int(k), total=float(total), tokens=int(ntok), answer=ans))
        elif kind == 'repeat':
            cur['repeats'].append(int(rest.split()[0]))
        elif kind == 'pause':
            cur['end'] = w
    return segs


def wall_at(pos, t):
    """Wall time at which playback position t was reached (linear between the 4 Hz samples; the clip plays at 1x)."""
    pts = [(w, p) for w, p in pos if p > 0]
    for (w0, p0), (w1, p1) in zip(pts, pts[1:]):
        if p0 <= t <= p1 and p1 > p0:
            return w0 + (t - p0) * (w1 - w0) / (p1 - p0)
    if pts and t < pts[0][1]:
        return pts[0][0] - (pts[0][1] - t)
    if pts:
        return pts[-1][0] + (t - pts[-1][1])
    return math.nan


def score(cfg):
    out = []
    for log in sorted(P.glob(f'logs/{cfg}_b*.log')):
        for s in parse(log):
            it = items[s['id']]
            e = it['event']
            resp = [u for u in s['updates'] if u['answer'].startswith('</Response>')]
            early = [u for u in resp if u['second'] < math.floor(e) - 1]
            hit = next((u for u in resp if math.floor(e) - 1 <= u['second'] <= e + WINDOW), None)
            silent = [u['total'] for u in s['updates'] if not u['answer'].startswith('</Response>')]
            r = dict(id=it['id'], task=it['task'], event=e, thermal=s['thermal'], early=len(early),
                     responses=len(resp), silent_update=st.median(silent) if silent else None,
                     prompt=it['prompt'], reference=it['reference'])
            if hit:
                we = wall_at(s['pos'], e)
                wk = wall_at(s['pos'], hit['second'] + 1)
                r.update(hit=True, second=hit['second'], answer=hit['answer'], update_total=hit['total'],
                         tokens=hit['tokens'], latency=hit['w'] - we, waiting=hit['second'] + 1 - e,
                         processing=hit['w'] - wk)
            else:
                r['hit'] = False
            out.append(r)
    return out


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(round(p * (len(x) - 1))))] if x else float('nan')


def table(rows, label):
    hits = [r for r in rows if r['hit']]
    lat = [r['latency'] for r in hits]
    wt = [r['waiting'] for r in hits]
    pr = [r['processing'] for r in hits]
    f = lambda x: f"{st.mean(x):.2f} / {st.median(x):.2f} / {pct(x, .9):.2f}" if x else "-"
    return (f"| {label} | {len(rows)} | {len(hits)} ({100 * len(hits) / max(1, len(rows)):.0f}%) | "
            f"{sum(1 for r in rows if r['early'])} | {sum(1 for r in hits if not r['early'])} | {f(lat)} | {f(wt)} | {f(pr)} |")


if __name__ == '__main__':
    cfgs = sys.argv[1:] or ['both', 'watch']
    res = {c: score(c) for c in cfgs}
    json.dump(res, open(P / 'results.json', 'w'), indent=1)
    for c, rows in res.items():
        if not rows:
            continue
        print(f"\n### {c} ({len(rows)} items)\n")
        print("| Task | Items | Responded in time | Responded before the event | In time, nothing before "
              "| Latency s (mean / median / p90) | Waiting for the second to end | Processing |")
        print("|---|---|---|---|---|---|---|---|")
        for t in ['CRR', 'REC', 'SSR']:
            print(table([r for r in rows if r['task'] == t], t))
        print(table(rows, '**All**'))
