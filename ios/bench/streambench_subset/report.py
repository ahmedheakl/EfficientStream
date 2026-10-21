"""StreamBench subset results from the phone (results_<context>.jsonl) and grades (grades.json:
{"<context>": {"<question id>": 1 | 0}}) as Markdown.

ES_BENCH=<work folder> python report.py compress compress4 evict8
"""
import json, os, sys, statistics as st, collections
from pathlib import Path

S = Path(os.environ.get('ES_BENCH', 'es-bench')) / 'sb_subset'
grades = json.load(open(S / 'grades.json')) if (S / 'grades.json').exists() else {}


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(round(p * (len(x) - 1))))] if x else float('nan')


def f(x):
    return f"{st.mean(x):.3f} | {st.median(x):.3f} | {pct(x, .9):.3f}" if x else "- | - | -"


for ctx in sys.argv[1:] or ['compress', 'compress4', 'evict8']:
    path = S / f'results_{ctx}.jsonl'
    if not path.exists():
        continue
    rows = [json.loads(l) for l in open(path) if l.strip()]
    secs = [r for r in rows if r['kind'] == 'second']
    qs = [r for r in rows if r['kind'] == 'question' and 'answer' in r]
    vids = [r for r in rows if r['kind'] == 'video']
    print(f"\n## {ctx}: {len(vids)} videos, {len(qs)} questions, {len(secs)} seconds of video\n")
    g = grades.get(ctx, {})
    if g:
        print("| Question type | Correct |\n|---|---|")
        by = collections.defaultdict(list)
        for q in qs:
            if str(q['id']) in g:
                by[q['class']].append(g[str(q['id'])])
                by['All'].append(g[str(q['id'])])
        for c in ['SF', 'SM', 'LM', 'OS', 'KG', 'CI', 'All']:
            v = by.get(c, [])
            print(f"| {c} | {sum(v)}/{len(v)} ({100 * sum(v) / max(1, len(v)):.1f}%) |")
        print()
    silent = [r['total'] for r in secs if r['answer'].strip().startswith('</Silence>')]
    writing = [r['total'] for r in secs if not r['answer'].strip().startswith('</Silence>')]
    print("| Per second of video (stream) | mean s | median s | p90 s |\n|---|---|---|---|")
    print(f"| Every second ({len(secs)}) | {f([r['total'] for r in secs])} |")
    print(f"| Silent seconds ({len(silent)}) | {f(silent)} |")
    print(f"| Seconds that write a caption or summary ({len(writing)}) | {f(writing)} |")
    print(f"| Vision tower (8 frames) | {f([r['vision'] for r in secs])} |")
    print(f"| Prefill | {f([r['prefill'] for r in secs])} |")
    if any(r.get('compress') for r in secs):
        print(f"| Choosing + dropping frame blocks | {f([r['compress'] for r in secs])} |")
    print("\n| Per question | mean s | median s | p90 s |\n|---|---|---|---|")
    print(f"| Question -> first words | {f([q['ttft'] for q in qs])} |")
    print(f"| Question -> full answer | {f([q['total'] for q in qs])} |")
    print(f"\nContext at the question: median {st.median(q['contextTokens'] for q in qs):.0f} tokens "
          f"(max {max(q['contextTokens'] for q in qs)}), frames {st.median(q['frames'] for q in qs):.0f}, "
          f"captions as text {st.median(q['captions'] for q in qs):.0f}; prompt tokens per question "
          f"{st.median(q['promptTokens'] for q in qs):.0f}; answer tokens {st.median(q['tokens'] for q in qs):.0f}")
    kept = collections.Counter(r['kept'] for r in secs if 'kept' in r)
    if kept:
        print(f"Kept block (of the 4 in a second): {dict(sorted(kept.items()))}")
    print(f"Wall time {sum(v['wall'] for v in vids) / 60:.0f} min for {sum(v['seconds'] for v in vids) / 60:.0f} min "
          f"of video; thermal states at questions: {dict(collections.Counter(q['thermal'] for q in qs))}")
