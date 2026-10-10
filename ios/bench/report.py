"""Summarises the phone's results.jsonl (+ grades.json: {"<run>": {"<id>": 1 | 0}}) as Markdown."""
import json, sys, statistics as st, collections
res = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
grades = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
runs = sorted({r['run'] for r in res})
def mean(x): x = [v for v in x if v is not None]; return st.mean(x) if x else float('nan')
def pct(x, p): x = sorted(x); return x[min(len(x) - 1, int(p * len(x)))] if x else float('nan')
out = []
for run in runs:
    R = [r for r in res if r['run'] == run]
    T = [r['timings'] for r in R]
    out.append(f"\n## {run} ({len(R)} questions, {R[0]['device']})\n")
    g = grades.get(run, {})
    if g:
        out.append("| Benchmark | Open-ended correct | Multiple choice correct |\n|---|---|---|")
        for b in ['StreamBench', 'OVO-Bench', 'StreamingBench', 'all']:
            B = [r for r in R if b == 'all' or r['bench'] == b]
            oe = [g[str(r['id'])] for r in B if str(r['id']) in g]
            mc = [r['mcCorrect'] for r in B if r.get('mcCorrect') is not None]
            out.append(f"| {b} | {sum(oe)}/{len(oe)} ({100*sum(oe)/max(len(oe),1):.0f}%) | "
                       + (f"{sum(mc)}/{len(mc)} ({100*sum(mc)/len(mc):.0f}%)" if mc else "-") + " |")
    keys = [('textEmbed', 'Question embedding (MobileCLIP2-S0)'), ('topK', 'Similarity + top-16'),
            ('thumbnails', 'Thumbnails (features from store time)'), ('latestPrep', 'Latest frame resize + JPEG'),
            ('latestVision', 'Latest frame vision tower'), ('tokenize', 'Tokenize'), ('prefill', 'Prefill'),
            ('decode', 'Decode (tokens after the first)'), ('timeToFirstToken', '**Question -> first word**'),
            ('total', '**Question -> full answer**')]
    out.append("\n| Stage | mean s | median s | p90 s |\n|---|---|---|---|")
    for k, n in keys:
        v = [t[k] for t in T]
        out.append(f"| {n} | {mean(v):.3f} | {st.median(v):.3f} | {pct(v, .9):.3f} |")
    dec = [t['decodeTokensPerSecond'] for t in T if t['generatedTokens'] > 1]
    pre = [t['promptTokens'] / t['prefill'] for t in T]
    out.append(f"\nPrompt tokens mean {mean([t['promptTokens'] for t in T]):.0f}; answer tokens mean "
               f"{mean([t['generatedTokens'] for t in T]):.1f}; decode {mean(dec):.1f} tok/s (min {min(dec):.1f}, "
               f"max {max(dec):.1f}); prefill {mean(pre):.0f} tok/s.")
    sm = [r['storeMean'] for r in R if r['storeFrames'] > 0]
    out.append(f"Per stored frame: resize+JPEG {mean([s['prep'] for s in sm]):.3f} s, MobileCLIP2-S0 "
               f"{mean([s['embed'] for s in sm]):.3f} s, thumbnail vision {mean([s['thumbnail'] for s in sm]):.3f} s.")
    la = [r for r in R if r.get('latentsAtQuestionEmbeds') is not None]
    if la:
        e = [r['latentsAtQuestionEmbeds'] for r in la]; th = [r['latentsAtQuestionThumbnails'] for r in la]
        c = [r['timings']['candidates'] for r in la]
        out.append(f"Latents at question time instead (measured): +{mean(e):.3f} s embeddings of {mean(c):.0f} "
                   f"frames (max {max(e):.3f}), +{mean(th):.3f} s thumbnails, = +{mean([a+b for a,b in zip(e,th)]):.2f} s "
                   f"per question (TTFT {mean([r['timings']['timeToFirstToken'] for r in la]):.2f} -> "
                   f"{mean([r['timings']['timeToFirstToken']+a+b for r,a,b in zip(la,e,th)]):.2f} s).")
    out.append(f"Memory: {mean([r['memoryMB'] for r in R]):.0f} MB after a question, peak "
               f"{max(r['peakMemoryMB'] for r in R):.0f} MB. Thermal: {dict(collections.Counter(r['thermal'] for r in R))}.")
print("\n".join(out))
