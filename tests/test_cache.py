"""Checks of the cache plumbing on a tiny randomly initialised Qwen3.5 text stack (no weights, CPU, seconds): appending in
chunks equals one prefill, a fork is unaffected by the stream going on, eviction, M-RoPE positions, the caption stream
end to end with a question on a fork, and the memory compaction and frame selection.   python tests/test_cache.py"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import DynamicCache, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from streaming_qwen import compact, memory_text
from streaming_qwen.model import Model, Second, Still, evict, fork, mrope
from streaming_qwen.retrieval import select_frames
from streaming_qwen.stream import CaptionStream

torch.manual_seed(0)
cfg = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, linear_num_value_heads=4, linear_num_key_heads=2,
                        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "rope_type": "default",
                                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})
cfg._attn_implementation = "sdpa"

# a fake tokenizer: special tokens whole, other characters one id each
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>",
            "<|image_pad|>", "</Silence>", "</Observe>", "</Summary>", "</Response>"]
SP = {s: 256 + i for i, s in enumerate(SPECIALS)}


def ids(text):
    out = []
    for part in re.split("(" + "|".join(map(re.escape, SPECIALS)) + ")", text):
        out += [SP[part]] if part in SP else [min(ord(c), 255) for c in part]
    return out


def decode(xs):
    s = "".join(SPECIALS[x - 256] if 256 <= x < 256 + len(SPECIALS) else chr(x) for x in xs)
    return s.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()


m = Model.__new__(Model)
m.lm, m.head, m.device = mq.Qwen3_5TextModel(cfg).eval(), torch.nn.Linear(64, 300, bias=False), "cpu"
m.video_pad, m.image_pad, m.eos, m.silence = SP["<|video_pad|>"], SP["<|image_pad|>"], {SP["<|im_end|>"], SP["<|endoftext|>"]}, SP["</Silence>"]
m.tags = [SP["</Silence>"], SP["</Observe>"], SP["</Summary>"]]
m.attn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
m.ids, m.decode, m.sync = ids, decode, lambda: None
m.new_cache = lambda: DynamicCache(config=cfg)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))


def run(chunks):
    c, pos = m.new_cache(), 0
    for ch in chunks:
        lo, pos = m.forward(ch, c, pos)
    return lo, c, pos


seq = [int(x) for x in torch.randint(0, 250, (90,))]
full, _, _ = run([seq])
inc, _, _ = run([seq[:40], seq[40:77], seq[77:78], seq[78:]])
check("appending in chunks == one prefill", torch.allclose(full, inc, atol=1e-4), f"max diff {(full - inc).abs().max():.1e}")

with torch.inference_mode():
    _, c, pos = run([seq[:60]])
    f1, f2 = fork(c), fork(c)
    l1, _ = m.forward(seq[60:70], f1, pos)
    m.forward(seq[70:90], c, pos)                       # the stream goes on
    l2, _ = m.forward(seq[60:70], f2, pos)
check("a fork is unaffected by the stream going on", torch.allclose(l1, l2, atol=1e-5))

with torch.inference_mode():
    _, c, pos = run([seq[:60]])
    state = {k: v.clone() for k, v in c.layers[0].recurrent_states.items()}
    evict(c, m.attn_layers, 10, 30)
    untouched = all(torch.equal(v, c.layers[0].recurrent_states[k]) for k, v in state.items())
    lo, pos2 = m.forward(seq[60:65], c, pos)
check("evict [10, 30): the attention cache shrinks, positions go on, the gated-delta state is untouched",
      c.layers[3].keys.shape[-2] == 45 and pos2 == 65 and untouched and bool(torch.isfinite(lo).all()),
      f"attention length {c.layers[3].keys.shape[-2]}, next position {pos2}")

p, nxt = mrope([1, 2] + [SP["<|video_pad|>"]] * 6 + [3] + [SP["<|image_pad|>"]] * 4 + [4],
               [(2, 3), (2, 2)], 5, (SP["<|video_pad|>"], SP["<|image_pad|>"]))
check("M-RoPE: text +1, a video block and an image block advance by max(rows, cols)",
      p[0].tolist() == [5, 6] + [7] * 6 + [10] + [11] * 4 + [13] and nxt == 14, f"t {p[0].tolist()}, next {nxt}")


def second(s, k=2):
    vis = "".join(f"<{s + j / k:.1f} seconds><|vision_start|>" + "<|video_pad|>" * 6 + "<|vision_end|>" for j in range(k))
    return Second(s, vis + f"<{s:.1f}-{s + 1:.1f} seconds>", [(2, 3)] * k, torch.randn(6 * k, 64))


st = CaptionStream(m, window=6, evict_every=2, max_new_tokens=4)
lengths = []
for s in range(20):
    u = st.update(second(s))
    lengths.append(u["context"])
    assert u["context"] == st.cache.layers[3].keys.shape[-2]
check("the stream keeps the first turn plus 6 to 8 seconds", 7 <= len(st.turns) <= 9 and st.turns[0][0] == 0,
      f"turns {[t for t, _ in st.turns]}, cache tokens per second {lengths}")
snap = st.snapshot()
still = Still(5.0, "<5.000s><|vision_start|>" + "<|image_pad|>" * 4 + "<|vision_end|>", (2, 2), torch.randn(4, 64))
r = st.ask("What happened?", snap, memory="[0s-3s] a cup on the table", stills=[still], max_new_tokens=4)
check("a question on a fork leaves the stream's cache as it was", st.cache.layers[3].keys.shape[-2] == snap.context,
      f"{r['prompt_tokens']} prompt tokens on a {r['context']}-token cache, seconds {snap.first}-{snap.last}")

ev = [dict(kind="observe", start=0, end=1, text="[0.5 seconds] a cup"), dict(kind="observe", start=1, end=2, text="[1.5 seconds] a cup"),
      dict(kind="observe", start=3, end=4, text="a plate"), dict(kind="summary", start=0, end=5, text="setting the table"),
      dict(kind="observe", start=95, end=96, text="a dog")]
kept = compact(ev, time=100, keep_recent=60)
check("compaction: a summary replaces the old details it covers, recent ones stay",
      [e["text"] for e in kept] == ["setting the table", "a dog"], str([e["text"] for e in kept]))
kept = compact(ev, time=30, keep_recent=60)
check("compaction: identical details merge into one spanning their times",
      [(e["text"], e["start"], e["end"]) for e in kept if e["kind"] == "observe"][0] == ("a cup", 0, 2))
check("memory text: summaries, then details", memory_text(kept).splitlines()[4:8] == ["[0s-5s] setting the table", "</SEMANTIC_SUMMARIES>", "<EVENT_CAPTIONS>", "[0s-2s] a cup"])
idx = select_frames([0, 1, 2, 10, 11, 20], [0.9, 0.85, 0.8, 0.7, 0.1, 0.6], [5, 5, 5, 5, 5, 5], k=3)
check("frame selection: best matches, 3 s apart, in time order", idx == [0, 3, 5], str(idx))
print("all passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
