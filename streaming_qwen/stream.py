"""The caption stream with a streaming KV cache, and questions on it.

Once a second the second's frames go in as one user turn (2-frame blocks with time labels) and the model answers
</Silence>, </Observe> + a detail, or </Summary> + a summary of a finished segment. One cache is kept for the whole video:
  append   each second prefills only its own turn (about 123 tokens at 1 fps, 423 at 8 fps), and the answer's tokens go
           into the same cache as they are generated, never re-reading the window.
  evict    when `window + evict_every` seconds have piled up, the seconds older than the last `window` are cut out of the
           6 attention layers' caches, `evict_every` seconds at once. Nothing is recomputed: positions keep counting up
           (the cut leaves a gap) and the 18 gated-delta layers keep their state over everything seen. The first turn
           (system prompt, instruction, second 0) stays.
  ask      a question forks the cache (copy-on-write) and prefills only its own turn: the captions of the seconds no
           longer in the cache (compacted), the earlier frames that best match it, and the question. The stream goes on
           unchanged.
"""
import time
from dataclasses import dataclass

import torch

from .memory import CaptionMemory, compact, memory_text
from .model import evict, fork
from .prompts import CAPTION_INSTRUCTION, CAPTION_SYSTEM, RETRIEVED_POSTAMBLE, RETRIEVED_PREAMBLE


@dataclass
class Snapshot:
    """What a question forks: the cache after the latest second (its answer in, the <|im_end|> where decoding stopped
    not), the next position, and the seconds the cache holds after the first turn."""
    cache: object
    position: int
    pending: str
    context: int
    first: int
    last: int


class CaptionStream:
    def __init__(self, model, window=32, evict_every=8, max_new_tokens=128):
        self.m, self.window, self.evict_every, self.max_new = model, window, evict_every, max_new_tokens
        self.cache, self.position, self.pending, self.evicted = None, 0, "", False
        self.turns = []    # (second, cache index of the turn's <|im_start|>)

    @torch.inference_mode()
    def update(self, sec):
        """One second (a model.Second): appended, answered, and the oldest seconds cut when due. Returns a dict."""
        m, t0 = self.m, time.time()
        if self.cache is None:
            self.cache, self.position = m.new_cache(), 0
            text = (f"<|im_start|>system\n{CAPTION_SYSTEM}<|im_end|>\n<|im_start|>user\n"
                    f"{CAPTION_INSTRUCTION.replace('<VIDEO_CONTEXT>', sec.visual)}<|im_end|>\n<|im_start|>assistant\n")
            self.turns.append((sec.second, 0))
        else:
            # the previous answer's tokens are in the cache; its <|im_end|> (where decoding stopped) is not
            close = self.pending + "<|im_end|>\n"
            self.turns.append((sec.second, self.cache.get_seq_length() + len(m.ids(close))))
            text = close + f"<|im_start|>user\n{sec.visual}<|im_end|>\n<|im_start|>assistant\n"
        self.pending = ""
        ids = m.ids(text)
        logits, self.position = m.forward(ids, self.cache, self.position, sec.grids, sec.feats)
        m.sync(); t1 = time.time()
        gen, cur = [], int(logits.argmax())
        while cur not in m.eos and len(gen) < self.max_new:
            gen.append(cur)
            if len(gen) == 1 and cur == m.silence:
                # a one-token answer: the step that would only predict its <|im_end|> is skipped, and the token is
                # prefilled with the next second
                self.pending = m.decode([cur])
                break
            logits, self.position = m.forward([cur], self.cache, self.position)
            cur = int(logits.argmax())
        answer = m.decode(gen)
        m.sync(); t2 = time.time()
        if len(self.turns) > self.window + self.evict_every:
            self._evict_oldest()
        return dict(second=sec.second, answer=answer, prefill_tokens=len(ids), decode_tokens=len(gen),
                    vision=sec.vision, prefill=t1 - t0, decode=t2 - t1, total=sec.vision + time.time() - t0,
                    context=self.cache.get_seq_length(), position=self.position)

    def _evict_oldest(self):
        """The turns after the first, up to the last `window`, leave the attention layers' caches."""
        keep = len(self.turns) - self.window
        if keep <= 1:
            return
        a, b = self.turns[1][1], self.turns[keep][1]
        evict(self.cache, self.m.attn_layers, a, b)
        self.turns = [self.turns[0]] + [(s, i - (b - a)) for s, i in self.turns[keep:]]
        self.evicted = True

    def snapshot(self):
        """The cache after the latest second, forked: a question on it leaves the stream as it was."""
        if self.cache is None:
            return None
        first = self.turns[1][0] if self.evicted and len(self.turns) > 1 else self.turns[0][0]
        return Snapshot(fork(self.cache), self.position, self.pending, self.cache.get_seq_length(), first,
                        self.turns[-1][0])

    @torch.inference_mode()
    def ask(self, question, snap, memory=None, stills=(), max_new_tokens=96):
        """`question` on `snap`: one user turn (the `memory` text, the retrieved `stills` (model.Still, oldest first),
        the question) prefilled on its cache, the answer decoded greedily, never starting with one of the stream's own
        tags. Returns a dict."""
        m = self.m
        user = (memory + "\n" if memory else "")
        if stills:
            user += RETRIEVED_PREAMBLE + "".join(s.visual for s in stills) + RETRIEVED_POSTAMBLE + "\n"
        text = snap.pending + f"<|im_end|>\n<|im_start|>user\n{user}{question}<|im_end|>\n<|im_start|>assistant\n"
        ids = m.ids(text)
        feats = torch.cat([s.feats for s in stills]) if stills else None
        m.sync(); t0 = time.time()
        logits, position = m.forward(ids, snap.cache, snap.position, [s.grid for s in stills], feats)
        logits[m.tags] = -1e9
        cur = int(logits.argmax()); m.sync(); ttft = time.time() - t0
        gen = []
        while cur not in m.eos and len(gen) < max_new_tokens:
            gen.append(cur)
            logits, position = m.forward([cur], snap.cache, position)
            cur = int(logits.argmax())
        m.sync()
        raw = answer = m.decode(gen)
        for tag in ("</Response>", "Overall Summary:"):
            if answer.startswith(tag):
                answer = answer[len(tag):].strip()
        return dict(answer=answer, raw=raw, ttft=ttft, total=time.time() - t0, prompt_tokens=len(ids), tokens=len(gen),
                    context=snap.context, first=snap.first, last=snap.last)


class StreamingQwen:
    """The whole loop: feed() once a second with that second's frames; ask() at any moment."""

    def __init__(self, model, index, window=32, evict_every=8, retrieved=4, keep_recent=60.0):
        from .retrieval import FrameMemory
        self.m, self.stream, self.captions = model, CaptionStream(model, window, evict_every), CaptionMemory()
        self.frames, self.k, self.keep_recent = FrameMemory(index), retrieved, keep_recent

    def feed(self, second, frames, times):
        """Second `second` of the video: its PIL frames and their times. Returns the stream's update dict."""
        self.frames.add(second, frames[0])
        u = self.stream.update(self.m.second(frames, times, second))
        self.captions.add(u["answer"], float(second), float(second + 1))
        return u

    def ask(self, question):
        """A question now, on the stream's cache, with the captions of the seconds no longer in it (compacted) and the
        earlier frames that best match it. Returns the answer dict (None before the first second)."""
        snap = self.stream.snapshot()
        if snap is None:
            return None
        older = self.captions.before(snap.first)
        kept = compact(older, snap.last, self.keep_recent)
        found = self.frames.retrieve(question, before=snap.first, k=self.k)
        r = self.stream.ask(question, snap, memory_text(kept) if kept else None,
                            [self.m.still(im, t) for t, im in found])
        r.update(captions=len(kept), captions_before_compaction=len(older), retrieved=[t for t, _ in found])
        return r
