"""Streaming Qwen3.5-0.8B: Qwen3.5-0.8B fine-tuned with OneStreamer's state tokens on OneStreamer-1M records + LLaVA-Video QA.

Ways to use it (all as in training / evaluation; transformers >= 5.0, flash-linear-attention recommended):
  ask_phcm(video, t, question, options)  PHCM, OneStreamer's caption memory: while the video plays the model writes
                                  </Observe> details and </Summary> segment summaries (memorize()); at second t it answers
                                  from that memory plus the 16 most recent frames. Needs weights trained on the caption-
                                  memory data: stream-qwen3.5-0.8b-r4 (run 4) yes, stream-qwen3.5-0.8b (run 2) no
  ask(frames, question, options)  answer from the 16 most recent frames (OneStreamer's "recent frames" setting)
  stream(video, prompt)           one update per second of video, 4 frames per second, the last 32 seconds in context;
                                  the model answers every second with </Silence>, </Standby> or </Response> + text
  caption(video)                  live captions: every 4 seconds, one sentence about the last 8 seconds of video; best
                                  with the captioning weights (stream-qwen3.5-0.8b-caption: the same model trained further
                                  on one-sentence captions of 8-second clips written by Qwen3-VL-4B)
The narration the model learned from Streamo's continuous-description data is stream(video, NARRATE): terse change
notes ("Man turns head right.", every ~2 s), as in that data; caption() gives fuller sentences.
"""
import math
import numpy as np
import torch

STREAM_SYSTEM = 'You are a conservative assistant for real-time streaming video event detection.\nYou receive the video incrementally, with every update labeled by an absolute time interval such as <0s-1s>. Base each decision only on evidence observed so far.\n\nEvery </Response> is a final detection of one distinct event boundary. An unmatched, premature, or duplicate response is an error, so prioritize precision over frequent responses.\n\nAt every update, use exactly one of these three protocols:\n\n1. Use </Silence> when:\n   - No relevant event is currently supported by clear visual evidence.\n   - The requested event has not started, or a suspected event is no longer supported.\n   - The same event was already reported and no unmistakably new occurrence or step has completed.\n   - The scene is unchanged, shows only aftermath, or merely repeats evidence already used for a response.\n   Output exactly </Silence> and nothing else.\n\n2. Use </Standby> only when:\n   - A specific relevant event is visibly developing, but its completion or identity still needs confirmation.\n   - There is concrete new evidence, not just generic activity or uncertainty.\n   </Standby> is reversible. Return to </Silence> if later frames do not confirm a new event.\n   Output exactly </Standby> and nothing else.\n\n3. Use </Response> only when:\n   - The visual evidence clearly establishes that a new, distinct requested event or step has just completed.\n   - All essential entities, actions, attributes, and relationships required by the question are confirmed.\n   - This event has not already been reported.\n   Output </Response> followed by a concise answer.\n\nTask-specific rules:\n- For a single-event question, respond at most once in the entire stream. After responding, always use </Silence>.\n- For "How many times" questions, respond exactly once when each genuinely new occurrence completes. Do not repeat the same count while no new occurrence has happened.\n- For step-description questions, respond exactly once when each genuinely new requested step completes. Do not repeatedly report an ongoing or already completed step.\n\nDo not predict future actions, treat a precursor as completion, or use </Response> merely because the current frames are relevant. When uncertain, prefer </Standby> or </Silence>.\n'
QA_SYSTEM = 'You answer a multiple-choice question about a video after the observation period has ended.\nYou may receive two evidence sources:\n1. A causal textual memory produced while the video was playing, without access to the current question. It may be incomplete or imperfect; use it to recover earlier events.\n2. Recent video frames from immediately before the question time, ordered from oldest to newest. Use them for the latest visual state.\n\nUse the recent frames as stronger evidence when they conflict with the textual memory. Base the answer only on the supplied memory and frames, select the single best option, and output only its option letter.'
QA_DIRECT_SYSTEM = 'You answer a question about a video after the observation period has ended.\nYou may receive two evidence sources:\n1. A causal textual memory produced while the video was playing, without access to the current question. It may be incomplete or imperfect; use it to recover earlier events.\n2. Recent video frames from immediately before the question time, ordered from oldest to newest. Use them for the latest visual state.\n\nUse the recent frames as stronger evidence when they conflict with the textual memory. Base the answer only on the supplied memory and frames. Follow the response format requested by the question exactly and output only the requested answer.'
POST_PROMPT = 'Answer with ONLY the single uppercase letter of the correct option.'
NARRATE = "Narrate the video in real time, updating the description frame-by-frame or moment-by-moment as events unfold."
CAPTION_Q = "Describe what happens in this clip in one sentence."
MIN_PIXELS, MAX_PIXELS = 3136, 100352
CAPTION_SYSTEM = 'You are a streaming visual captioner. Process each video update using only visible evidence.\nUse </Silence> alone when no visual unit is ready to report.\nUse </Observe> for completed visual details, with timestamps in seconds to one decimal place: [16.0 seconds] for a moment or [0.0 - 20.0 seconds] for an interval.\nUse </Summary> with [0.0 - 20.0 seconds] for a completed semantic segment, after all its details and before any details of the next segment.\nEmit one kind of action per turn. Multiple details in one action retain their individual prefixes and timestamps.\nAt <STREAM_END>, finish pending active captions in separate turns without advancing video time.\nOnly after an explicit user question about the complete video, use </Response> Overall Summary: followed by the overall account.\nNever predict unseen content.'
CAPTION_INSTRUCTION = '<VIDEO_CONTEXT>\nContinuously observe the video, use </Observe> for completed visual details, and use </Summary> to summarize each completed semantic segment.'
MEMORY_PREAMBLE = '<LONG_TERM_VIDEO_MEMORY>\nThis memory contains only observations generated before the question time.\nTimes are relative to the beginning of the evaluated video clip.'
MEMORY_POSTAMBLE = '</LONG_TERM_VIDEO_MEMORY>'
MEMORY_MAX_TOKENS = 12000


class StreamQwen:
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, attn=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import importlib.util
        # flash attention when installed: with "sdpa" the vision encoder returned wrong features on our ROCm GPUs (the
        # model then described every video alike); the model was trained and evaluated with flash attention
        attn = attn or ("flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa")
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn).to(device).eval()
        self.device = device
        # stream(): each second's frames go through the vision tower once, not once per update while they stay in the
        # 32-second window (the encoder attends within each second's frames only, so the result is the same)
        inner, self._vkeys, self._vcache = self.model.model, None, {}
        encode = inner.get_video_features

        def cached(pixel_values_videos, video_grid_thw=None, **kw):
            if self._vkeys is None or len(self._vkeys) != len(video_grid_thw):
                return encode(pixel_values_videos, video_grid_thw, **kw)
            from transformers.modeling_outputs import BaseModelOutputWithPooling
            parts = torch.split(pixel_values_videos, video_grid_thw.prod(-1).tolist())
            new = [i for i, k in enumerate(self._vkeys) if k not in self._vcache]
            if new:
                out = encode(torch.cat([parts[i] for i in new]), video_grid_thw[new], **kw).pooler_output
                self._vcache.update({self._vkeys[i]: e for i, e in zip(new, out)})
            return BaseModelOutputWithPooling(pooler_output=tuple(self._vcache[k] for k in self._vkeys))
        inner.get_video_features = cached

    def _generate(self, inputs, max_new_tokens):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        for k in ("pixel_values", "pixel_values_videos"):
            if k in inputs: inputs[k] = inputs[k].to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.proc.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)[0]
        for end in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(end)[0]
        return text.strip()

    def ask(self, frames, question, options=None, max_new_tokens=16, memory=None):
        """frames: [(seconds, PIL image)] oldest to newest (16 recent frames, 1 per second); options: ["A. ...", ...] or None.
        memory: caption-memory events from memorize() (PHCM); None answers from the frames alone ("16 recent frames")."""
        from PIL import ImageOps
        f = self.proc.image_processor.patch_size * self.proc.image_processor.merge_size
        mem = "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>" if memory is None else self.render_memory(memory, question)
        content = [{"type": "text", "text": mem},
                   {"type": "text", "text": "<RECENT_VISUAL_CONTEXT>\nThe following frames are ordered from oldest to newest. "
                    "They retain their decoded source pixels; only minimal border padding may be added to align them to the model patch grid."}]
        images = []
        for t, im in frames:
            w, h = im.size
            im = ImageOps.expand(im, border=(0, 0, -w % f, -h % f), fill=0)
            content += [{"type": "text", "text": f"<{t:.3f}s>"}, {"type": "image", "image": im}]; images.append(im)
        q = question if options is None else "\n".join([f"Question:\n{question}", "Options:", *options, POST_PROMPT, "Best option:("])
        content.append({"type": "text", "text": "</RECENT_VISUAL_CONTEXT>\n" + q})
        chat = [{"role": "system", "content": [{"type": "text", "text": QA_SYSTEM if options else QA_DIRECT_SYSTEM}]}, {"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return self._generate(self.proc(text=[prompt], images=images, do_resize=False, return_tensors="pt"), max_new_tokens)

    def _second(self, frames, times, a, b):
        """One second of video (4 frames) as the model was trained on it: two temporal blocks with time labels."""
        import torchvision.transforms.functional as TF
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        vp_ = self.proc.video_processor; T = int(vp_.temporal_patch_size)
        frames, times = list(frames), list(times)
        while len(frames) % T: frames.append(frames[-1]); times.append(times[-1])
        vt = torch.stack([torch.from_numpy(np.array(x)).permute(2, 0, 1) for x in frames])
        f = int(self.proc.image_processor.patch_size) * int(self.proc.image_processor.merge_size)
        h, w = smart_resize(vt.shape[-2], vt.shape[-1], factor=f, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        vt = TF.resize(vt, [h, w], interpolation=TF.InterpolationMode.BICUBIC)
        vp = vp_(videos=[vt], do_sample_frames=False, do_resize=False, return_tensors="pt")
        g = vp["video_grid_thw"][0]; n = int(g[1] * g[2]) // (int(vp_.merge_size) ** 2)
        block = "<|vision_start|>" + "<|video_pad|>" * n + "<|vision_end|>"
        vis = "".join(f"<{sum(times[i:i + T]) / T:.1f} seconds>" + block for i in range(0, len(times), T)) + f"<{a:.1f}-{b:.1f} seconds>"
        return vis, vp

    @staticmethod
    def _frames(video, fps):
        """[(seconds, PIL image)] at `fps` frames per second (the first decoded frame of each 1/fps interval)."""
        import cv2
        from PIL import Image
        cap = cv2.VideoCapture(video); vfps = cap.get(cv2.CAP_PROP_FPS) or 30; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(n):
            ok, fr = cap.read()
            if not ok: break
            t = i / vfps
            if math.floor(t * fps + 1e-7) > math.floor((i - 1) / vfps * fps + 1e-7) or i == 0:
                frames.append((t, Image.fromarray(fr[:, :, ::-1])))
        cap.release()
        return frames

    def caption(self, video, every=4, span=8, fps=4, max_new_tokens=64):
        """Live captions during the stream: every `every` seconds, one sentence about the last `span` seconds of video
        (span x fps frames). Yields (second, caption); a caption equal to the previous one is skipped."""
        frames, last = self._frames(video, fps), None
        for end in range(every, int(math.ceil(frames[-1][0] + 1e-7)) + 1, every):
            sel = [(t, im) for t, im in frames if end - span <= t < end]
            vis, vp = self._second([im for _, im in sel], [t for t, _ in sel], float(max(0, end - span)), float(end))
            chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": vis + "\n" + CAPTION_Q}]
            text = self.proc.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inputs = self.proc.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs["pixel_values_videos"], inputs["video_grid_thw"] = vp["pixel_values_videos"], vp["video_grid_thw"]
            inputs["mm_token_type_ids"] = (inputs["input_ids"] == self.proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")).int() * 2
            c = self._generate(inputs, max_new_tokens)
            if c != last:
                yield end, c
            last = c

    def stream(self, video, prompt, fps=4, window=32, max_new_tokens=128, no_repeat=0):
        """Yields (second, answer) for every second of the video; the prompt is shown in the first turn.
        no_repeat=k: an answer that repeats one of the last k responses word for word is replaced by </Silence>, also in
        the context the model sees next (a small model otherwise keeps copying its own previous response)."""
        frames = self._frames(video, fps)
        turns, said = [], []; self._vcache = {}
        for s in range(int(math.ceil(frames[-1][0] + 1e-7))):
            sel = [(t, im) for t, im in frames if s <= t < s + 1]
            if not sel: continue
            vis, vp = self._second([im for _, im in sel], [t for t, _ in sel], float(s), float(s + 1))
            turns.append(dict(visual=vis, vp=vp, answer=None, key=s)); turns = turns[-window:]
            chat = [{"role": "system", "content": STREAM_SYSTEM}]
            for j, tr in enumerate(turns):
                chat.append({"role": "user", "content": (prompt + "\n" + tr["visual"]) if j == 0 else tr["visual"]})
                if tr["answer"] is not None: chat.append({"role": "assistant", "content": tr["answer"]})
            text = self.proc.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            inputs = self.proc.tokenizer(text, return_tensors="pt", add_special_tokens=False)
            inputs["pixel_values_videos"] = torch.cat([tr["vp"]["pixel_values_videos"] for tr in turns])
            inputs["video_grid_thw"] = torch.cat([tr["vp"]["video_grid_thw"] for tr in turns])
            ids = inputs["input_ids"]; vpad = self.proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            inputs["mm_token_type_ids"] = (ids == vpad).int() * 2
            self._vkeys = [tr["key"] for tr in turns]
            self._vcache = {k: v for k, v in self._vcache.items() if k in self._vkeys}
            ans = self._generate(inputs, max_new_tokens)
            self._vkeys = None
            if no_repeat and ans.startswith("</Response>"):
                if ans in said[-no_repeat:]: ans = "</Silence>"
                else: said.append(ans)
            turns[-1]["answer"] = ans
            yield s, ans

    # ---- PHCM (OneStreamer's caption memory + 16 recent frames), as the release runs it in evaluation ----------------
    # The release's code paths: onestreamer/caption_video.py (caption session), caption_optimized.py (memory pass and
    # question-time flush), memory.py (memory text with a 12,000-token budget), video.py (frame sampling), local_runner.py.

    @staticmethod
    def _rgb(frame):
        import cv2
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _probe(video):
        import cv2
        cap = cv2.VideoCapture(video)
        fps, total = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if fps <= 0 or total <= 0: raise ValueError(f"cannot read video metadata: {video}")
        return fps, total

    @classmethod
    def _stream_frames(cls, video, start, end, fps):
        """[(seconds, PIL image)] at `fps` from start to end: frame floor(t x video fps) for t = start, start + 1/fps, ..."""
        import cv2
        vfps, total = cls._probe(video)
        lo, hi = max(0, int(math.floor(start * vfps))), min(total, int(math.ceil(end * vfps)))
        idx = []
        for k in range(max(1, int(math.ceil((end - start) * fps)))):
            t = start + k / fps
            if t >= end: break
            i = min(hi - 1, max(lo, int(math.floor(t * vfps))))
            if not idx or i != idx[-1][0]: idx.append((i, t))
        cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, lo); cur, out = lo, []
        for i, t in idx:
            while cur <= i:
                ok, fr = cap.read()
                if not ok: raise RuntimeError(f"cannot decode frame {cur} of {video}")
                if cur == i: out.append((t, cls._rgb(fr)))
                cur += 1
        cap.release()
        return out

    @classmethod
    def recent_frames(cls, video, t, n=16, fps=1.0, start=0.0):
        """The n frames 1/fps apart ending at the last frame before second t, at their source resolution (oldest first)."""
        import cv2
        vfps, total = cls._probe(video)
        lo, hi = max(0, int(math.floor(start * vfps))), min(total, int(math.ceil(t * vfps)))
        idx = sorted({hi - 1 - int(round(k * vfps / fps)) for k in range(n) if hi - 1 - int(round(k * vfps / fps)) >= lo}) or [hi - 1]
        cap, out = cv2.VideoCapture(video), []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
            if not ok: raise RuntimeError(f"cannot decode frame {i} of {video}")
            out.append((i / vfps, cls._rgb(fr)))
        cap.release()
        return out

    def _caption_turn(self, history, max_new_tokens, stream_end=False):
        """One caption-stream decision over the updates in the window (the first one carries the instruction)."""
        chat = [{"role": "system", "content": CAPTION_SYSTEM}]
        for j, tr in enumerate(history):
            chat.append({"role": "user", "content": CAPTION_INSTRUCTION.replace("<VIDEO_CONTEXT>", tr["visual"]) if j == 0 else tr["visual"]})
            if tr["answer"] is not None: chat.append({"role": "assistant", "content": tr["answer"]})
        if stream_end: chat.append({"role": "user", "content": "<STREAM_END>"})
        tok = self.proc.tokenizer
        inputs = tok(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), return_tensors="pt")
        inputs["pixel_values_videos"] = torch.cat([tr["vp"]["pixel_values_videos"] for tr in history])
        inputs["video_grid_thw"] = torch.cat([tr["vp"]["video_grid_thw"] for tr in history])
        inputs["mm_token_type_ids"] = (inputs["input_ids"] == tok.convert_tokens_to_ids("<|video_pad|>")).int() * 2
        self._vkeys = [tr["key"] for tr in history]
        self._vcache = {k: v for k, v in self._vcache.items() if k in self._vkeys}
        try:
            return self._generate(inputs, max_new_tokens)
        finally:
            self._vkeys = None

    @staticmethod
    def _parse(raw):
        """(kind, text) of a caption-stream output: "silence", "observe", "summary" or "invalid"."""
        import re
        m = re.match(r"^\s*</([A-Za-z]+)>\s*(.*)$", raw or "", re.DOTALL)
        kind = {"Silence": "silence", "Observe": "observe", "Summary": "summary"}.get(m.group(1)) if m else None
        return (kind, m.group(2).strip()) if kind else ("invalid", (raw or "").strip())

    def memorize(self, video, until, start=0.0, fps=4, window=32, max_new_tokens=128):
        """PHCM caption memory, built while the video plays and before any question: one update per second (fps frames,
        the last `window` updates in context), each answered </Silence>, </Observe> + a detail or </Summary> + a summary
        of the finished segment. until: the question time, or a list of question times (one pass serves them all).
        Returns the memory events at that time (a dict time -> events for a list): dict(kind "observe" | "summary",
        start, end, text). At a question time an unfinished segment (the last update was an </Observe>) is closed by one
        <STREAM_END> turn whose </Summary> is added for that question only; the stream itself goes on unchanged.
        self.memory_outputs keeps every update's raw output."""
        times = sorted({float(x) for x in ([until] if isinstance(until, (int, float)) else until)}); end = times[-1]
        rounds = {}
        for t, im in self._stream_frames(video, start, end, fps):
            rounds.setdefault(int(math.floor(t - start + 1e-7)), []).append((t, im))
        history, events, outputs, result = [], [], [], {}
        fallback, first_obs = start, None                    # the open segment starts at its first </Observe>
        pending = iter(times); nxt = next(pending); self._vcache = {}
        for r in sorted(rounds):
            a, b = start + r, min(start + r + 1, end)
            vis, vp = self._second([im for _, im in rounds[r]], [t for t, _ in rounds[r]], a, b)
            history.append(dict(visual=vis, vp=vp, answer=None, key=r)); history = history[-window:]
            raw = self._caption_turn(history, max_new_tokens); history[-1]["answer"] = raw
            kind, text = self._parse(raw)
            if kind == "observe" and text:
                first_obs = a if first_obs is None else first_obs
                events.append(dict(kind="observe", start=a, end=b, text=text))
            elif kind == "summary" and text:
                events.append(dict(kind="summary", start=min(first_obs if first_obs is not None else fallback, b), end=b, text=text))
                fallback, first_obs = b, None
            outputs.append(dict(start=a, end=b, kind=kind, output=raw))
            while nxt is not None and b >= nxt - 1e-6:
                ev = [e for e in events if e["end"] <= nxt + 1e-6]
                last = [o for o in outputs if o["end"] <= nxt + 1e-6]
                if last and last[-1]["kind"] == "observe":
                    kind2, text2 = self._parse(self._caption_turn(history, max_new_tokens, stream_end=True))
                    if kind2 == "summary" and text2:
                        seg = max([e["end"] for e in ev if e["kind"] == "summary"] or [start])
                        seg = next((e["start"] for e in ev if e["kind"] == "observe" and e["start"] >= seg - 1e-6), seg)
                        ev = ev + [dict(kind="summary", start=seg, end=nxt, text=text2)]
                result[nxt] = ev
                nxt = next(pending, None)
        if nxt is not None:
            raise RuntimeError(f"the video ends before question time {nxt}")
        self.memory_outputs = outputs
        return result[times[0]] if isinstance(until, (int, float)) else result

    @staticmethod
    def _memory_text(events):
        t = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
        line = lambda e: f"[{t(e['start'])}s-{t(e['end'])}s] {e['text']}"
        return "\n".join([MEMORY_PREAMBLE, "<SEMANTIC_SUMMARIES>", *[line(e) for e in events if e["kind"] == "summary"], "</SEMANTIC_SUMMARIES>",
                          "<EVENT_CAPTIONS>", *[line(e) for e in events if e["kind"] == "observe"], "</EVENT_CAPTIONS>", MEMORY_POSTAMBLE])

    def render_memory(self, events, question, max_tokens=MEMORY_MAX_TOKENS):
        """The memory text the model answers from. Over max_tokens: summaries newest first, then observations ranked by
        word overlap with the question, recency and position, as many as fit, in time order."""
        import re
        n = lambda text: len(self.proc.tokenizer.encode(text, add_special_tokens=False))
        events = [e for e in events if e["kind"] in ("observe", "summary") and e["text"]]
        text = self._memory_text(events)
        if n(text) <= max_tokens:
            return text
        stop = {"a", "an", "the", "is", "are", "was", "were", "what", "where", "who", "how", "did", "do", "to", "in", "on", "of", "and", "i"}
        terms = set(re.findall(r"[a-z0-9]+", question.lower())) - stop
        caps = [e for e in events if e["kind"] == "observe"]; total = max(1, len(caps) - 1)
        score = lambda i, e: 4.0 * len(terms & set(re.findall(r"[a-z0-9]+", e["text"].lower()))) + i / total + (
            1.0 if i in {0, len(caps) - 1} or i % max(1, len(caps) // 8) == 0 else 0.0)
        ranked = [e for _, _, e in sorted(((score(i, e), i, e) for i, e in enumerate(caps)), key=lambda x: (x[0], x[1]), reverse=True)]
        chosen = []
        for e in [e for e in reversed(events) if e["kind"] == "summary"] + ranked:
            if e in chosen: continue
            trial = sorted(chosen + [e], key=lambda x: (x["end"], x["kind"]))
            if n(self._memory_text(trial)) <= max_tokens: chosen = trial
        return self._memory_text(chosen)

    def ask_phcm(self, video, t, question, options=None):
        """Answer at second t of the video with PHCM: the caption memory written up to t plus the 16 most recent frames."""
        return self.ask(self.recent_frames(video, t), question, options, memory=self.memorize(video, t))




if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--model", default="ahmedheakl/efficientstream")
    ap.add_argument("--sub", default="stream-qwen3.5-0.8b-r4", help="weights folder in the repository (run 4: PHCM-capable)")
    ap.add_argument("--prompt", default="Tell me when the person picks something up.")
    ap.add_argument("--caption", action="store_true", help="live captions: a sentence every 4 s about the last 8 s")
    ap.add_argument("--memory", action="store_true", help="print the PHCM caption memory written while the video plays")
    ap.add_argument("--question", help="answer this question at --time with PHCM (caption memory + 16 recent frames)")
    ap.add_argument("--options", nargs="+", help='multiple-choice options, e.g. "A. red" "B. blue"')
    ap.add_argument("--time", type=float, help="question time in seconds (default: the end of the video)")
    ap.add_argument("--no-memory", action="store_true", help="answer from the 16 recent frames only")
    a = ap.parse_args()
    path, sub = a.model, "stream-qwen3.5-0.8b-caption" if a.caption else a.sub
    if not __import__("os").path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(a.model, allow_patterns=[sub + "/*"]) + "/" + sub
    m = StreamQwen(path)
    if a.question or a.memory:
        fps, total = m._probe(a.video); t = a.time if a.time is not None else math.floor(total / fps)
        mem = None if a.no_memory else m.memorize(a.video, t)
        for e in mem or []:
            print(f"[{e['start']:g}-{e['end']:g}s] {e['kind']}: {e['text']}", flush=True)
        if a.question:
            print("answer:", m.ask(m.recent_frames(a.video, t), a.question, a.options, memory=mem), flush=True)
    else:
        for s, ans in (m.caption(a.video) if a.caption else m.stream(a.video, a.prompt)):
            print(f"{s:5d}s  {ans}", flush=True)
