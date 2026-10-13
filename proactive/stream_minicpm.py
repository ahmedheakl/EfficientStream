"""Streaming MiniCPM-V-4.6: MiniCPM-V-4.6 fine-tuned with OneStreamer's state tokens on OneStreamer-1M records + LLaVA-Video QA
(the same data and schedule as the streaming Qwen3.5-0.8B of stream_qwen.py, so the two compare directly).

Same calls as stream_qwen.py (transformers >= 5.17 for MiniCPM-V-4.6; flash-linear-attention recommended on GPU):
  ask(frames, question, options)  answer from the 16 most recent frames (OneStreamer's "recent frames" setting)
  stream(video, prompt)           one update per second of video, 4 frames per second, the last 32 seconds in context;
                                  the model answers every second with </Silence>, </Standby> or </Response> + text
  caption(video)                  every 4 seconds, one sentence about the last 8 seconds (16 frames)
The model sees each frame as one 448-pixel image (no slices), 66 tokens for a 16:9 frame, after a time label, exactly as
in training.
"""
import math
import torch
from stream_qwen import STREAM_SYSTEM, QA_SYSTEM, QA_DIRECT_SYSTEM, POST_PROMPT, CAPTION_Q, StreamQwen

IMG_KW = dict(slice_mode=False, scale_resolution=448, use_image_id=False)


class StreamMiniCPM:
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, attn=None, cache=True):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import importlib.util
        attn = attn or ("flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa")
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn).to(device).eval()
        self.device, self.ph = device, self.proc.image_token
        tok = self.proc.tokenizer
        self.eos = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
        # stream(): each frame goes through the vision tower once, not once per update while it stays in the 32-second
        # window (3.7x faster). The tower attends within each image only: in float32 the cached features equal the
        # uncached ones to 2e-5; in bfloat16 they differ by a few percent, as with any change of how many frames are
        # encoded together, so an answer near a tie can differ (27 of 30 updates identical in our test). cache=False
        # encodes the whole window every update.
        inner, self._keys, self._cache = self.model.model, None, {}
        if not cache:
            return
        encode = inner.get_image_features

        def cached(pixel_values, target_sizes, downsample_mode=None):
            if self._keys is None or len(self._keys) != len(target_sizes):
                return encode(pixel_values, target_sizes, downsample_mode=downsample_mode)
            from transformers.modeling_outputs import BaseModelOutputWithPooling
            parts = torch.split(pixel_values, (target_sizes[:, 0] * target_sizes[:, 1] * pixel_values.shape[2]).tolist(), dim=-1)
            new = [i for i, k in enumerate(self._keys) if k not in self._cache]
            if new:
                out = encode(torch.cat([parts[i] for i in new], dim=-1), target_sizes[new], downsample_mode=downsample_mode).pooler_output
                self._cache.update({self._keys[i]: e for i, e in zip(new, out)})
            return BaseModelOutputWithPooling(pooler_output=[self._cache[k] for k in self._keys])
        inner.get_image_features = cached

    def _generate(self, chat, images, max_new_tokens):
        """chat: [(role, text)] with one image placeholder per image, in order; laid out as in training."""
        text = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in chat) + "<|im_start|>assistant\n"
        o = self.proc(text=[text], images=[images] if images else None, add_special_tokens=False, return_tensors="pt", **IMG_KW)
        o = {k: v.to(self.device) for k, v in o.items()}
        if "pixel_values" in o: o["pixel_values"] = o["pixel_values"].to(self.model.dtype)
        with torch.inference_mode():
            out = self.model.generate(**o, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=self.eos,
                                      pad_token_id=self.proc.tokenizer.pad_token_id or self.eos[0])
        text = self.proc.tokenizer.decode(out[0, o["input_ids"].shape[-1]:], skip_special_tokens=False)
        for end in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(end)[0]
        return text.strip()

    def ask(self, frames, question, options=None, max_new_tokens=16):
        """frames: [(seconds, PIL image)] oldest to newest (16 recent frames, 1 per second); options: ["A. ...", ...] or None."""
        q = question if options is None else "\n".join([f"Question:\n{question}", "Options:", *options, POST_PROMPT, "Best option:("])
        user = ("<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY><RECENT_VISUAL_CONTEXT>\nThe following frames are "
                "ordered from oldest to newest. They retain their decoded source pixels; only minimal border padding may be added to align "
                "them to the model patch grid." + "".join(f"<{t:.3f}s>{self.ph}" for t, _ in frames) + "</RECENT_VISUAL_CONTEXT>\n" + q)
        return self._generate([("system", QA_SYSTEM if options else QA_DIRECT_SYSTEM), ("user", user)], [im for _, im in frames], max_new_tokens)

    def caption(self, video, every=4, span=8, n=16, max_new_tokens=64):
        """Live captions during the stream: every `every` seconds, one sentence about the last `span` seconds (n frames).
        Yields (second, caption); a caption equal to the previous one is skipped."""
        frames, last = StreamQwen._frames(video, n / span), None
        for end in range(every, int(math.ceil(frames[-1][0] + 1e-7)) + 1, every):
            sel = [(t, im) for t, im in frames if end - span <= t < end][-n:]
            vis = "".join(f"<{t:.1f} seconds>{self.ph}" for t, _ in sel)
            c = self._generate([("system", "You are a helpful assistant."), ("user", vis + "\n" + CAPTION_Q)], [im for _, im in sel], max_new_tokens)
            if c != last:
                yield end, c
            last = c

    def stream(self, video, prompt, fps=4, window=32, max_new_tokens=128, no_repeat=0):
        """Yields (second, answer) for every second of the video; the prompt is shown in the first turn.
        no_repeat=k: an answer that repeats one of the last k responses word for word is replaced by </Silence>."""
        frames = StreamQwen._frames(video, fps)
        turns, said = [], []; self._cache = {}
        for s in range(int(math.ceil(frames[-1][0] + 1e-7))):
            sel = [(t, im) for t, im in frames if s <= t < s + 1]
            if not sel: continue
            vis = "".join(f"<{t:.1f} seconds>{self.ph}" for t, _ in sel) + f"<{float(s):.1f}-{float(s + 1):.1f} seconds>"
            turns.append(dict(visual=vis, images=[im for _, im in sel], keys=[(s, i) for i in range(len(sel))], answer=None))
            turns = turns[-window:]
            chat = [("system", STREAM_SYSTEM)]
            for j, tr in enumerate(turns):
                chat.append(("user", (prompt + "\n" + tr["visual"]) if j == 0 else tr["visual"]))
                if tr["answer"] is not None: chat.append(("assistant", tr["answer"]))
            self._keys = [k for tr in turns for k in tr["keys"]]
            self._cache = {k: v for k, v in self._cache.items() if k in self._keys}
            ans = self._generate(chat, [im for tr in turns for im in tr["images"]], max_new_tokens)
            self._keys = None
            if no_repeat and ans.startswith("</Response>"):
                if ans in said[-no_repeat:]: ans = "</Silence>"
                else: said.append(ans)
            turns[-1]["answer"] = ans
            yield s, ans


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--model", default="ahmedheakl/efficientstream")
    ap.add_argument("--prompt", default="Tell me when the person picks something up.")
    ap.add_argument("--caption", action="store_true", help="live captions: a sentence every 4 s about the last 8 s")
    a = ap.parse_args()
    path, sub = a.model, "stream-minicpm-v4.6"
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(a.model, allow_patterns=[sub + "/*"]) + "/" + sub
    m = StreamMiniCPM(path)
    for s, ans in (m.caption(a.video) if a.caption else m.stream(a.video, a.prompt)):
        print(f"{s:5d}s  {ans}", flush=True)
