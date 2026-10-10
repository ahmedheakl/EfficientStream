"""Streaming Qwen3.5-0.8B: Qwen3.5-0.8B fine-tuned with OneStreamer's state tokens on OneStreamer-1M records + LLaVA-Video QA.

Two ways to use it (both as in training / evaluation; transformers >= 5.0, flash-linear-attention recommended):
  ask(frames, question, options)  answer from the 16 most recent frames (OneStreamer's "recent frames" setting)
  stream(video, prompt)           one update per second of video, 4 frames per second, the last 32 seconds in context;
                                  the model answers every second with </Silence>, </Standby> or </Response> + text
"""
import math
import numpy as np
import torch

STREAM_SYSTEM = 'You are a conservative assistant for real-time streaming video event detection.\nYou receive the video incrementally, with every update labeled by an absolute time interval such as <0s-1s>. Base each decision only on evidence observed so far.\n\nEvery </Response> is a final detection of one distinct event boundary. An unmatched, premature, or duplicate response is an error, so prioritize precision over frequent responses.\n\nAt every update, use exactly one of these three protocols:\n\n1. Use </Silence> when:\n   - No relevant event is currently supported by clear visual evidence.\n   - The requested event has not started, or a suspected event is no longer supported.\n   - The same event was already reported and no unmistakably new occurrence or step has completed.\n   - The scene is unchanged, shows only aftermath, or merely repeats evidence already used for a response.\n   Output exactly </Silence> and nothing else.\n\n2. Use </Standby> only when:\n   - A specific relevant event is visibly developing, but its completion or identity still needs confirmation.\n   - There is concrete new evidence, not just generic activity or uncertainty.\n   </Standby> is reversible. Return to </Silence> if later frames do not confirm a new event.\n   Output exactly </Standby> and nothing else.\n\n3. Use </Response> only when:\n   - The visual evidence clearly establishes that a new, distinct requested event or step has just completed.\n   - All essential entities, actions, attributes, and relationships required by the question are confirmed.\n   - This event has not already been reported.\n   Output </Response> followed by a concise answer.\n\nTask-specific rules:\n- For a single-event question, respond at most once in the entire stream. After responding, always use </Silence>.\n- For "How many times" questions, respond exactly once when each genuinely new occurrence completes. Do not repeat the same count while no new occurrence has happened.\n- For step-description questions, respond exactly once when each genuinely new requested step completes. Do not repeatedly report an ongoing or already completed step.\n\nDo not predict future actions, treat a precursor as completion, or use </Response> merely because the current frames are relevant. When uncertain, prefer </Standby> or </Silence>.\n'
QA_SYSTEM = 'You answer a multiple-choice question about a video after the observation period has ended.\nYou may receive two evidence sources:\n1. A causal textual memory produced while the video was playing, without access to the current question. It may be incomplete or imperfect; use it to recover earlier events.\n2. Recent video frames from immediately before the question time, ordered from oldest to newest. Use them for the latest visual state.\n\nUse the recent frames as stronger evidence when they conflict with the textual memory. Base the answer only on the supplied memory and frames, select the single best option, and output only its option letter.'
QA_DIRECT_SYSTEM = 'You answer a question about a video after the observation period has ended.\nYou may receive two evidence sources:\n1. A causal textual memory produced while the video was playing, without access to the current question. It may be incomplete or imperfect; use it to recover earlier events.\n2. Recent video frames from immediately before the question time, ordered from oldest to newest. Use them for the latest visual state.\n\nUse the recent frames as stronger evidence when they conflict with the textual memory. Base the answer only on the supplied memory and frames. Follow the response format requested by the question exactly and output only the requested answer.'
POST_PROMPT = 'Answer with ONLY the single uppercase letter of the correct option.'
MIN_PIXELS, MAX_PIXELS = 3136, 100352


class StreamQwen:
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, attn="sdpa"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn).to(device).eval()
        self.device = device

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

    def ask(self, frames, question, options=None, max_new_tokens=16):
        """frames: [(seconds, PIL image)] oldest to newest (16 recent frames, 1 per second); options: ["A. ...", ...] or None."""
        from PIL import ImageOps
        f = self.proc.image_processor.patch_size * self.proc.image_processor.merge_size
        content = [{"type": "text", "text": "<LONG_TERM_VIDEO_MEMORY>None for this ablation.</LONG_TERM_VIDEO_MEMORY>"},
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

    def stream(self, video, prompt, fps=4, window=32, max_new_tokens=128):
        """Yields (second, answer) for every second of the video; the prompt is shown in the first turn."""
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
        turns = []
        for s in range(int(math.ceil(frames[-1][0] + 1e-7))):
            sel = [(t, im) for t, im in frames if s <= t < s + 1]
            if not sel: continue
            vis, vp = self._second([im for _, im in sel], [t for t, _ in sel], float(s), float(s + 1))
            turns.append(dict(visual=vis, vp=vp, answer=None)); turns = turns[-window:]
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
            turns[-1]["answer"] = self._generate(inputs, max_new_tokens)
            yield s, turns[-1]["answer"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--model", default="ahmedheakl/efficientstream")
    ap.add_argument("--prompt", default="Tell me when the person picks something up.")
    a = ap.parse_args()
    path = a.model
    if not __import__("os").path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(a.model, allow_patterns=["stream-qwen3.5-0.8b/*"]) + "/stream-qwen3.5-0.8b"
    m = StreamQwen(path)
    for s, ans in m.stream(a.video, a.prompt):
        print(f"{s:5d}s  {ans}")
