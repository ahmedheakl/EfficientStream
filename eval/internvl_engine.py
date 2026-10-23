"""OneStreamerEngine.answer() on InternVL3.5 (Hugging Face format), for the V4 check (results/HYBRID_RULES.md, V4).

InternVL sees an image as 448-px tiles of 256 tokens each, not as native-resolution patches, so the H1 recipe is mapped onto tiles:
a frame kept at source pixels gets the processor's dynamic tiling (up to MAX_TILES tiles + one thumbnail tile, crop_to_patches); a
frame that was resized (answer_variants.resize_frames marks it with info["resized"]) or is no larger than one tile gets one tile.
Prompts, chat structure and the system prompt are OneStreamer's, as for the Qwen models. prepare_inputs() is shared with
latency_bench.timed_answer.
"""
import torch

from onestreamer.engine import OneStreamerEngine

MAX_TILES = 4  # 16 source frames x (4 + 1) tiles x 256 tokens + 12k memory tokens stay inside the 40,960-token context


class InternVLEngine(OneStreamerEngine):

    def __init__(self, model_path, dtype="bfloat16", attn_implementation="flash_attention_2", device="cuda", **_):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=getattr(torch, dtype),
                                                                 attn_implementation=attn_implementation, device_map=device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.debug, self._debug_fp, self._debug_step, self.debug_max_text_chars = False, None, 0, 4000

    @staticmethod
    def tiled(image):
        return not image.info.get("resized") and max(image.size) > 448

    def prepare_inputs(self, chat, images):
        """(model inputs on the CPU, metrics): the chat template, each image's tiles, and the image tokens expanded per image."""
        P = self.processor
        prompt = P.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        parts = prompt.split(P.image_token)
        assert len(parts) == len(images) + 1, (len(parts), len(images))
        pv, n = [], []
        for im in images:
            o = P.image_processor(images=[im], crop_to_patches=self.tiled(im), max_patches=MAX_TILES, return_tensors="pt")
            pv.append(o["pixel_values"])
            n.append(int(o["num_patches"][0]))
        text = parts[0] + "".join(f"{P.start_image_token}{P.image_token * P.image_seq_length * k}{P.end_image_token}{p}"
                                  for k, p in zip(n, parts[1:]))
        enc = P.tokenizer([text], return_tensors="pt", add_special_tokens=False)
        inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if pv:
            inputs["pixel_values"] = torch.cat(pv)
        ids = enc["input_ids"]
        vis = int((ids == P.image_token_id).sum())
        return inputs, {"input_tokens": int(ids.shape[-1]), "visual_tokens": vis, "text_tokens": int(ids.shape[-1]) - vis,
                        "input_image_sizes": [list(im.size) for im in images], "tiles": n}

    def generate_chat(self, chat, images, max_tokens, temperature=0.0, do_resize=False):
        inputs, self.last_input_metrics = self.prepare_inputs(chat, images)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        tok = self.processor.tokenizer
        gen = dict(max_new_tokens=max_tokens, do_sample=temperature > 0.0, use_cache=True, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen.update(temperature=temperature) if temperature > 0.0 else gen.update(temperature=1.0, top_p=1.0, top_k=0)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)[0]
        return self._strip_end_tokens(text)
