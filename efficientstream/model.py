"""MiniCPM-V-4.6 as the answering model: one forward pass over the mixed-scale view (thumbnails at 448 px, one vision
view each; the latest frame at up to 9 views), the assistant turn forced to "<answer>".

Multiple choice: the option letter is read from the next-token logits at the end of the prefill (no generation).
Open-ended: greedy decoding after "<answer>" until "</answer>".

The images are processed one by one, each with its own resolution and slice budget; the stock processor applies one
setting to all images of a prompt, so the expansion of the image placeholders is done here with its own helpers."""
from __future__ import annotations

import collections
from dataclasses import dataclass

import torch
from PIL import Image

from .prompt import ANSWER_CLOSE, ANSWER_OPEN, SYSTEM

PATCH = 14
DOWNSAMPLE = "16x"


@dataclass
class Answer:
    text: str                       # the answer: the option letter (multiple choice) or the generated text
    probs: list[float] | None       # multiple choice: softmax over the option letters
    prompt_tokens: int
    visual_tokens: int


class Answerer:
    def __init__(self, path: str, device: str = "cuda", dtype: str = "bfloat16"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.proc = AutoProcessor.from_pretrained(path)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=getattr(torch, dtype)).to(device).eval()
        self.lm = self.model.model.language_model
        self.embed = self.model.get_input_embeddings()
        self.image_processor = self.proc.image_processor
        self.image_token_id = self.proc.image_token_id
        self.forced = self.tok(ANSWER_OPEN, add_special_tokens=False)["input_ids"]
        self.letters = []
        for i in range(10):
            ids = self.tok(chr(65 + i), add_special_tokens=False)["input_ids"]
            assert len(ids) == 1, f"option letter {chr(65 + i)} is not one token: {ids}"
            self.letters.append(ids[0])
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}

    # ------------------------------------------------------------------------------------------------- prompt
    def build_prompt(self, parts: list[dict]) -> str:
        """`parts` = the user turn as content parts: {"type": "text", "text": ...} or {"type": "image"}."""
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": parts}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def encode(self, prompt: str, images: list[tuple[Image.Image, int, int]]):
        """Token ids (with the forced "<answer>") and the vision views [(pixels, (rows, cols))].
        `images` = [(image, scale_resolution, max_slice_nums)] in prompt order."""
        reps, views = [], []
        for im, scale, slices in images:
            o = self.image_processor(images=[im], max_slice_nums=slices, scale_resolution=scale,
                                     downsample_mode=DOWNSAMPLE, return_tensors="pt")
            reps.append(self.proc.replace_image_token(o, image_idx=0, image_token_divisor=16))
            px, off = o["pixel_values"][0], 0
            for h, w in o["target_sizes"].tolist():
                views.append((px[..., off:off + h * w * PATCH], (h, w)))
                off += h * w * PATCH
            assert off == px.shape[-1]
        assert prompt.count(self.proc.image_token) == len(images), "one image placeholder per image"
        text = [prompt]
        text, _ = self.proc.get_text_with_replacements(text, self.proc._prepend_local_ids(text, reps, self.proc.image_token), [])
        ids = self.tok(text[0])["input_ids"] + self.forced
        return ids, views

    # ----------------------------------------------------------------------------------------------- forward
    def _embed(self, ids: list[int], views):
        ids_t = torch.tensor([ids], device=self.device)
        emb = self.embed(ids_t)
        groups = collections.OrderedDict()
        for i, (_, hw) in enumerate(views):
            groups.setdefault(hw, []).append(i)
        feats = [None] * len(views)
        for hw, ix in groups.items():           # one vision call per patch grid
            pv = torch.cat([views[i][0] for i in ix], -1).unsqueeze(0).to(self.device)
            sizes = torch.tensor([hw] * len(ix), dtype=torch.int32, device=self.device)
            out = self.model.model.get_image_features(pv, sizes, downsample_mode=DOWNSAMPLE).pooler_output
            for i, f in zip(ix, out):
                feats[i] = f
        feats = torch.cat(feats, 0).to(device=emb.device, dtype=emb.dtype)
        mask = ids_t == self.image_token_id
        assert int(mask.sum()) == feats.shape[0], f"{int(mask.sum())} image tokens, {feats.shape[0]} features"
        return emb.masked_scatter(mask.unsqueeze(-1).expand_as(emb), feats), int(mask.sum())

    @torch.no_grad()
    def answer(self, parts, images, n_options: int, max_new_tokens: int = 128) -> Answer:
        ids, views = self.encode(self.build_prompt(parts), images)
        emb, n_vis = self._embed(ids, views)
        am = torch.ones(1, emb.shape[1], dtype=torch.long, device=self.device)
        # keep use_cache=True: without a cache the model's gated-delta layers take another kernel path and the
        # logits change
        out = self.lm(inputs_embeds=emb, attention_mask=am, use_cache=True)
        h = out.last_hidden_state
        if n_options:
            logits = self.model.lm_head(h[0, -1]).float()
            probs = torch.softmax(logits[self.letters[:n_options]], 0).tolist()
            letter = chr(65 + max(range(n_options), key=lambda i: probs[i]))
            return Answer(letter, probs, len(ids), n_vis)
        past, gen = out.past_key_values, []
        nxt = int(self.model.lm_head(h[0, -1]).argmax())
        while nxt not in self.eos and len(gen) < max_new_tokens:
            gen.append(nxt)
            if ANSWER_CLOSE in self.tok.decode(gen[-6:]):
                break
            am = torch.cat([am, am[:, :1]], 1)
            o = self.lm(inputs_embeds=self.embed(torch.tensor([[nxt]], device=self.device)), attention_mask=am,
                        past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(self.model.lm_head(o.last_hidden_state[0, -1]).argmax())
        text = self.tok.decode(gen, skip_special_tokens=True).split(ANSWER_CLOSE)[0].strip()
        return Answer(text, None, len(ids), n_vis)
