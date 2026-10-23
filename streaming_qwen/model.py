"""The streaming Qwen3.5-0.8B on transformers (>= 5.19): loading, one second of video through the vision tower, a
retrieved frame through it, and a prefill on a KV cache from a given M-RoPE position. Plus the cache operations the
stream needs: fork (copy-on-write) and evict (cut tokens out of the attention layers only).

The model is hybrid: of its 24 layers, every 4th is full attention (a KV cache that grows with every token), the other
18 are gated DeltaNet layers (a fixed-size conv window and recurrent state per layer, updated in place). transformers'
DynamicCache holds both, and from 5.19 on its gated DeltaNet continues the cached state over a multi-token append (each
second, and each question, is one).
"""
import copy
import importlib.util
import time
from dataclasses import dataclass

import numpy as np
import torch

from .prompts import MAX_PIXELS, MIN_PIXELS, TAGS


def fork(cache):
    """A copy of the cache that the original's later updates leave alone: attention KV shared (an append concatenates
    into new tensors, an eviction slices), the gated-delta layers' conv and recurrent states copied (they are updated
    in place)."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        c = copy.copy(layer)
        for name in ("conv_states", "recurrent_states"):
            v = getattr(layer, name, None)
            if isinstance(v, dict):
                setattr(c, name, {k: t.clone() if torch.is_tensor(t) else t for k, t in v.items()})
            elif torch.is_tensor(v):
                setattr(c, name, v.clone())
        for name in ("is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state",
                     "conv_kernel_size"):
            if isinstance(getattr(layer, name, None), dict):
                setattr(c, name, dict(getattr(layer, name)))
        new.layers.append(c)
    return new


def evict(cache, attn_layers, a, b):
    """Drops cache tokens [a, b) from the attention layers (later ones move down; their positions stay as they were).
    The gated-delta layers are left alone: their state is a running sum that cannot give tokens back."""
    for i in attn_layers:
        layer = cache.layers[i]
        layer.keys = torch.cat([layer.keys[..., :a, :], layer.keys[..., b:, :]], dim=-2)
        layer.values = torch.cat([layer.values[..., :a, :], layer.values[..., b:, :]], dim=-2)


def mrope(ids, grids, start, pads):
    """M-RoPE positions [3, len(ids)] from `start`, and the next free position. Text: the same position on all three
    axes, one per token. A run of vision pads (`pads`: the video and image pad ids) of rows x cols tokens: time = its first
    position, row and column added; the next text starts max(rows, cols) later. grids: (rows, cols) per run, in order."""
    t, h, w, p, gi, i = [], [], [], start, 0, 0
    while i < len(ids):
        if ids[i] in pads:
            rows, cols = grids[gi]
            gi += 1
            for r in range(rows):
                for c in range(cols):
                    t.append(p); h.append(p + r); w.append(p + c)
            i += rows * cols
            p += max(rows, cols)
        else:
            t.append(p); h.append(p); w.append(p)
            i += 1
            p += 1
    assert gi == len(grids), f"{gi} vision runs in the text for {len(grids)} grids"
    return torch.tensor([t, h, w]), p


@dataclass
class Second:
    """One second of video through the vision tower: the turn's text (video pads expanded), its 2-frame blocks' grids
    (rows, cols) and features [tokens, hidden]."""
    second: int
    visual: str
    grids: list
    feats: torch.Tensor
    vision: float = 0.0


@dataclass
class Still:
    """A retrieved frame through the vision tower: its time, the text that shows it (image pads expanded), its grid and
    features."""
    time: float
    visual: str
    grid: tuple
    feats: torch.Tensor


class Model:
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, attn=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        attn = attn or ("flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa")
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn).to(device).eval()
        self.device, self.lm, self.head = device, self.model.model.language_model, self.model.lm_head
        tok = self.tok = self.proc.tokenizer
        self.video_pad, self.image_pad = (tok.convert_tokens_to_ids(t) for t in ("<|video_pad|>", "<|image_pad|>"))
        self.eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        self.silence = tok.convert_tokens_to_ids("</Silence>")
        self.tags = [tok.convert_tokens_to_ids(t) for t in TAGS]
        self.attn_layers = [i for i, t in enumerate(self.lm.config.layer_types) if t == "full_attention"]
        self.merge = int(self.proc.video_processor.merge_size)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        text = self.tok.decode(ids, skip_special_tokens=False)
        for end in ("<|im_end|>", "<|endoftext|>"):
            text = text.split(end)[0]
        return text.strip()

    def new_cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.lm.config)

    def sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device.startswith("mps"):
            torch.mps.synchronize()

    @torch.inference_mode()
    def second(self, frames, times, a):
        """One second of video (its PIL frames and their times) as the model was trained on it: frames resized
        (smart_resize within MIN/MAX_PIXELS), grouped into 2-frame temporal blocks (the last frame repeated to fill a
        block), each block with its time label, then the second's range."""
        import torchvision.transforms.functional as TF
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        vp = self.proc.video_processor
        T = int(vp.temporal_patch_size)
        frames, times = list(frames), list(times)
        while len(frames) % T:
            frames.append(frames[-1]); times.append(times[-1])
        vt = torch.stack([torch.from_numpy(np.array(x.convert("RGB"))).permute(2, 0, 1) for x in frames])
        f = int(self.proc.image_processor.patch_size) * self.merge
        h, w = smart_resize(vt.shape[-2], vt.shape[-1], factor=f, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        vt = TF.resize(vt, [h, w], interpolation=TF.InterpolationMode.BICUBIC)
        out = vp(videos=[vt], do_sample_frames=False, do_resize=False, return_tensors="pt")
        g = out["video_grid_thw"]
        rows, cols = int(g[0, 1]) // self.merge, int(g[0, 2]) // self.merge
        self.sync(); t0 = time.time()
        feats = self.model.model.get_video_features(out["pixel_values_videos"].to(self.device, self.model.dtype),
                                                    g.to(self.device)).pooler_output
        feats = torch.cat(list(feats)); self.sync()
        block = "<|vision_start|>" + "<|video_pad|>" * (rows * cols) + "<|vision_end|>"
        visual = "".join(f"<{sum(times[i:i + T]) / T:.1f} seconds>" + block for i in range(0, len(times), T))
        visual += f"<{a:.1f}-{a + 1:.1f} seconds>"
        return Second(a, visual, [(rows, cols)] * (len(frames) // T), feats, time.time() - t0)

    @torch.inference_mode()
    def still(self, image, t):
        """A retrieved frame (a PIL image, both sides multiples of 32) through the vision tower, at its decoded pixels."""
        out = self.proc.image_processor(images=[image.convert("RGB")], do_resize=False, return_tensors="pt")
        g = out["image_grid_thw"]
        rows, cols = int(g[0, 1]) // self.merge, int(g[0, 2]) // self.merge
        feats = self.model.model.get_image_features(out["pixel_values"].to(self.device, self.model.dtype),
                                                    g.to(self.device)).pooler_output
        visual = f"<{t:.3f}s><|vision_start|>" + "<|image_pad|>" * (rows * cols) + "<|vision_end|>"
        return Still(t, visual, (rows, cols), torch.cat(list(feats)))

    @torch.inference_mode()
    def forward(self, ids, cache, position, grids=(), feats=None):
        """Prefills `ids` on `cache` from M-RoPE position `position`: the last token's logits and the next position.
        Vision pads take `feats` in order. The attention mask works in cache indices (the cache's length onwards), the
        rotary embedding in positions, so the two can differ after an eviction."""
        pos3, nxt = mrope(ids, list(grids), position, (self.video_pad, self.image_pad))
        x = torch.tensor(ids, device=self.device)
        emb = self.lm.embed_tokens(x)
        if feats is not None:
            emb[(x == self.video_pad) | (x == self.image_pad)] = feats.to(emb.dtype)
        past = cache.get_seq_length()
        index = torch.arange(past, past + len(ids))
        position_ids = torch.cat([index[None], pos3]).unsqueeze(1).to(self.device)   # [4, 1, n]: cache index; t, h, w
        out = self.lm(inputs_embeds=emb[None], position_ids=position_ids, past_key_values=cache, use_cache=True)
        return self.head(out.last_hidden_state[:, -1]).float()[0], nxt
