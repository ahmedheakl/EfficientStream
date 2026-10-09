"""The frame index: MobileCLIP2-S0 (open_clip format) embeds every stored frame once, when it enters the store, and the
question once, when it is asked. Retrieval is a dot product over at most `store_budget` unit vectors."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image


class FrameIndex:
    def __init__(self, path: str, device: str = "cuda", dtype: torch.dtype = torch.float16, round_fp16: bool = True):
        import open_clip

        self.device, self.dtype, self.round_fp16 = device, dtype, round_fp16
        model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{path}")
        cfg = json.load(open(os.path.join(path, "open_clip_config.json")))
        name = cfg.get("model_name") or "MobileCLIP2-S0"
        # MobileCLIP2-S0 uses its own preprocessing (resize 256, centre crop 256, mean 0 / std 1): it is in the config
        self.preprocess = preprocess
        self.model = model.to(device, dtype).eval()
        self.tokenizer = open_clip.get_tokenizer(name)

    def _finish(self, e: torch.Tensor) -> np.ndarray:
        e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        return e.astype(np.float16).astype(np.float32) if self.round_fp16 else e

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        px = torch.stack([self.preprocess(im) for im in images]).to(self.device, self.dtype)
        return self._finish(self.model.encode_image(px))

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        return self._finish(self.model.encode_text(self.tokenizer(texts).to(self.device)))


def top_k_in_time_order(frame_embs: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    """Indices of the k rows most similar to `query` (stable on ties), returned in ascending (time) order."""
    if len(frame_embs) <= k:
        return list(range(len(frame_embs)))
    sims = frame_embs.astype(np.float32) @ query.astype(np.float32)
    return sorted(int(i) for i in np.argsort(-sims, kind="stable")[:k])
