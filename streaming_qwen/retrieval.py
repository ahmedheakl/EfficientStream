"""Frames kept while the video plays, for questions about earlier moments: one frame a second with its MobileCLIP2-S0
embedding and how sharp it is. retrieve() picks the frames that best match a question: the `pool` most similar, sharp
ones preferred, at least `gap` seconds apart, returned in time order and shrunk to a long side of 384."""
import json
import os

import numpy as np
import torch


class FrameIndex:
    """MobileCLIP2-S0 (open_clip format, the encoder/ folder of ahmedheakl/efficientstream): unit-length embeddings of
    frames and questions in one space."""

    def __init__(self, path, device="cuda", dtype=None):
        import open_clip
        dtype = dtype or (torch.float16 if device.startswith("cuda") else torch.float32)
        model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{path}")
        cfg = json.load(open(os.path.join(path, "open_clip_config.json")))
        self.device, self.dtype, self.preprocess = device, dtype, preprocess
        self.model = model.to(device, dtype).eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.get("model_name") or "MobileCLIP2-S0")

    def _finish(self, e):
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_images(self, images):
        px = torch.stack([self.preprocess(im) for im in images]).to(self.device, self.dtype)
        return self._finish(self.model.encode_image(px))

    @torch.no_grad()
    def embed_text(self, texts):
        return self._finish(self.model.encode_text(self.tokenizer(texts).to(self.device)))


def resize_long(im, side):
    """Shrink so the long side is at most `side`, keeping the aspect ratio; sides are multiples of 32."""
    from PIL import Image
    w, h = im.size
    k = min(1.0, side / max(w, h))
    return im.convert("RGB").resize((max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)), Image.BICUBIC)


def sharpness(im):
    """How sharp a frame is: the variance of the Laplacian of the grey frame at 320 px wide (blurry frames score low)."""
    import cv2
    g = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    return float(cv2.Laplacian(cv2.resize(g, (320, max(1, round(h * 320 / w)))), cv2.CV_64F).var())


def select_frames(times, sims, sharp, k=4, pool=32, beta=0.3, gap=3.0):
    """k frames for a question: the `pool` most similar, scored z(similarity) + beta * z(log(1 + sharpness)), taken best
    first while at least `gap` seconds from those already taken. Indices in time order."""
    times, sims, sh = np.asarray(times, float), np.asarray(sims, float), np.log1p(np.asarray(sharp, float))
    top = np.argsort(-sims, kind="stable")[:pool]
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    score = z(sims[top]) + beta * z(sh[top]) if len(top) > 1 else sims[top]
    chosen = []
    for i in np.argsort(-score, kind="stable"):
        if all(abs(times[top[i]] - times[top[j]]) >= gap for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return sorted((int(top[i]) for i in chosen), key=lambda j: times[j])


class FrameMemory:
    def __init__(self, index, side=384):
        self.index, self.side, self.frames = index, side, []      # (time, image, embedding, sharpness)

    def add(self, t, im):
        """Keeps the frame of second t (call once a second)."""
        self.frames.append((float(t), resize_long(im, self.side), self.index.embed_images([im])[0], sharpness(im)))

    def retrieve(self, question, before, k=4):
        """The k kept frames before second `before` that best match `question`: [(time, image)], oldest first."""
        cand = [f for f in self.frames if f[0] < before - 1e-6]
        if not cand:
            return []
        q = self.index.embed_text([question])[0]
        sims = np.stack([f[2] for f in cand]).astype(np.float32) @ q.astype(np.float32)
        return [(cand[i][0], cand[i][1]) for i in select_frames([f[0] for f in cand], sims, [f[3] for f in cand], k=k)]
