"""The streaming session: feed frames as they arrive, ask questions at any moment.

    es = EfficientStream.from_pretrained("ahmedheakl/efficientstream")
    session = es.session()
    for t, frame in frames:                      # decoded frames in time order, any rate >= 0.5 fps
        session.add_frame(t, frame)
        if t == 42.0:
            print(session.ask("What is on the table?", ["A. a cup", "B. a phone", "C. keys", "D. nothing"]).text)

Per frame, only every 2 s does any work happen: the frame is resized, stored, and embedded once by MobileCLIP2-S0.
At question time: one text embedding, a top-16 lookup over at most 83 vectors, and one forward pass of MiniCPM-V-4.6
over about 1,700 tokens (multiple choice; open-ended answers then decode greedily)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

from PIL import Image

from .config import StreamConfig
from .frames import (BoundedStore, StoredFrame, fit_long_side, format_time, grid_size, jpeg_roundtrip, letterbox,
                     to_pil)
from .index import FrameIndex, top_k_in_time_order
from .model import Answer, Answerer
from .prompt import question_block


@dataclass
class Result:
    text: str                       # option letter or open-ended answer
    probs: list[float] | None
    frame_times: list[float]        # times of the thumbnails shown (the latest frame is `now`)
    now: float
    prompt_tokens: int
    seconds: float                  # question time -> answer (excludes video decoding, which happened before)


class EfficientStream:
    """The two models: the answerer (MiniCPM-V-4.6) and the frame index (MobileCLIP2-S0)."""

    def __init__(self, backbone_dir: str, encoder_dir: str, config: StreamConfig | None = None, device: str = "cuda"):
        import torch

        self.config = config or StreamConfig()
        self.device = device
        self.answerer = Answerer(backbone_dir, device=device, dtype=self.config.dtype)
        self.index = FrameIndex(encoder_dir, device=device, dtype=torch.float16 if device != "cpu" else torch.float32,
                                round_fp16=self.config.round_embeddings_fp16)

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: str = "cuda", config: StreamConfig | None = None, **hub_kwargs):
        """A local folder or a Hugging Face repo with `backbone/`, `encoder/` and `efficientstream_config.json`."""
        root = repo_or_dir
        if not os.path.isdir(root):
            from huggingface_hub import snapshot_download
            root = snapshot_download(repo_or_dir, **hub_kwargs)
        if config is None and os.path.exists(os.path.join(root, "efficientstream_config.json")):
            config = StreamConfig(**json.load(open(os.path.join(root, "efficientstream_config.json"))))
        return cls(os.path.join(root, "backbone"), os.path.join(root, "encoder"), config, device)

    def session(self) -> "StreamSession":
        return StreamSession(self)


class StreamSession:
    def __init__(self, es: EfficientStream):
        self.es, self.cfg = es, es.config
        self.store = BoundedStore(self.cfg.store_budget)
        self.latest: tuple[float, Image.Image] | None = None
        self._next_grid = 0.0

    # ------------------------------------------------------------------------------------------- the stream
    def add_frame(self, t: float, frame) -> None:
        """A decoded frame (PIL image or H x W x 3 RGB uint8 array) at time `t` seconds. Frames must come in time
        order. Every frame becomes the latest frame; one frame every 1 / grid_fps seconds also enters the store."""
        im = to_pil(frame)
        self.set_latest(t, im)
        if t + 1e-6 >= self._next_grid:
            self.add_store_frame(t, im)
            while self._next_grid <= t + 1e-6:
                self._next_grid += 1.0 / self.cfg.grid_fps

    def set_latest(self, t: float, frame) -> None:
        """Only the latest frame: shown at high resolution as "(now)" when a question is asked."""
        self.latest = (float(t), to_pil(frame))

    def add_store_frame(self, t: float, frame, resize: bool = True) -> None:
        """Only the store: resize (unless `resize` is False, for frames already at store size), embed, store.
        add_frame calls this; call it directly to replay a pre-sampled frame store."""
        im = to_pil(frame)
        if resize:
            im = im.resize(grid_size(im.width, im.height, self.cfg.grid_max_pixels), Image.BICUBIC)
            im = jpeg_roundtrip(im, self.cfg.store_jpeg_quality)
        emb = self.es.index.embed_images([im])[0]
        self.store.add(StoredFrame(float(t), im, emb))

    # ------------------------------------------------------------------------------------------- questions
    def ask(self, question: str, options: list[str] | None = None) -> Result:
        """Answer at the current moment (the time of the latest frame). `options` = ["A. ...", "B. ...", ...] for
        multiple choice (the answer is the letter), None for an open-ended answer."""
        if self.latest is None:
            raise RuntimeError("no frame yet: call add_frame first")
        import numpy as np

        start = time.perf_counter()
        cfg, (now, latest) = self.cfg, self.latest
        cand = self.store.before(now)
        if cand and cfg.retrieve_k > 0:
            q = self.es.index.embed_text([question])[0]
            picked = [cand[i] for i in top_k_in_time_order(np.stack([f.embedding for f in cand]), q, cfg.retrieve_k)]
        else:
            picked = []
        parts, images = [], []
        for f in picked:
            parts += [{"type": "text", "text": f"frame time[{format_time(f.time, cfg.timestamps)}]: "}, {"type": "image"},
                      {"type": "text", "text": "\n"}]
            images.append((letterbox(f.image, cfg.thumb_side), cfg.thumb_side, 1))
        parts += [{"type": "text", "text": f"frame time[{format_time(now, cfg.timestamps)}] (now): "}, {"type": "image"},
                  {"type": "text", "text": "\n"}]
        hi = jpeg_roundtrip(fit_long_side(latest, cfg.latest_long_side), cfg.latest_jpeg_quality)
        images.append((hi, cfg.thumb_side, cfg.latest_max_slices))
        parts.append({"type": "text", "text": question_block(question, options)})
        a: Answer = self.es.answerer.answer(parts, images, len(options or []), cfg.max_new_tokens)
        if self.es.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        return Result(a.text, a.probs, [f.time for f in picked], now, a.prompt_tokens, time.perf_counter() - start)

    def state(self) -> dict:
        return dict(stored=len(self.store), store_times=[f.time for f in self.store.frames],
                    now=None if self.latest is None else self.latest[0], config=asdict(self.cfg))
