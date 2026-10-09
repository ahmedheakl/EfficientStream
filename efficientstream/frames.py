"""Image handling and the bounded frame store. Pure Python + Pillow + NumPy: nothing here needs a GPU."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def to_pil(frame) -> Image.Image:
    """A PIL image (RGB) from a PIL image or an H x W x 3 uint8 RGB array."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected an RGB uint8 array of shape H x W x 3, got {arr.shape} {arr.dtype}")
    return Image.fromarray(arr, "RGB")


def jpeg_roundtrip(im: Image.Image, quality: int | None) -> Image.Image:
    if quality is None:
        return im
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def grid_size(width: int, height: int, max_pixels: int, factor: int = 32) -> tuple[int, int]:
    """Largest multiple-of-`factor` size under `max_pixels`, aspect ratio kept (never upscales)."""
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    w = max(factor, int(round(width * scale / factor)) * factor)
    h = max(factor, int(round(height * scale / factor)) * factor)
    while w * h > max_pixels:
        if w >= h:
            w -= factor
        else:
            h -= factor
    return w, h


def letterbox(im: Image.Image, side: int) -> Image.Image:
    """Fit into a side x side black square, centred, aspect ratio kept."""
    if im.size == (side, side):
        return im
    s = side / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    sq = Image.new("RGB", (side, side), (0, 0, 0))
    sq.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return sq


def fit_long_side(im: Image.Image, side: int) -> Image.Image:
    """Aspect ratio kept, long side = `side`."""
    if max(im.size) == side:
        return im
    s = side / max(im.size)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)


def format_time(t: float, style: str = "hms") -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}" if style == "hms" else f"{t}s"


@dataclass
class StoredFrame:
    time: float
    image: Image.Image          # the stored (grid-size) frame
    embedding: np.ndarray       # unit-norm MobileCLIP2-S0 image embedding, float32


@dataclass
class BoundedStore:
    """One-pass bounded store of past frames.

    Frames arrive in time order. When the store holds more than `budget` frames, one frame is evicted: the one whose
    removal leaves the smallest gap between its neighbours (the densest region). The first and the newest frame are
    never evicted, so the store always spans the whole stream at a resolution that coarsens as the stream grows.
    """
    budget: int
    frames: list[StoredFrame] = field(default_factory=list)

    def add(self, frame: StoredFrame) -> StoredFrame | None:
        """Append a frame; return the evicted frame, if any."""
        if self.frames and frame.time < self.frames[-1].time:
            raise ValueError(f"frames must arrive in time order ({frame.time} < {self.frames[-1].time})")
        self.frames.append(frame)
        if len(self.frames) <= self.budget:
            return None
        hi = len(self.frames) - 1                  # the newest frame is not evictable
        if hi <= 1:
            return self.frames.pop(1)
        t = [f.time for f in self.frames]
        gaps = [t[j + 1] - t[j - 1] for j in range(1, hi)]
        return self.frames.pop(1 + int(np.argmin(gaps)))

    def before(self, t: float) -> list[StoredFrame]:
        """Stored frames strictly older than `t` (the latest frame is shown separately)."""
        return [f for f in self.frames if f.time < t - 1e-6]

    def __len__(self) -> int:
        return len(self.frames)
