"""Decoding helpers for files (a phone app feeds camera frames to StreamSession.add_frame instead)."""
from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_frames(path: str, fps: float = 2.0, end: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """(time in seconds, RGB uint8 frame) at about `fps` frames per second, in time order, up to `end`."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, n, nxt = 1.0 / fps, 0, 0.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = n / src
            n += 1
            if end is not None and t > end + 1e-6:
                break
            if t + 1e-6 >= nxt:
                yield t, bgr[:, :, ::-1].copy()
                nxt += step
    finally:
        cap.release()
