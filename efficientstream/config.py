"""Settings of the efficient streaming pipeline. The defaults are the evaluated configuration (row N4 of the Setup-3
study): MiniCPM-V-4.6, no captions, a MobileCLIP2-S0 frame index, the latest frame at 896 px + 16 retrieved thumbnails."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamConfig:
    # --- the frame store (runs while the video streams; no language model involved) -----------------------------
    grid_fps: float = 0.5
    """Frames per second admitted to the store (one every 2 s)."""
    grid_max_pixels: int = 256 * 32 * 32
    """A stored frame is resized to the largest multiple-of-32 size under this many pixels (aspect ratio kept)."""
    store_jpeg_quality: int | None = 88
    """Stored frames take a JPEG round trip at this quality, as in the evaluated store. None = keep raw pixels."""
    store_budget: int = 83
    """Bounded store: at most this many frames. On overflow the frame in the densest region is evicted, so the store
    stays spread over the whole stream (64 frames of `grid_max_pixels` = 83 thumbnails of 448 x 448)."""

    # --- the view built at question time -------------------------------------------------------------------------
    retrieve_k: int = 16
    """Thumbnails shown: the stored frames most similar to the question (MobileCLIP2-S0 cosine), in time order."""
    thumb_side: int = 448
    """Thumbnails are letterboxed to a square of this side (one vision view, 64 visual tokens each)."""
    latest_long_side: int = 896
    """The latest frame keeps its aspect ratio with this long side (up to `latest_max_slices` views)."""
    latest_max_slices: int = 9
    latest_jpeg_quality: int | None = 88
    """The latest frame takes a JPEG round trip at this quality, as in the evaluated pipeline. None = raw pixels."""
    timestamps: str = "hms"
    """Frame time stamps in the prompt: "hms" (00:06:57) or "s" (417s)."""

    # --- answering ----------------------------------------------------------------------------------------------
    max_new_tokens: int = 128
    """Open-ended answers: greedy tokens after the forced "<answer>" (stops at "</answer>")."""
    round_embeddings_fp16: bool = True
    """Frame and question embeddings are rounded to float16 before the similarity, as in the evaluated index."""
    dtype: str = "bfloat16"
    """Compute dtype of the language model; the frame index runs in float16."""
