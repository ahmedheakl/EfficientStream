"""EfficientStream: streaming video question answering with a bounded frame store, a MobileCLIP2-S0 frame index and
MiniCPM-V-4.6 answering in one forward pass."""
from .config import StreamConfig
from .session import EfficientStream, Result, StreamSession

__all__ = ["EfficientStream", "StreamSession", "StreamConfig", "Result"]
__version__ = "0.1.0"
