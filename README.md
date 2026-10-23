# Streaming Qwen3.5-0.8B

The streaming Qwen3.5-0.8B (our run 4, `stream-qwen3.5-0.8b-r4` in
[ahmedheakl/efficientstream](https://huggingface.co/ahmedheakl/efficientstream)) as one loop:

1. **Continuous captioning.** Once a second, the second's frames go in as one chat turn and the model answers
   `</Silence>`, `</Observe>` + a visual detail, or `</Summary>` + a summary of a finished segment. The details and
   summaries are kept as the caption memory.
2. **A streaming KV cache.** One cache is kept for the whole video. Each second prefills only its own turn (about 123
   tokens at 1 fps, 223 at 4 fps, 423 at 8 fps) and the answer's tokens, never re-reading the window. When 40 seconds have
   piled up, the oldest 8 are cut out of the 6 attention layers' caches, leaving the last 32; nothing is recomputed:
   positions keep counting up and the 18 gated DeltaNet layers keep their state. The first turn (system prompt,
   instruction, second 0) always stays.
3. **Compaction when you ask.** A question gets the captions of the seconds no longer in the cache, compacted: a summary
   replaces the details it covers (except details from the last 60 s), and identical details are merged.
4. **Retrieved frames.** One frame a second is kept with its MobileCLIP2-S0 embedding. A question also gets the 4 earlier
   frames that best match it (the 32 most similar, sharp ones preferred, at least 3 s apart), shown as images.

A question forks the cache (copy-on-write) and prefills only its own turn: compacted captions, retrieved frames,
question. The stream goes on unchanged.

## Run

```bash
pip install -r requirements.txt
python run.py video.mp4 --questions "30:What is on the table?" "95:Where did I put the cup?"
```

Weights download from Hugging Face on first use (`stream-qwen3.5-0.8b-r4/` and MobileCLIP2-S0 in `encoder/`). Options:
`--fps` (default 4, as trained), `--window` (seconds kept in the cache, 32), `--evict-every` (seconds cut at once, 8),
`--retrieved` (frames shown with a question, 4).

In code:

```python
from streaming_qwen import FrameIndex, Model, StreamingQwen

sq = StreamingQwen(Model("<stream-qwen3.5-0.8b-r4 folder>"), FrameIndex("<encoder folder>"))
for second, frames, times in video_seconds:     # PIL frames of that second, 4 at 4 fps
    u = sq.feed(second, frames, times)          # u["answer"]: </Silence> or a caption
answer = sq.ask("Where did I put the cup?")["answer"]
```

## Files

| | |
|---|---|
| `streaming_qwen/model.py` | loading, a second of video and a retrieved frame through the vision tower, prefill from an M-RoPE position; `fork` and `evict` on the cache |
| `streaming_qwen/stream.py` | the caption stream on one KV cache (`CaptionStream`), questions on a fork, and the whole loop (`StreamingQwen`) |
| `streaming_qwen/memory.py` | the caption memory, its compaction, and the text block a question reads |
| `streaming_qwen/retrieval.py` | MobileCLIP2-S0 embeddings, the kept frames, frame selection |
| `streaming_qwen/prompts.py` | the prompts and frame size the model was trained with |
| `run.py` | stream a video, ask at given seconds |
| `tests/test_cache.py` | the cache plumbing on a tiny random model (CPU, seconds): chunked appends equal one prefill, forks are independent, eviction, positions, compaction, frame selection |

Needs transformers 5.19 or later: from 5.19 on, its gated DeltaNet continues the cached state over a multi-token append.
