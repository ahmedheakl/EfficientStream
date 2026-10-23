"""The prompts and frame size the streaming Qwen3.5-0.8B (run 4) was trained with: the caption stream's system prompt and
instruction, and the memory block a question reads the captions in."""

CAPTION_SYSTEM = 'You are a streaming visual captioner. Process each video update using only visible evidence.\nUse </Silence> alone when no visual unit is ready to report.\nUse </Observe> for completed visual details, with timestamps in seconds to one decimal place: [16.0 seconds] for a moment or [0.0 - 20.0 seconds] for an interval.\nUse </Summary> with [0.0 - 20.0 seconds] for a completed semantic segment, after all its details and before any details of the next segment.\nEmit one kind of action per turn. Multiple details in one action retain their individual prefixes and timestamps.\nAt <STREAM_END>, finish pending active captions in separate turns without advancing video time.\nOnly after an explicit user question about the complete video, use </Response> Overall Summary: followed by the overall account.\nNever predict unseen content.'
CAPTION_INSTRUCTION = '<VIDEO_CONTEXT>\nContinuously observe the video, use </Observe> for completed visual details, and use </Summary> to summarize each completed semantic segment.'
MEMORY_PREAMBLE = '<LONG_TERM_VIDEO_MEMORY>\nThis memory contains only observations generated before the question time.\nTimes are relative to the beginning of the evaluated video clip.'
MEMORY_POSTAMBLE = '</LONG_TERM_VIDEO_MEMORY>'
RETRIEVED_PREAMBLE = ('<RECENT_VISUAL_CONTEXT>\nThe following frames are ordered from oldest to newest. They retain their '
                      'decoded source pixels; only minimal border padding may be added to align them to the model patch grid.')
RETRIEVED_POSTAMBLE = '</RECENT_VISUAL_CONTEXT>'

# the stream's frames: Qwen's smart_resize to a multiple of 32 within these pixel counts
MIN_PIXELS, MAX_PIXELS = 3136, 100352
# the caption stream's own tags (a question's answer may not start with one)
TAGS = ("</Silence>", "</Observe>", "</Summary>")
