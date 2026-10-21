#!/usr/bin/env python3
"""Convert the streaming Qwen3.5-0.8B (stream-qwen3.5-0.8b/ in ahmedheakl/efficientstream) into the files the iOS
app loads.

    python ios/tools/export_qwen.py --src <stream-qwen3.5-0.8b folder> --out ios/models/qwen

Writes <out>/: model.safetensors (renamed / re-laid-out, bfloat16), config.json, tokenizer.json,
tokenizer_config.json, prompt.json (special tokens, letter ids, the runner's system prompts).

Nothing is changed numerically except: the language model's RMSNorm weights are stored as float32 with the
checkpoint's +1 offset folded in (as for MiniCPM-V-4.6, the same Qwen3.5 text model), the patch embedding (Conv3d
[out, C, T, 16, 16]) is stored as a linear layer over (C, T, row, column) patch vectors, the depthwise conv weights
as [channels, kernel, 1], and lm_head is dropped (tied to the embeddings; byte-identical in the checkpoint).
Only numpy is needed (safetensors are read and written directly).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import struct

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_safetensors(path):
    """name -> (dtype, shape, raw bytes)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a)
            out[name] = (meta["dtype"], meta["shape"], f.read(b - a))
    return out


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, raw bytes)."""
    header, offset = {}, 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in sorted(tensors):
            f.write(tensors[name][2])


def bf16_array(t):
    dtype, shape, data = t
    assert dtype == "BF16", dtype
    return np.frombuffer(data, dtype=np.uint16).reshape(shape)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)  # round to nearest even


def runner_constants(runner=None):
    """System prompts of proactive/stream_qwen.py (parsed, not imported: the runner imports torch). The caption-memory
    (PHCM) prompts are there from the run-4 runner on."""
    tree = ast.parse(open(runner or os.path.join(REPO, "proactive", "stream_qwen.py")).read())
    names = {"STREAM_SYSTEM", "QA_SYSTEM", "QA_DIRECT_SYSTEM", "POST_PROMPT", "MIN_PIXELS", "MAX_PIXELS"}
    optional = {"CAPTION_SYSTEM", "CAPTION_INSTRUCTION", "MEMORY_PREAMBLE", "MEMORY_POSTAMBLE", "MEMORY_MAX_TOKENS"}
    names |= optional
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    out[target.id] = ast.literal_eval(node.value)
                elif isinstance(target, ast.Tuple):
                    for t, v in zip(target.elts, node.value.elts):
                        if isinstance(t, ast.Name) and t.id in names:
                            out[t.id] = ast.literal_eval(v)
    assert names - optional <= set(out), names - optional - set(out)
    return out


def export(src: str, out: str, runner: str = None) -> None:
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    sd = read_safetensors(os.path.join(src, "model.safetensors"))
    offset_norms = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight",
                    ".k_norm.weight")
    w = {}
    for k, t in sd.items():
        dtype, shape, data = t
        if k == "lm_head.weight":
            assert data == sd["model.language_model.embed_tokens.weight"][2], "lm_head is not tied"
            continue
        if k == "model.visual.patch_embed.proj.weight":
            # Conv3d [out, C, T, kh, kw] -> linear over (C, T, kh, kw) patch vectors (the processor's order)
            a = bf16_array(t).reshape(shape[0], -1)
            w["vision.patch_embed.weight"] = ("BF16", a.shape, a.tobytes())
        elif k == "model.visual.patch_embed.proj.bias":
            w["vision.patch_embed.bias"] = t
        elif k.startswith("model.visual."):
            w["vision." + k[len("model.visual."):]] = t
        elif k.startswith("model.language_model."):
            name = "lm." + k[len("model.language_model."):]
            if name.endswith("conv1d.weight"):
                a = np.ascontiguousarray(bf16_array(t).transpose(0, 2, 1))  # [C, 1, K] -> [C, K, 1]
                t = ("BF16", a.shape, a.tobytes())
            if name.endswith(offset_norms) or name == "lm.norm.weight":
                a = bf16_to_f32(bf16_array(t)) + np.float32(1.0)
                t = ("F32", a.shape, a.astype(np.float32).tobytes())
            w[name] = t
        else:
            raise ValueError(f"unexpected weight {k}")
    write_safetensors(os.path.join(out, "model.safetensors"), w)

    tc, vc = cfg["text_config"], cfg["vision_config"]
    rope = tc["rope_parameters"]
    assert rope.get("mrope_interleaved") and not vc.get("deepstack_visual_indexes"), "unsupported variant"
    config = {
        "vision_config": {k: vc[k] for k in ("hidden_size", "intermediate_size", "depth", "num_heads", "patch_size",
                                             "temporal_patch_size", "spatial_merge_size", "num_position_embeddings",
                                             "out_hidden_size")},
        "text_config": {**{k: tc[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                              "num_attention_heads", "num_key_value_heads", "head_dim",
                                              "rms_norm_eps", "vocab_size", "linear_num_key_heads",
                                              "linear_num_value_heads", "linear_key_head_dim",
                                              "linear_value_head_dim", "linear_conv_kernel_dim", "layer_types")},
                        "rope_theta": rope["rope_theta"], "partial_rotary_factor": rope["partial_rotary_factor"]},
        "mrope_section": rope["mrope_section"],
        "image_token_id": cfg["image_token_id"],
        "video_token_id": cfg["video_token_id"],
        "vision_start_token_id": cfg["vision_start_token_id"],
        "vision_end_token_id": cfg["vision_end_token_id"],
    }
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)

    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    tok = json.load(open(os.path.join(src, "tokenizer.json")))
    added = {a["content"]: a["id"] for a in tok["added_tokens"]}
    vocab = tok["model"]["vocab"]
    proc = json.load(open(os.path.join(src, "processor_config.json")))
    gen = json.load(open(os.path.join(src, "generation_config.json")))
    eos = gen["eos_token_id"] if isinstance(gen["eos_token_id"], list) else [gen["eos_token_id"]]
    im = proc["image_processor"]
    prompt = {
        "special_tokens": {t: i for t, i in added.items() if t.startswith(("<|", "</"))},
        # the runner cuts the answer at <|im_end|> or <|endoftext|>
        "eos_token_ids": sorted(set(eos + [added["<|im_end|>"], added["<|endoftext|>"]])),
        "letter_ids": [vocab[chr(65 + i)] for i in range(10)],
        "state_tokens": {t: added[t] for t in ("</Silence>", "</Standby>", "</Response>", "</Observe>", "</Summary>")
                         if t in added},
        "image_mean": im["image_mean"], "image_std": im["image_std"],
        **runner_constants(runner),
    }
    json.dump(prompt, open(os.path.join(out, "prompt.json"), "w"), indent=1)
    print(f"qwen: {len(w)} tensors -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the stream-qwen3.5-0.8b folder (hf download ahmedheakl/efficientstream "
                                                 "--include 'stream-qwen3.5-0.8b/*')")
    ap.add_argument("--out", default=os.path.join(REPO, "ios", "models", "qwen"))
    ap.add_argument("--runner", help="the runner whose prompts to use (default: this checkout's proactive/stream_qwen.py)")
    a = ap.parse_args()
    export(a.src, a.out, a.runner)


if __name__ == "__main__":
    main()
