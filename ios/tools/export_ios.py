#!/usr/bin/env python3
"""Convert the EfficientStream weights into the files the iOS app loads.

    python ios/tools/export_ios.py --weights ahmedheakl/efficientstream --out ios/models

Writes
    <out>/minicpm/       MiniCPM-V-4.6 for MLX: model.safetensors (renamed / re-laid-out, bfloat16 unless --bits),
                         config.json, tokenizer.json, tokenizer_config.json, prompt.json (special tokens)
    <out>/mobileclip/    MobileCLIP2-S0 for Core ML: ImageEncoder.mlpackage, TextEncoder.mlpackage, tokenizer.json,
                         config.json (preprocessing, context length, padding id)
    <out>/efficientstream_config.json   the pipeline settings (StreamConfig)

Nothing is fine-tuned or changed numerically except: the RMSNorm weights of the language model are stored as
float32 with the checkpoint's +1 offset folded in, and `--bits` (off by default) quantizes the language model's
linear layers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


# ----------------------------------------------------------------------------------------------- MiniCPM-V-4.6
def export_minicpm(src: str, out: str, bits: int = 0, group_size: int = 64) -> None:
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    sd = load_file(os.path.join(src, "model.safetensors"))
    offset_norms = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight",
                    ".k_norm.weight")
    w: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("model.vision_tower.embeddings.patch_embedding.weight"):
            # conv [out, c, kh, kw] -> linear over (kh, kw, c) patches
            w["vision.patch_embedding.weight"] = v.permute(0, 2, 3, 1).reshape(v.shape[0], -1).contiguous()
        elif k == "model.vision_tower.embeddings.patch_embedding.bias":
            w["vision.patch_embedding.bias"] = v
        elif k == "model.vision_tower.embeddings.position_embedding.weight":
            w["vision.position_embedding.weight"] = v
        elif k.startswith("model.vision_tower.encoder.layers."):
            w["vision.layers." + k[len("model.vision_tower.encoder.layers."):]] = v
        elif k.startswith("model.vision_tower.post_layernorm."):
            w["vision.post_layernorm." + k.rsplit(".", 1)[1]] = v
        elif k.startswith("model.vision_tower.vit_merger."):
            w["vision.vit_merger." + k[len("model.vision_tower.vit_merger."):]] = v
        elif k.startswith("model.merger.mlp.0."):
            w["merger." + k[len("model.merger.mlp.0."):]] = v
        elif k.startswith("model.language_model."):
            name = "lm." + k[len("model.language_model."):]
            if name.endswith("conv1d.weight"):
                v = v.permute(0, 2, 1).contiguous()          # [C, 1, K] -> [C, K, 1]
            if name.endswith(offset_norms) or name == "lm.norm.weight":
                v = v.float() + 1.0
            w[name] = v
        else:
            raise ValueError(f"unexpected weight {k}")
    assert cfg.get("merger_times", 1) == 1, "the iOS model implements one downsample merger"

    quant = None
    if bits:
        import mlx.core as mx
        quant = {"bits": bits, "group_size": group_size}
        for name in [n for n in w if n.startswith("lm.layers.") and n.endswith(".weight")]:
            if any(s in name for s in ("layernorm", "norm.weight", "conv1d", "in_proj_a", "in_proj_b")):
                continue
            if w[name].ndim != 2:
                continue
            q, s, b = mx.quantize(mx.array(w[name].float().numpy()).astype(mx.bfloat16), group_size=group_size, bits=bits)
            base = name[: -len(".weight")]
            w[name] = torch.from_numpy(np.array(q))
            w[base + ".scales"] = torch.from_numpy(np.array(s.astype(mx.float32))).to(torch.bfloat16)
            w[base + ".biases"] = torch.from_numpy(np.array(b.astype(mx.float32))).to(torch.bfloat16)

    save_file(w, os.path.join(out, "model.safetensors"))
    tc, vc = cfg["text_config"], cfg["vision_config"]
    rope = tc.get("rope_parameters") or {}
    config = {
        "vision_config": {k: vc[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                             "num_attention_heads", "patch_size", "image_size", "layer_norm_eps")},
        "text_config": {**{k: tc[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                              "num_attention_heads", "num_key_value_heads", "head_dim",
                                              "rms_norm_eps", "vocab_size", "linear_num_key_heads",
                                              "linear_num_value_heads", "linear_key_head_dim",
                                              "linear_value_head_dim", "linear_conv_kernel_dim", "layer_types")},
                        "rope_theta": rope.get("rope_theta", 10000000),
                        "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25)},
        "insert_layer_id": cfg["insert_layer_id"],
        "image_token_id": cfg["image_token_id"],
    }
    if quant:
        config["quantization"] = quant
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)

    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    # special tokens and ids used to build prompts (read from the processor, not hard-coded in Swift)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(src)
    tok = proc.tokenizer
    names = ("image_token", "image_start_token", "image_end_token", "slice_start_token", "slice_end_token",
             "image_id_start_token", "image_id_end_token")
    special = {n: getattr(tok, n) for n in names}
    gen = json.load(open(os.path.join(src, "generation_config.json")))
    eos = gen.get("eos_token_id", cfg.get("eos_token_id"))
    prompt = {
        **special,
        "special_tokens": {t: tok.convert_tokens_to_ids(t) for t in
                           sorted({*special.values(), "<|im_start|>", "<|im_end|>", "<think>", "</think>"})},
        "eos_token_ids": sorted(set((eos if isinstance(eos, list) else [eos]) + [tok.eos_token_id])),
        "answer_open_ids": tok("<answer>", add_special_tokens=False)["input_ids"],
        "letter_ids": [tok(chr(65 + i), add_special_tokens=False)["input_ids"][0] for i in range(10)],
        "image_processor": {k: proc.image_processor.to_dict()[k] for k in ("image_mean", "image_std", "patch_size")},
        "image_token_divisor": 16,
    }
    json.dump(prompt, open(os.path.join(out, "prompt.json"), "w"), indent=1)
    print(f"minicpm: {len(w)} tensors -> {out}")


# ----------------------------------------------------------------------------------------------- MobileCLIP2-S0
def _patch_coremltools_cast() -> None:
    """coremltools' torch `int`/`bool` cast calls int() on a folded 1-element array, which NumPy >= 2.5 rejects
    ("only 0-dimensional arrays can be converted to Python scalars"); cast the element instead."""
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    original = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and np.ndim(x.val) > 0 and np.size(x.val) == 1:
            context.add(mb.const(val=dtype(np.asarray(x.val).item()), name=node.name), node.name)
            return
        original(context, node, dtype, dtype_name)

    ops._cast = _cast


def export_mobileclip(src: str, out: str) -> None:
    import coremltools as ct
    import numpy as np
    import open_clip
    import torch

    # float32: in float16 the image embedding drifts to ~0.994 cosine of PyTorch's, enough to change retrieval
    _patch_coremltools_cast()
    os.makedirs(out, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(f"local-dir:{src}")
    model.eval()
    try:  # MobileOne-style reparameterisation (inference form), as Apple recommends for export
        from open_clip.timm_model import TimmModel  # noqa: F401
        from timm.utils.model import reparameterize_model
        model = reparameterize_model(model)
    except Exception as e:  # pragma: no cover
        print("note: no reparameterisation:", e)
    ocfg = json.load(open(os.path.join(src, "open_clip_config.json")))
    size = ocfg["model_cfg"]["vision_cfg"]["image_size"]
    ctx = ocfg["model_cfg"]["text_cfg"]["context_length"]

    class Img(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):  # x: [1, 3, S, S] in [0, 1] (mean 0 / std 1: no normalisation)
            return torch.nn.functional.normalize(self.m.encode_image(x), dim=-1)

    class Txt(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):  # ids: [1, ctx] int32
            return torch.nn.functional.normalize(self.m.encode_text(ids.long()), dim=-1)

    # the fused nn.MultiheadAttention fast path traces to an op Core ML cannot convert; use the plain ops
    torch.backends.mha.set_fastpath_enabled(False)
    ex_img = torch.rand(1, 3, size, size)
    # the tokenizer EfficientStream's FrameIndex uses (efficientstream/index.py)
    tokenizer = open_clip.get_tokenizer(ocfg.get("model_name") or "MobileCLIP2-S0")
    ex_txt = tokenizer(["a photo of a cat"]).to(torch.int32)
    with torch.no_grad():
        ti = torch.jit.trace(Img(model), ex_img)
        tt = torch.jit.trace(Txt(model), ex_txt)
        ref_i, ref_t = Img(model)(ex_img).numpy(), Txt(model)(ex_txt).numpy()
    for name, traced, inp, ref in (
        ("ImageEncoder", ti, ct.TensorType(name="image", shape=ex_img.shape), ref_i),
        ("TextEncoder", tt, ct.TensorType(name="ids", shape=ex_txt.shape, dtype=np.int32), ref_t),
    ):
        ml = ct.convert(traced, inputs=[inp], outputs=[ct.TensorType(name="embedding")],
                        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
                        convert_to="mlprogram")
        path = os.path.join(out, name + ".mlpackage")
        shutil.rmtree(path, ignore_errors=True)
        ml.save(path)
        try:
            got = ml.predict({inp.name: (ex_img.numpy() if name == "ImageEncoder" else ex_txt.numpy())})["embedding"]
            print(f"mobileclip {name}: cosine to PyTorch {float((got * ref).sum()):.5f}")
        except Exception as e:  # predict needs macOS
            print(f"mobileclip {name}: saved (no check: {e})")
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
              "LICENSE"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(out, f))
    pre = ocfg["preprocess_cfg"]
    pad = getattr(tokenizer, "tokenizer", None)
    json.dump({"image_size": size, "context_length": ctx, "mean": pre["mean"], "std": pre["std"],
               "interpolation": pre.get("interpolation", "bilinear"), "resize_mode": pre.get("resize_mode", "shortest"),
               "embed_dim": ocfg["model_cfg"]["embed_dim"],
               "pad_token_id": int(pad.pad_token_id) if pad is not None and pad.pad_token_id is not None else 0},
              open(os.path.join(out, "config.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="ahmedheakl/efficientstream", help="Hugging Face repo or local folder")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    p.add_argument("--bits", type=int, default=0, choices=[0, 4, 8],
                   help="quantize the language model's linear layers (0 = keep bfloat16, the evaluated precision)")
    p.add_argument("--only", choices=["minicpm", "mobileclip"])
    a = p.parse_args()
    root = a.weights
    if not os.path.isdir(root):
        from huggingface_hub import snapshot_download
        root = snapshot_download(a.weights)
    os.makedirs(a.out, exist_ok=True)
    if a.only in (None, "minicpm"):
        export_minicpm(os.path.join(root, "backbone"), os.path.join(a.out, "minicpm"), a.bits)
    if a.only in (None, "mobileclip"):
        export_mobileclip(os.path.join(root, "encoder"), os.path.join(a.out, "mobileclip"))
    shutil.copy(os.path.join(root, "efficientstream_config.json"), os.path.join(a.out, "efficientstream_config.json"))


if __name__ == "__main__":
    main()
