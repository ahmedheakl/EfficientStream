"""RT (results/HYBRID_RULES.md): stronger thumbnail-retrieval banks. One decode pass per video at 1 fps (same frames, times and
sharpness as frame_bank), several embeddings per frame:
  pad      CLIP ViT-H/14 on the whole frame padded to a square (no centre crop)
  crop3    CLIP ViT-H/14 on left / centre / right square crops (top / centre / bottom for portrait)
  tile7    CLIP ViT-H/14 on the padded frame + a 2 x 3 grid of tiles (3 x 2 for portrait), each padded to a square
  sg2      SigLIP2 so400m-patch14-384 on the whole frame (its own preprocessing)
  sg2tile  SigLIP2 on the whole frame + the same tiles
Out: out/frame_bank_rt/<bench>/<video>.npz with ts, sharp and one float16 array per variant (frames x views x dim).
Selection (answer_variants --select <variant>): question embedded with the variant's text encoder, frame score = best view, then as
CLIP + sharpness (candidates / rank_scores("sharp") / pick of frame_bank).
Build: python frame_bank2.py --bench ovo|sb [--split dev|test|all] --shard k --nshards n [--limit N]
"""
import argparse, collections, hashlib, time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out/frame_bank_rt"
SIGLIP = "google/siglip2-so400m-patch14-384"
VARIANTS = ("pad", "crop3", "tile7", "sg2", "sg2tile")
_ENC = {}


def _pad(im):
    from PIL import Image
    w, h = im.size
    s = max(w, h)
    out = Image.new("RGB", (s, s))
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


def _crops(im):
    w, h = im.size
    s = min(w, h)
    if w >= h:
        return [im.crop((x, 0, x + s, s)) for x in (0, (w - s) // 2, w - s)]
    return [im.crop((0, y, s, y + s)) for y in (0, (h - s) // 2, h - s)]


def _tiles(im):
    w, h = im.size
    cols, rows = (3, 2) if w >= h else (2, 3)
    return [im.crop((round(c * w / cols), round(r * h / rows), round((c + 1) * w / cols), round((r + 1) * h / rows)))
            for r in range(rows) for c in range(cols)]


def clip():
    if "clip" not in _ENC:
        import frame_bank as FB
        _ENC["clip"] = FB.load_clip()
    return _ENC["clip"]


def siglip():
    if "sg2" not in _ENC:
        import torch
        from transformers import AutoModel, AutoProcessor
        _ENC["sg2"] = (AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to("cuda").eval(), AutoProcessor.from_pretrained(SIGLIP))
    return _ENC["sg2"]


def _img_feats(kind, images):
    import torch
    model, proc = clip() if kind == "clip" else siglip()
    out = []
    for i in range(0, len(images), 128):
        x = proc(images=images[i:i + 128], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            e = model.get_image_features(pixel_values=x["pixel_values"].to(model.dtype)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def text_vec(variant, text):
    import torch
    if variant.startswith("sg2"):
        model, proc = siglip()
        x = proc(text=[text], return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(model.device)
    else:
        model, proc = clip()
        x = proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(model.device)
    with torch.inference_mode():
        e = model.get_text_features(**x).float()
    return (e / e.norm(dim=-1, keepdim=True))[0].cpu().numpy()


def embed_frames(images):
    """images: list of PIL frames -> dict variant -> (frames x views x dim) float16."""
    n = len(images)
    pads, crops, tiles = [_pad(im) for im in images], [c for im in images for c in _crops(im)], [_pad(t) for im in images for t in _tiles(im)]
    nt = len(tiles) // n
    cp, cc, ct = _img_feats("clip", pads), _img_feats("clip", crops), _img_feats("clip", tiles)
    sw, st = _img_feats("sg2", images), _img_feats("sg2", [t for im in images for t in _tiles(im)])
    d, ds = cp.shape[1], sw.shape[1]
    return dict(pad=cp[:, None], crop3=cc.reshape(n, 3, d), tile7=np.concatenate([cp[:, None], ct.reshape(n, nt, d)], 1),
                sg2=sw[:, None], sg2tile=np.concatenate([sw[:, None], st.reshape(n, nt, ds)], 1))


_BANK = {}


def load_bank(bench, video):
    k = (bench, video)
    if k not in _BANK:
        _BANK.clear()
        d = np.load(OUT / bench / (video.replace("/", "__") + ".npz"))
        _BANK[k] = {x: d[x] for x in d.files}
    return _BANK[k]


def candidates(variant, bench, video, question, lo, hi, pool=32):
    """As frame_bank.candidates, with the variant's embeddings (frame score = best view): (times, scores, log(1 + sharpness))."""
    b = load_bank(bench, video)
    m = (b["ts"] >= lo) & (b["ts"] < hi)
    if not m.any():
        return b["ts"][m], np.zeros(0, np.float32), np.zeros(0, np.float32)
    sim = (b[variant][m].astype(np.float32) @ text_vec(variant, question)).max(axis=1)
    ts, sh = b["ts"][m], np.log1p(b["sharp"][m])
    top = np.argsort(-sim)[:pool]
    return ts[top], sim[top], sh[top]


def build(args):
    import cv2
    from PIL import Image
    import answer_variants as AV
    import frame_bank as FB
    dev = lambda v: int(hashlib.md5(v.encode()).hexdigest(), 16) % 2 == 0
    ends, vroot = collections.defaultdict(float), None
    for b in [k for k, v in FB.BANK_OF.items() if v == args.bench]:
        rows, vroot, _ = AV.load_bench(b)
        for r in rows:
            if args.split == "all" or (args.split == "dev") == dev(r["video"]):
                ends[r["video"]] = max(ends[r["video"]], float(r["end"]))
    out = OUT / args.bench
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(ends)[args.shard::args.nshards][:args.limit or None]
    print(f"shard {args.shard}: {len(vids)} videos", flush=True)
    for j, vid in enumerate(vids):
        dst = out / (vid.replace("/", "__") + ".npz")
        if dst.exists():
            continue
        old = FB.load_bank(args.bench, vid)
        t0 = time.time()
        cap = cv2.VideoCapture(str(vroot / vid))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {round(t * fps): float(t) for t in range(int(ends[vid]) + 1) if round(t * fps) < n}
        ts, imgs, sharp, i = [], [], [], 0
        while i <= max(want, default=-1):
            if i in want:
                ok, fr = cap.read()
                if not ok:
                    break
                ts.append(want[i]); sharp.append(FB.sharpness(fr)); imgs.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            elif not cap.grab():
                break
            i += 1
        parts = collections.defaultdict(list)
        for k in range(0, len(imgs), 64):
            for v, a in embed_frames(imgs[k:k + 64]).items():
                parts[v].append(a)
        emb = {v: np.concatenate(a) for v, a in parts.items()}
        same = len(ts) == len(old["ts"]) and np.allclose(ts, old["ts"])
        np.savez(dst.with_suffix(".tmp.npz"), ts=np.array(ts, np.float32), sharp=np.array(sharp, np.float32), **emb)
        Path(dst.with_suffix(".tmp.npz")).rename(dst)
        print(f"shard {args.shard}: {j + 1}/{len(vids)} {vid} {len(ts)} frames in {time.time() - t0:.0f}s (times as frame_bank: {same})", flush=True)
    print("BANK2_DONE", args.bench, args.shard, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    build(ap.parse_args())
