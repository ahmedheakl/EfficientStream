"""Cuts a multi-segment recording of the proactive-mode demo (-model qwen -esqwendemo) into one 1920x1080 video per
segment: the phone on the right; real time around the start and around every response, long silent stretches
fast-forwarded (labelled); on the left the watch prompt, a timeline of the watched seconds filling in as each check
finishes (grey: silent, green: it spoke up), the latest response in large type and the frame of that second.

ES_DEMO=<folder with the take> python make_proactive_multi.py proactive4.json <out folder>
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_demo as md  # noqa: E402

GREY, GREEN = (90, 100, 122), (52, 211, 153)


def segments(rec_log, console_log, offset):
    rec0 = float(re.search(r"first frame wall ([\d.]+)", Path(rec_log).read_text()).group(1)) - offset
    segs = []
    for line in Path(console_log).read_text().splitlines():
        m = re.match(r".*ESQDEMO ([\d.]+) (\w+) ?(.*)", line)
        if not m:
            continue
        t, kind, rest = float(m.group(1)) - rec0, m.group(2), m.group(3)
        if kind == "segment":
            segs.append({"updates": []})
        elif kind == "update" and segs:
            u = re.match(r"(\d+) ([\d.]+) (\d+) (.*)", rest)
            segs[-1]["updates"].append({"t": t, "second": int(u.group(1)), "seconds": float(u.group(2)),
                                        "tokens": int(u.group(3)), "answer": u.group(4)})
        elif segs:
            segs[-1].setdefault(kind, t)
    return segs


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def is_response(u):
    return u["answer"].startswith("</Response>")


def panel(sc, sg, t, mode, speed, thumbs):
    im = md.background()
    md.phone_frame(im)
    d = ImageDraw.Draw(im)
    x, w = md.LEFT_X, md.LEFT_W
    d.text((x, 64), "EfficientStream", font=md.font(36, "Bold"), fill=md.FG)
    d.text((x, 110), "Streaming Qwen3.5-0.8B (OneStreamer recipe), proactive mode, on the phone", font=md.font(24),
           fill=md.DIM)
    y = 190
    d.text((x, y), sc["title"], font=md.font(58, "Bold"), fill=md.FG)
    y += 76
    for line in md.wrap(d, sc["subtitle"], md.font(28), w):
        d.text((x, y), line, font=md.font(28), fill=md.DIM)
        y += 38
    y += 18
    if mode == "ff":
        md.chip(d, x, y, "ff", f"Fast-forward {speed:.0f}×  ·  silent, checking every second")
    else:
        md.chip(d, x, y, "rt", "Live  ·  real time, on the phone")
    y += 54 + 32
    if t < sg["start"]:
        return im, y
    d.text((x, y), "Watching for", font=md.font(22, "Semibold"), fill=md.ACCENT)
    y += 30
    for line in md.wrap(d, f"“{sc['prompt']}”", md.font(34, "Semibold"), w):
        d.text((x, y), line, font=md.font(34, "Semibold"), fill=md.FG)
        y += 44
    y += 16
    # timeline of the watched seconds
    s0, n = sc["start"], sc["seconds"]
    done = [u for u in sg["updates"] if u["t"] <= t]
    bar_y = y + 14
    d.rounded_rectangle([x, bar_y, x + w, bar_y + 14], 7, fill=(32, 40, 62))
    step = w / n
    for u in done:
        i = u["second"] - s0
        if not 0 <= i < n:
            continue
        x0 = x + i * step
        if is_response(u):
            d.rectangle([x0, bar_y - 10, x0 + max(step - 1, 3), bar_y + 24], fill=GREEN)
        else:
            d.rectangle([x0, bar_y, x0 + max(step - 1, 2), bar_y + 14], fill=GREY)
    now = min(n, max(0.0, t - sg["start"]))
    nx = x + now * step
    d.ellipse([nx - 7, bar_y - 1, nx + 7, bar_y + 13], fill=md.LIVE)
    d.text((x, bar_y + 26), mmss(s0), font=md.font(18), fill=md.DIM)
    end_label = mmss(s0 + n)
    d.text((x + w - d.textlength(end_label, font=md.font(18)), bar_y + 26), end_label, font=md.font(18), fill=md.DIM)
    y = bar_y + 58
    last = done[-1] if done else None
    responses = [u for u in done if is_response(u)]
    if last:
        state = "spoke up" if is_response(last) else "silent"
        d.text((x, y), f"Second {mmss(last['second'])}: {state}  ·  checked in {last['seconds']:.2f} s  ·  "
                       f"{len(responses)} response{'s' if len(responses) != 1 else ''} so far",
               font=md.font(24, "Semibold"), fill=GREEN if state == "spoke up" else md.FG)
    else:
        d.text((x, y), "Checking every second…", font=md.font(24, "Semibold"), fill=md.DIM)
    y += 46
    if responses:
        r = responses[-1]
        text = r["answer"][len("</Response>"):].strip()
        ew = 384
        tw = w - ew - 28
        d.text((x, y), f"Response at {mmss(r['second'])}", font=md.font(24, "Bold"), fill=GREEN)
        yy = y + 36
        for line in md.wrap(d, text, md.font(26), tw - 22)[:6]:
            d.text((x + 22, yy), line, font=md.font(26), fill=(214, 220, 232))
            yy += 34
        d.line([(x, y + 40), (x, yy - 4)], fill=GREEN, width=4)
        frame, box = thumbs[r["second"]]
        eh = round(ew * frame.height / frame.width)
        fx, fy = x + w - ew, y + 4
        fr = frame.resize((ew, eh))
        if box:
            bx0, by0, bx1, by1 = box
            m = Image.new("L", fr.size, 100)
            ImageDraw.Draw(m).rectangle([bx0 * ew, by0 * eh, bx1 * ew, by1 * eh], fill=0)
            fr = Image.composite(Image.new("RGB", fr.size, (0, 0, 0)), fr, m)
        im.paste(fr, (fx, fy))
        if box:
            d.rounded_rectangle([fx + bx0 * ew, fy + by0 * eh, fx + bx1 * ew, fy + by1 * eh], 6, outline=GREEN, width=4)
        d.text((fx, fy + eh + 6), f"The frame at {mmss(r['second'])}", font=md.font(18), fill=md.DIM)
    return im, y


def render(sc, sg, cfg, take, out, tmp):
    base_box = sc.get("box")
    seconds = {u["second"] for u in sg["updates"] if is_response(u)}
    thumbs = {}
    for s in seconds:
        p = tmp / f"thumb_{s}.png"
        at = sc.get("thumb_times", {}).get(str(s), s + 0.5)  # a frame of the second the model responded about
        md.run(["ffmpeg", "-y", "-ss", f"{at:.2f}", "-i", os.path.expanduser(sc["source"]), "-frames:v", "1", str(p)])
        thumbs[s] = (Image.open(p).convert("RGB"), sc.get("boxes", {}).get(str(s), base_box))
    # real time around the start and every response, the rest fast-forwarded
    a, b = sg["type"] - 0.2, sg["pause"] + cfg.get("tail", 1.5)
    keep = [(a, sg["start"] + 3.0)] + [(u["t"] - 3.5, u["t"] + 3.5) for u in sg["updates"] if is_response(u)]
    keep.sort()
    merged = []
    for k0, k1 in keep:
        k0, k1 = max(k0, a), min(k1, b)
        if merged and k0 <= merged[-1][1] + cfg.get("min_gap", 3.0):
            merged[-1] = (merged[-1][0], max(merged[-1][1], k1))
        else:
            merged.append((k0, k1))
    pieces_t = []
    cur = a
    for k0, k1 in merged:
        if k0 > cur:
            pieces_t.append(("ff", cur, k0))
        pieces_t.append(("rt", max(k0, cur), k1))
        cur = k1
    if cur < b:
        pieces_t.append(("ff" if b - cur > cfg.get("min_gap", 3.0) else "rt", cur, b))
    speed = cfg.get("speed", 6.0)
    md.mask().save(tmp / "mask.png")
    files = []
    md.title_card(sc["intro"], cfg["title_footer"]).save(tmp / "intro.png")
    md.still_piece(str(tmp / "intro.png"), cfg.get("intro_seconds", 1.6), str(tmp / "p00.mp4"))
    files.append(tmp / "p00.mp4")
    total = cfg.get("intro_seconds", 1.6)
    for i, (mode, t0, t1) in enumerate(pieces_t):
        sp = speed if mode == "ff" else 1.0
        n = max(1, round((t1 - t0) / sp * md.FPS))
        fdir = tmp / f"f{i}"
        fdir.mkdir()
        cache = {}
        for k in range(n):
            t = t0 + k * sp / md.FPS
            key = (mode, t >= sg["start"], sum(u["t"] <= t for u in sg["updates"]),
                   round(min(sc["seconds"], max(0, t - sg["start"])) * (4 if mode == "rt" else 1)))
            if key not in cache:
                path = fdir / f"s_{len(cache):04d}.png"
                panel(sc, sg, t, mode, sp, thumbs)[0].save(path)
                cache[key] = path
            os.link(cache[key], fdir / f"f_{k:05d}.png")
        p = tmp / f"p{i + 1:02d}.mp4"
        md.phone_piece(take, ["-framerate", str(md.FPS), "-i", str(fdir / "f_%05d.png")], str(tmp / "mask.png"),
                       t0, t1 - t0, sp, n / md.FPS, str(p))
        files.append(p)
        total += n / md.FPS
    rs = [u for u in sg["updates"] if is_response(u)]
    silent = [u["seconds"] for u in sg["updates"] if not is_response(u)]
    outro = [[f"{len(sg['updates'])} checks, {len(rs)} responses", 54, "Bold", [245, 247, 250]],
             [f"Silent check: {sum(silent) / max(1, len(silent)):.2f} s on average  ·  a response: "
              f"{sum(u['seconds'] for u in rs) / max(1, len(rs)):.1f} s ({sum(u['tokens'] for u in rs) // max(1, len(rs))} tokens of text)",
              32, "Regular", [148, 160, 184]]]
    md.title_card(outro, cfg["title_footer"]).save(tmp / "outro.png")
    md.still_piece(str(tmp / "outro.png"), cfg.get("outro_seconds", 2.4), str(tmp / "p99.mp4"))
    files.append(tmp / "p99.mp4")
    total += cfg.get("outro_seconds", 2.4)
    (tmp / "list.txt").write_text("".join(f"file '{p}'\n" for p in files))
    md.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-c", "copy",
            "-movflags", "+faststart", out])
    print(f"wrote {out}: {total:.1f} s; pieces " +
          ", ".join(f"{m} {t1 - t0:.1f}s" for m, t0, t1 in pieces_t) + f"; {len(rs)} responses")


def main(cfg_path, out_dir):
    cfg = json.loads(Path(cfg_path).read_text())
    base = Path(os.environ.get("ES_DEMO", "."))
    segs = segments(base / cfg["rec_log"], base / cfg["console_log"], cfg.get("offset", 0.0))
    os.makedirs(out_dir, exist_ok=True)
    for sc in cfg["scenes"]:
        tmp = Path(tempfile.mkdtemp())
        render(sc, segs[sc["segment"]], cfg, str(base / cfg["video"]), os.path.join(out_dir, sc["file"]), tmp)


if __name__ == "__main__":
    main(*sys.argv[1:3])
