"""Cuts a recording of the proactive-mode demo (-model qwen -esqwendemo) into a 1920x1080 video: the phone on the
right in real time (not sped up), and on the left the watch prompt, one box per second filling in as each of the
model's checks finishes (grey: silent, green: it spoke up), the response in large type and the moment it refers to.

Times come from the app's ESQDEMO console lines and record_iphone's log, corrected by the screen-to-recording offset
(sync_offset.py; about 0.23 s on the iPhone 17 Pro over USB).

ES_DEMO=<folder with the take> python make_proactive.py proactive.json out.mp4
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

GREY, GREEN, AMBER = (90, 100, 122), (52, 211, 153), (251, 191, 36)


def events(rec_log, console_log, offset):
    rec0 = float(re.search(r"first frame wall ([\d.]+)", Path(rec_log).read_text()).group(1)) - offset
    ev = {"updates": []}
    for line in Path(console_log).read_text().splitlines():
        m = re.match(r".*ESQDEMO ([\d.]+) (\w+) ?(.*)", line)
        if not m:
            continue
        t, kind, rest = float(m.group(1)) - rec0, m.group(2), m.group(3)
        if kind == "update":
            u = re.match(r"(\d+) ([\d.]+) (\d+) (.*)", rest)
            ev["updates"].append({"t": t, "second": int(u.group(1)), "seconds": float(u.group(2)),
                                  "tokens": int(u.group(3)), "answer": u.group(4)})
        else:
            ev.setdefault(kind, t)
    return ev


def mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def card(cfg, ev, t, evidence):
    im = md.background()
    md.phone_frame(im)
    d = ImageDraw.Draw(im)
    x, w = md.LEFT_X, md.LEFT_W
    d.text((x, 64), "EfficientStream", font=md.font(36, "Bold"), fill=md.FG)
    d.text((x, 110), cfg["model_line"], font=md.font(24), fill=md.DIM)
    y = 200
    d.text((x, y), cfg["title"], font=md.font(64, "Bold"), fill=md.FG)
    y += 84
    for line in md.wrap(d, cfg["subtitle"], md.font(30), w):
        d.text((x, y), line, font=md.font(30), fill=md.DIM)
        y += 40
    y += 22
    md.chip(d, x, y, "rt", "Live  ·  real time, on the phone")
    y += 54 + 40
    started = t >= ev["start"]
    if started:
        d.text((x, y), "Watching for", font=md.font(24, "Semibold"), fill=md.ACCENT)
        y += 34
        for line in md.wrap(d, f"“{cfg['prompt']}”", md.font(36, "Semibold"), w):
            d.text((x, y), line, font=md.font(36, "Semibold"), fill=md.FG)
            y += 46
        y += 18
        # one box per second, filled as its check finishes
        done = [u for u in ev["updates"] if u["t"] <= t]
        seconds = cfg["seconds"]
        bw, gap = 64, 10
        for i, s in enumerate(seconds):
            bx = x + i * (bw + gap)
            u = next((u for u in done if u["second"] == s), None)
            col = (GREEN if u["answer"].startswith("</Response>") else AMBER if u["answer"].startswith("</Standby>")
                   else GREY) if u else (32, 40, 62)
            d.rounded_rectangle([bx, y, bx + bw, y + 44], 8, fill=col)
            d.text((bx + bw / 2 - d.textlength(mmss(s), font=md.font(16)) / 2, y + 52), mmss(s), font=md.font(16),
                   fill=md.DIM)
        y += 84
        last = done[-1] if done else None
        if last:
            state = "spoke up" if last["answer"].startswith("</Response>") else "silent"
            d.text((x, y), f"Second {mmss(last['second'])}: {state}  ·  checked in {last['seconds']:.2f} s",
                   font=md.font(26, "Semibold"), fill=md.FG if state == "silent" else GREEN)
        else:
            d.text((x, y), "Checking every second…", font=md.font(26, "Semibold"), fill=md.DIM)
        y += 52
        resp = next((u for u in done if u["answer"].startswith("</Response>")), None)
        if resp:
            text = resp["answer"][len("</Response>"):].strip()
            # the response on the left, the moment it refers to on the right
            ew = 400
            tw = w - ew - 30
            d.text((x, y), f"Response at {mmss(resp['second'])}", font=md.font(26, "Bold"), fill=GREEN)
            yy = y + 38
            for line in md.wrap(d, text, md.font(28), tw - 22):
                d.text((x + 22, yy), line, font=md.font(28), fill=(214, 220, 232))
                yy += 37
            d.line([(x, y + 42), (x, yy - 4)], fill=GREEN, width=4)
            ev_im, (bx0, by0, bx1, by1) = evidence
            eh = round(ew * ev_im.height / ev_im.width)
            fx, fy = x + w - ew, y + 4
            frame = ev_im.resize((ew, eh))
            dim = Image.new("RGB", frame.size, (0, 0, 0))
            m = Image.new("L", frame.size, 110)
            ImageDraw.Draw(m).rectangle([bx0 * ew, by0 * eh, bx1 * ew, by1 * eh], fill=0)
            frame = Image.composite(dim, frame, m)
            im.paste(frame, (fx, fy))
            d.rounded_rectangle([fx + bx0 * ew, fy + by0 * eh, fx + bx1 * ew, fy + by1 * eh], 6, outline=GREEN,
                                width=4)
            d.text((fx, fy + eh + 8), f"The current frame ({mmss(resp['second'])})", font=md.font(20), fill=md.DIM)
    ff = md.font(20)
    lines = md.wrap(d, cfg["footer"], ff, w)
    fy = md.H - 40 - 28 * len(lines)
    for line in lines:
        d.text((x, fy), line, font=ff, fill=md.DIM)
        fy += 28
    return im


def main(cfg_path, out):
    cfg = json.loads(Path(cfg_path).read_text())
    base = Path(os.environ.get("ES_DEMO", "."))
    ev = events(base / cfg["rec_log"], base / cfg["console_log"], cfg.get("offset", 0.0))
    take = str(base / cfg["video"])
    resp = next(u for u in ev["updates"] if u["answer"].startswith("</Response>"))
    a, b = ev["type"] - cfg.get("lead", 0.3), resp["t"] + cfg.get("tail", 2.0)
    src = os.path.expanduser(cfg["evidence"]["video"])
    tmp = Path(tempfile.mkdtemp())
    md.run(["ffmpeg", "-y", "-ss", str(cfg["evidence"]["t"]), "-i", src, "-frames:v", "1", str(tmp / "ev.png")])
    evidence = (Image.open(tmp / "ev.png").convert("RGB"), cfg["evidence"]["box"])
    md.mask().save(tmp / "mask.png")
    pieces = []
    md.title_card(cfg["intro"], cfg["title_footer"]).save(tmp / "intro.png")
    md.still_piece(str(tmp / "intro.png"), cfg["intro_seconds"], str(tmp / "p0.mp4"))
    pieces.append(tmp / "p0.mp4")
    fdir = tmp / "frames"
    fdir.mkdir()
    n = round((b - a) * md.FPS)
    rendered = {}
    for k in range(n):
        t = a + k / md.FPS
        key = (t >= ev["start"], sum(u["t"] <= t for u in ev["updates"]))
        if key not in rendered:
            path = fdir / f"state_{len(rendered):03d}.png"
            card(cfg, ev, t, evidence).save(path)
            rendered[key] = path
        os.link(rendered[key], fdir / f"f_{k:05d}.png")
    md.phone_piece(take, ["-framerate", str(md.FPS), "-i", str(fdir / "f_%05d.png")], str(tmp / "mask.png"), a, b - a,
                   1.0, n / md.FPS, str(tmp / "p1.mp4"))
    pieces.append(tmp / "p1.mp4")
    md.title_card(cfg["outro"], cfg["title_footer"]).save(tmp / "outro.png")
    md.still_piece(str(tmp / "outro.png"), cfg["outro_seconds"], str(tmp / "p2.mp4"))
    pieces.append(tmp / "p2.mp4")
    (tmp / "list.txt").write_text("".join(f"file '{p}'\n" for p in pieces))
    md.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-c", "copy",
            "-movflags", "+faststart", out])
    print(f"wrote {out}: {cfg['intro_seconds'] + n / md.FPS + cfg['outro_seconds']:.1f} s; live part {b - a:.1f} s "
          f"({len(rendered)} panel states; response at {resp['t'] - a:.1f} s into it)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
