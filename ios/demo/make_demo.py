"""Cuts a phone screen recording of the app's demo mode (-esdemo) into a 1920x1080 presentation video.

Each scene: the stretch where the app fast-forwards the video (sped up further here, labelled), then the question
being typed and answered in real time (not sped up). On the left, the question and the answer are mirrored in large
type as they appear on the phone (the answer revealed at the phone's streaming pace, from the logged times), then the
answer's measured time to first words and tokens/s.

A scene with an "evidence" entry also points at where its answer can be seen in the stream: a timeline of the stream
(filling up while the video is watched, 0:00 -> now when the question is asked), and, from the first answer words, the
evidence moment marked on it and its frame (from the source video) with the key object highlighted and labelled. The
evidence moment is chosen by hand (scenes file: video, time, box in normalized coordinates, label); it is where the
fact is visible in the stream, not a claim about which frames the model retrieved.

Times come from the app's ESDEMO console lines and record_iphone's log (frames are stamped with their arrival time,
so movie time = wall time - the first frame's wall time), corrected by the offset sync_offset.py measures.

ES_DEMO=<folder with the takes and logs> python make_demo.py scenes_60s_evidence.json out.mp4
(macOS: the captions use the system font, SF Pro. Evidence videos are resolved relative to ES_DEMO.)
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
PH_H = 1000
PH_W = round(1206 * PH_H / 2622) // 2 * 2
PH_X, PH_Y = W - 150 - PH_W, (H - PH_H) // 2
RADIUS = 62
LEFT_X, LEFT_W = 120, 1060
TIMING = {"ff_out": 1.3, "rt_lead": 0.4, "rt_tail": 1.3, "intro": 2.0, "outro": 3.0}

BG0, BG1 = (9, 12, 22), (20, 27, 45)
FG, DIM, ACCENT, LIVE, GOOD = (245, 247, 250), (148, 160, 184), (96, 165, 250), (239, 68, 68), (52, 211, 153)
AMBER = (251, 191, 36)
# evidence layout: timeline bar, evidence frame below it
TL_X0, TL_X1, TL_Y = LEFT_X, LEFT_X + LEFT_W, 662
EV_MAXW = 640
SF = "/System/Library/Fonts/SFNS.ttf"
_fonts = {}


def font(size, weight="Regular"):
    if (size, weight) not in _fonts:
        f = ImageFont.truetype(SF, size)
        f.set_variation_by_name(weight)
        _fonts[size, weight] = f
    return _fonts[size, weight]


_bg = None


def background():
    global _bg
    if _bg is None:
        _bg = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(_bg)
        for y in range(H):
            a = y / H
            d.line([(0, y), (W, y)], fill=tuple(round(BG0[i] * (1 - a) + BG1[i] * a) for i in range(3)))
    return _bg.copy()


def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def phone_frame(im):
    d = ImageDraw.Draw(im)
    b = 12
    d.rounded_rectangle([PH_X - b, PH_Y - b, PH_X + PH_W + b, PH_Y + PH_H + b], RADIUS + b, fill=(30, 34, 44),
                        outline=(70, 76, 92), width=2)


def chip(d, x, y, mode, label):
    f = font(26, "Semibold")
    tw = d.textlength(label, font=f)
    d.rounded_rectangle([x, y, x + 60 + tw + 26, y + 54], 27, fill=(32, 40, 62))
    cx, cy = x + 33, y + 27
    if mode == "ff":
        for dx in (-11, 3):
            d.polygon([(cx + dx, cy - 10), (cx + dx + 13, cy), (cx + dx, cy + 10)], fill=ACCENT)
    else:
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=LIVE)
    d.text((x + 60, y + 12), label, font=f, fill=FG)


def scene_card(sc, n, total, mode, label, footer, question=None, answer=None, stats=None):
    """The left panel. `answer` is the visible part of the answer (words revealed so far)."""
    im = background()
    phone_frame(im)
    d = ImageDraw.Draw(im)
    d.text((LEFT_X, 64), "EfficientStream", font=font(36, "Bold"), fill=FG)
    d.text((LEFT_X, 110), "Streaming video question answering, on the phone", font=font(24), fill=DIM)
    y = 215
    d.text((LEFT_X, y), f"{n:02d} / {total:02d}", font=font(24, "Semibold"), fill=ACCENT)
    y += 40
    for line in wrap(d, sc["title"], font(64, "Bold"), LEFT_W):
        d.text((LEFT_X, y), line, font=font(64, "Bold"), fill=FG)
        y += 76
    y += 6
    for line in wrap(d, sc["subtitle"], font(30), LEFT_W):
        d.text((LEFT_X, y), line, font=font(30), fill=DIM)
        y += 40
    y += 26
    chip(d, LEFT_X, y, mode, label)
    y += 54 + 44
    if question:
        fq = font(36, "Semibold")
        for line in wrap(d, f"“{question}”", fq, LEFT_W):
            d.text((LEFT_X, y), line, font=fq, fill=FG)
            y += 46
        y += 14
    if answer:
        fa = font(30)
        lines = wrap(d, answer, fa, LEFT_W - 22)
        d.line([(LEFT_X, y + 4), (LEFT_X, y + 4 + 40 * len(lines) - 8)], fill=ACCENT, width=4)
        for line in lines:
            d.text((LEFT_X + 22, y), line, font=fa, fill=(214, 220, 232))
            y += 40
        y += 12
    if stats:
        d.text((LEFT_X + 22, y), stats, font=font(26, "Semibold"), fill=GOOD)
        y += 36
    ff = font(20)
    flines = wrap(d, footer, ff, LEFT_W)
    fy = H - 40 - 28 * len(flines)
    if y > fy - 10:
        print(f"  warning: scene {n} text reaches y={y} (footer at {fy})")
    for line in flines:
        d.text((LEFT_X, fy), line, font=ff, fill=DIM)
        fy += 28
    return im


def title_card(lines, footer):
    im = background()
    d = ImageDraw.Draw(im)
    heights = []
    for text, size, weight, _ in lines:
        heights.append(len(wrap(d, text, font(size, weight), 1500)) * round(size * 1.3) + 26)
    y = (H - sum(heights)) // 2 - 20
    for text, size, weight, col in lines:
        f = font(size, weight)
        for line in wrap(d, text, f, 1500):
            d.text(((W - d.textlength(line, font=f)) / 2, y), line, font=f, fill=tuple(col))
            y += round(size * 1.3)
        y += 26
    f = font(22)
    d.text(((W - d.textlength(footer, font=f)) / 2, H - 80), footer, font=f, fill=DIM)
    return im


def grab_frame(video, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(raw)).convert("RGB")


def evidence_frame(ev, base, maxw, maxh):
    """The evidence frame, aspect-fit in maxw x maxh: dimmed outside the box, the box outlined and labelled."""
    im = grab_frame(base / ev["video"], ev["t"])
    s = min(maxw / im.width, maxh / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).convert("RGBA")
    w, h = im.size
    bx = [ev["box"][0] * w + 2, ev["box"][1] * h + 2, ev["box"][2] * w - 3, ev["box"][3] * h - 3]
    hole = Image.new("L", im.size, 0)
    ImageDraw.Draw(hole).rounded_rectangle(bx, 12, fill=255)
    shade = Image.new("RGBA", im.size, (0, 0, 0, 0))
    shade.putalpha(ImageChops.subtract(Image.new("L", im.size, 150), hole))
    im = Image.alpha_composite(im, shade)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(bx, 12, outline=AMBER, width=4)
    f = font(22, "Semibold")
    pw, ph = d.textlength(ev["label"], font=f) + 28, 36
    if bx[1] - ph - 8 >= 8:  # label above the box, else below, else inside its top
        px, py = bx[0], bx[1] - ph - 8
    elif bx[3] + ph + 8 <= h - 8:
        px, py = bx[0], bx[3] + 8
    else:
        px, py = bx[0] + 10, bx[1] + 10
    px = min(max(px, 8), w - pw - 8)
    d.rounded_rectangle([px, py, px + pw, py + ph], ph // 2, fill=AMBER)
    d.text((px + 14, py + ph / 2), ev["label"], font=f, fill=(24, 24, 24), anchor="lm")
    corners = Image.new("L", im.size, 0)
    ImageDraw.Draw(corners).rounded_rectangle([0, 0, w - 1, h - 1], 14, fill=255)
    im.putalpha(ImageChops.multiply(im.getchannel("A"), corners))
    return im


def tl_x(t, now):
    return TL_X0 + (TL_X1 - TL_X0) * min(max(t / now, 0.0), 1.0)


def ago(s):
    s = round(s)
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def timeline(d, now, head, mode, ev=None, show_ev=False, tl_y=TL_Y):
    """The stream from 0:00 to `now` (the question), filled up to `head`; ticks for the frames indexed every 2 s.
    Fast-forward: a moving head. Real time: the "now" marker and, with `show_ev`, the evidence marker."""
    TL_Y = tl_y
    d.rounded_rectangle([TL_X0, TL_Y, TL_X1, TL_Y + 8], 4, fill=(36, 44, 66))
    xh = tl_x(head, now)
    if xh > TL_X0 + 8:
        d.rounded_rectangle([TL_X0, TL_Y, xh, TL_Y + 8], 4, fill=(66, 108, 196))
    k = 0
    while 2 * k <= head + 1e-6:
        x = tl_x(2 * k, now)
        d.line([(x, TL_Y - 8), (x, TL_Y - 3)], fill=(96, 120, 168), width=1)
        k += 1
    small = font(20)
    d.text((TL_X0, TL_Y + 20), "0:00", font=small, fill=DIM)
    current = ev is not None and now - ev["t"] < 1.5
    if mode == "ff":
        d.text((TL_X0, TL_Y - 42), "Indexing the stream  ·  1 frame every 2 s", font=font(22), fill=DIM)
        if ev is not None and head >= ev["t"] and not current:
            xe = tl_x(ev["t"], now)
            d.ellipse([xe - 6, TL_Y + 4 - 6, xe + 6, TL_Y + 4 + 6], fill=AMBER)
        d.ellipse([xh - 8, TL_Y + 4 - 8, xh + 8, TL_Y + 4 + 8], fill=FG)
        d.text((min(max(xh, TL_X0 + 90), TL_X1 - 30), TL_Y + 20), mmss(head), font=font(20, "Semibold"),
               fill=FG, anchor="ma")
        return
    d.text((TL_X0, TL_Y - 42), "The stream so far", font=font(22), fill=DIM)
    d.text((TL_X1, TL_Y - 16), f"now  ·  {mmss(now)}", font=font(22, "Semibold"), fill=FG, anchor="rb")
    if show_ev and current:
        d.ellipse([TL_X1 - 14, TL_Y + 4 - 14, TL_X1 + 14, TL_Y + 4 + 14], outline=AMBER, width=4)
    d.ellipse([TL_X1 - 9, TL_Y + 4 - 9, TL_X1 + 9, TL_Y + 4 + 9], fill=LIVE)
    if show_ev and not current:
        xe = tl_x(ev["t"], now)
        d.ellipse([xe - 11, TL_Y + 4 - 11, xe + 11, TL_Y + 4 + 11], fill=AMBER, outline=BG1, width=3)


def ff_card(sc, n, total, label, footer, now, head):
    """Fast-forward: the scene's title card plus the stream timeline filling up."""
    im = scene_card(sc, n, total, "ff", label, footer)
    timeline(ImageDraw.Draw(im), now, head, "ff", sc.get("evidence"))
    return im


RT_TOP = 238  # real-time cards: where the question starts


def rt_layout(question, answer, stats):
    """Real-time cards of a scene: the timeline and the evidence frame go just below the scene's final text (the same
    place for the whole scene). Returns (timeline y, evidence frame y, evidence frame max height)."""
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    y = RT_TOP + 44 * len(wrap(d, f"\u201c{question}\u201d", font(34, "Semibold"), LEFT_W)) + 10
    y += 36 * len(wrap(d, answer, font(27), LEFT_W - 22)) + 8 + 32
    tl_y = max(y + 72, 470)
    ev_y = tl_y + 50
    return tl_y, ev_y, min(H - 35 - ev_y, 500)


def rt_card(sc, n, total, question, answer, stats, now, ev_img, show_ev, tl_y, ev_y):
    """Real time: compact header, the question and answer, the timeline, and (from the first answer words) the
    evidence marker joined to the evidence frame."""
    ev = sc["evidence"]
    im = background()
    phone_frame(im)
    d = ImageDraw.Draw(im)
    d.text((LEFT_X, 44), "EfficientStream", font=font(28, "Bold"), fill=FG)
    num = f"{n:02d} / {total:02d}"
    d.text((LEFT_X, 130), num, font=font(26, "Semibold"), fill=ACCENT, anchor="ls")
    d.text((LEFT_X + d.textlength(num, font=font(26, "Semibold")) + 22, 130), sc["title"], font=font(42, "Bold"),
           fill=FG, anchor="ls")
    chip(d, LEFT_X, 152, "rt", "Live  ·  real time, typed and answered on the phone")
    y = RT_TOP
    if question:
        fq = font(34, "Semibold")
        for line in wrap(d, f"“{question}”", fq, LEFT_W):
            d.text((LEFT_X, y), line, font=fq, fill=FG)
            y += 44
        y += 10
    if answer:
        fa = font(27)
        lines = wrap(d, answer, fa, LEFT_W - 22)
        d.line([(LEFT_X, y + 4), (LEFT_X, y + 4 + 36 * len(lines) - 8)], fill=ACCENT, width=4)
        for line in lines:
            d.text((LEFT_X + 22, y), line, font=fa, fill=(214, 220, 232))
            y += 36
        y += 8
    if stats:
        d.text((LEFT_X + 22, y), stats, font=font(24, "Semibold"), fill=GOOD)
        y += 32
    if y > tl_y - 48:
        print(f"  warning: scene {n} text reaches y={y}, the timeline caption starts at {tl_y - 48}")
    timeline(d, now, now, "rt", ev, show_ev, tl_y)
    if not show_ev:
        return im
    current = now - ev["t"] < 1.5
    xe = TL_X1 if current else tl_x(ev["t"], now)
    fw, fh = ev_img.size
    right = xe > (TL_X0 + TL_X1) / 2
    fx = TL_X1 - fw if right else TL_X0
    im.paste(ev_img, (round(fx), ev_y), ev_img)
    xc = min(max(xe, fx + 30), fx + fw - 30)
    d.line([(xe, tl_y + 16), (xe, ev_y - 14), (xc, ev_y - 14), (xc, ev_y)], fill=AMBER, width=3, joint="curve")
    tx = TL_X0 if right else fx + fw + 32
    tw = (fx - 32 - TL_X0) if right else (TL_X1 - tx)
    ty = ev_y + 6
    d.text((tx, ty), "EVIDENCE", font=font(22, "Semibold"), fill=AMBER)
    ty += 38
    big, sub = (("The current frame", f"{mmss(now)}, as the question is asked") if current else
                (f"{mmss(ev['t'])} into the stream", f"{ago(now - ev['t'])} before the question"))
    for line in wrap(d, big, font(34, "Bold"), tw):
        d.text((tx, ty), line, font=font(34, "Bold"), fill=FG)
        ty += 44
    ty += 4
    for line in wrap(d, sub, font(26), tw):
        d.text((tx, ty), line, font=font(26), fill=DIM)
        ty += 34
    return im


def mask():
    m = Image.new("L", (PH_W, PH_H), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, PH_W - 1, PH_H - 1], RADIUS, fill=255)
    return m


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise SystemExit(p.stderr[-2000:])


def phone_piece(take, card_args, maskp, start, dur, speed, out_dur, out):
    f = (f"[1:v]setpts=(PTS-STARTPTS)/{speed:.4f},fps={FPS},"
         f"scale={PH_W}:{PH_H}[p];[2:v]format=gray[m];[p][m]alphamerge[pm];"
         f"[0:v][pm]overlay={PH_X}:{PH_Y}:eof_action=repeat,format=yuv420p[v]")
    run(["ffmpeg", "-y", *card_args, "-hwaccel", "videotoolbox", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", take, "-loop", "1", "-i", maskp, "-filter_complex", f, "-map", "[v]", "-t", f"{out_dur:.3f}",
         "-r", str(FPS), "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-an", out])


def still_piece(card, dur, out):
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", card, "-t", f"{dur:.3f}",
         "-vf", "format=yuv420p", "-r", str(FPS), "-c:v", "libx264", "-crf", "16", "-an", out])


def parse_take(rec_log, console_log, offset=0.0):
    """`offset`: screen-to-recording latency (and clock difference), measured with sync.py."""
    rec0 = float(re.search(r"first frame wall ([\d.]+)", Path(rec_log).read_text()).group(1)) - offset
    segs = []
    for line in Path(console_log).read_text().splitlines():
        m = re.match(r".*ESDEMO ([\d.]+) (\w+) ?(.*)", line)
        if not m:
            continue
        t, kind, rest = float(m.group(1)) - rec0, m.group(2), m.group(3)
        if kind == "segment":
            segs.append({"start": t, "questions": []})
        elif kind == "type":
            segs[-1]["questions"].append({"type": t, "text": rest})
        elif kind == "send":
            q = segs[-1]["questions"][-1]
            q["send"], q["video_t"] = t, float(rest.split()[-1])
        elif kind == "answered":
            q = segs[-1]["questions"][-1]
            if "answered" in q:
                continue
            a = re.match(r"ttft ([\d.]+) total ([\d.]+) ([\d.]+) tok/s \| (.*)", rest)
            q.update(answered=t, ttft=float(a.group(1)), total=float(a.group(2)), tps=float(a.group(3)),
                     answer=a.group(4))
    return segs


def mmss(t):
    t = int(t + 0.5)  # nearest second, as the app shows times
    return f"{t // 60}:{t % 60:02d}"


def main(scenes_json, out):
    cfg = json.loads(Path(scenes_json).read_text())
    tm = {**TIMING, **cfg.get("timing", {})}
    base = Path(os.environ.get("ES_DEMO", "."))
    takes = {k: (str(base / v["video"]),
                 parse_take(base / v["rec_log"], base / v["console_log"], v.get("offset", 0.0)))
             for k, v in cfg["takes"].items()}
    footer = cfg["footer"]
    tmp = Path(tempfile.mkdtemp())
    mask().save(tmp / "mask.png")
    pieces = []
    title_card(cfg["intro"], cfg.get("title_footer", footer)).save(tmp / "intro.png")
    still_piece(str(tmp / "intro.png"), tm["intro"], str(tmp / "p00.mp4"))
    pieces.append(tmp / "p00.mp4")
    scenes = cfg["scenes"]
    total = 0.0
    for i, sc in enumerate(scenes):
        take, segs = takes[sc["take"]]
        sg = segs[sc["segment"]]
        q = sg["questions"][sc.get("question", 0)]
        n = i + 1
        ff_a, ff_b = sg["start"] + 1.0, q["type"] - tm["rt_lead"]
        rt_a, rt_b = ff_b, q["answered"] + sc.get("tail", tm["rt_tail"])
        ff_speed = (ff_b - ff_a) / tm["ff_out"]
        ev, now = sc.get("evidence"), q["video_t"]
        watch = f"Watching  ·  {mmss(now)} of video, sped up"
        p = tmp / f"p{n}ff.mp4"
        if ev:
            # fast-forward with the timeline filling: from the video position at ff_a (the app plays at `rate`x)
            # to the one at ff_b (real time by then, from a few seconds before the typing)
            # the app plays at `rate`x until 3 s of video before the typing, then in real time (DemoModel)
            rate = cfg.get("rate", 4)
            v_b = now - (q["send"] - ff_b)
            v_slow = now - (q["send"] - q["type"]) - 3.0
            t_slow = ff_b - (v_b - v_slow)
            t_load = t_slow - v_slow / rate
            fdir = tmp / f"s{n}ff"
            fdir.mkdir()
            nff = round(tm["ff_out"] * FPS)
            for k in range(nff):
                t = ff_a + (ff_b - ff_a) * k / max(nff - 1, 1)
                head = min(max((t - t_load) * rate, 0.0), v_slow) if t < t_slow else v_slow + (t - t_slow)
                ff_card(sc, n, len(scenes), watch, footer, now, head).save(fdir / f"f_{k:05d}.png")
            card_args = ["-framerate", str(FPS), "-i", str(fdir / "f_%05d.png")]
        else:
            # fast-forward: a static card
            card = tmp / f"s{n}ff.png"
            scene_card(sc, n, len(scenes), "ff", watch, footer).save(card)
            card_args = ["-loop", "1", "-framerate", str(FPS), "-i", str(card)]
        phone_piece(take, card_args, str(tmp / "mask.png"), ff_a, ff_b - ff_a, ff_speed, tm["ff_out"], str(p))
        pieces.append(p)
        # real time: one card per frame (identical states rendered once, linked)
        fdir = tmp / f"s{n}rt"
        fdir.mkdir()
        words = q["answer"].split()
        a0, a1 = q["send"] + q["ttft"], q["answered"]
        stats = f"First words {q['ttft']:.2f} s  ·  {q['tps']:.0f} tokens/s  ·  on the phone"
        if ev:
            tl_y, ev_y, ev_h = rt_layout(q["text"], q["answer"], stats)
            ev_img = evidence_frame(ev, base, EV_MAXW, ev_h)
        rendered = {}
        nframes = round((rt_b - rt_a) * FPS)
        for k in range(nframes):
            t = rt_a + k / FPS
            shown_q = q["text"] if t >= q["send"] else None
            nw = 0 if t < a0 else len(words) if t >= a1 else max(1, round(len(words) * (t - a0) / (a1 - a0)))
            key = (shown_q is not None, nw, t >= a1, ev is not None and t >= a0)
            if key not in rendered:
                path = fdir / f"state_{len(rendered):04d}.png"
                answer, done = " ".join(words[:nw]) or None, stats if t >= a1 else None
                if ev:
                    card = rt_card(sc, n, len(scenes), shown_q, answer, done, now, ev_img, t >= a0, tl_y, ev_y)
                else:
                    card = scene_card(sc, n, len(scenes), "rt", "Live  ·  real time, typed and answered on the phone",
                                      footer, shown_q, answer, done)
                card.save(path)
                rendered[key] = path
            os.link(rendered[key], fdir / f"f_{k:05d}.png")
        p = tmp / f"p{n}rt.mp4"
        phone_piece(take, ["-framerate", str(FPS), "-i", str(fdir / "f_%05d.png")], str(tmp / "mask.png"),
                    rt_a, rt_b - rt_a, 1.0, nframes / FPS, str(p))
        pieces.append(p)
        total += tm["ff_out"] + nframes / FPS
        print(f"scene {n} {sc['title']}: watching {ff_b - ff_a:.1f}s -> {tm['ff_out']}s, real time "
              f"{rt_b - rt_a:.1f}s ({len(rendered)} caption states) | {q['text']}")
    title_card(cfg["outro"], cfg.get("title_footer", footer)).save(tmp / "outro.png")
    still_piece(str(tmp / "outro.png"), tm["outro"], str(tmp / "p99.mp4"))
    pieces.append(tmp / "p99.mp4")
    (tmp / "list.txt").write_text("".join(f"file '{p}'\n" for p in pieces))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-c", "copy",
         "-movflags", "+faststart", out])
    print(f"wrote {out}: {tm['intro'] + total + tm['outro']:.1f} s ({tmp})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
