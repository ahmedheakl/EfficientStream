"""Cuts phone screen recordings of the app's demo modes into one presentation video in the style of a paper demo:
light theme, one section per kind of interaction (the question at the beginning: proactive response; in the middle:
perception; at the end: memory), each example playing in real time (the phone's time; never sped up in the edit).

Each example: on the left the source video, large, at the position the phone is playing (SOURCE time and playback
speed badges, a progress bar with the question and the responses marked); in the middle the question as it is typed
and the model's response as it arrives; on the right the phone itself (the screen recording, live). Under the video:
the per-second checks of the proactive mode (grey: silent, green: it spoke up), or the stored frames most similar to
a memory question (what the retrieval brought back, from the app's log).

Times come from the app's demo logs (ESQDEMO: -model qwen -esqwendemo; ESDEMO: -esdemo), including the playback
position four times a second, and record_iphone's log; all corrected by the screen-to-recording offset
(sync_offset.py).

ES_DEMO=<folder with the takes and logs> python make_showcase.py showcase.json out.mp4
(macOS: the text uses the system font, SF Pro. Sources are resolved relative to ES_DEMO.)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
SF = "/System/Library/Fonts/SFNS.ttf"
SF_MONO = "/System/Library/Fonts/SFNSMono.ttf"

# layout: video | question and answer | phone
VX, VY, VW, VH = 56, 178, 1000, 563
CX, CW = 1088, 404
COL_BOTTOM = 1040
PH_H = 744
PH_W = round(1206 * PH_H / 2622) // 2 * 2
PH_X, PH_Y = W - 64 - PH_W, 178
PH_R = 46

BG0, BG1 = (247, 249, 252), (232, 239, 248)
INK, DIM, FAINT = (15, 23, 42), (100, 116, 139), (203, 213, 225)
CARD, LINE = (255, 255, 255), (226, 232, 240)
GREEN, GREEN_BG = (5, 150, 105), (236, 253, 245)
Q_BG, Q_LINE, Q_INK = (254, 249, 231), (252, 211, 77), (120, 53, 15)
SECTION = {"proactive": (37, 99, 235), "perception": (217, 119, 6), "memory": (13, 148, 136), "all": (124, 58, 237)}
SILENT = (148, 163, 184)

_fonts = {}


def font(size, weight="Regular", mono=False):
    k = (size, weight, mono)
    if k not in _fonts:
        f = ImageFont.truetype(SF_MONO if mono else SF, size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _fonts[k] = f
    return _fonts[k]


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


def mmss(t):
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def mmss_app(t):
    """As the app shows stream time (rounded to the nearest second)."""
    s = int(round(max(0.0, t)))
    return f"{s // 60}:{s % 60:02d}"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


# ---------------------------------------------------------------- logs

def read_take(base, take, offset):
    """Segments of a take's demo log, times in seconds of the recording."""
    rec0 = float(re.search(r"first frame wall ([\d.]+)", (base / take["rec_log"]).read_text()).group(1)) - offset
    segs, pos = [], []
    for line in (base / take["console_log"]).read_text().splitlines():
        m = re.match(r".*ES(?:Q)?DEMO ([\d.]+) (\w+) ?(.*)", line)
        if not m:
            continue
        t, kind, rest = float(m.group(1)) - rec0, m.group(2), m.group(3)
        if kind == "pos":
            p, rate, playing = rest.split()
            pos.append((t, float(p), float(rate), playing == "1"))
            continue
        if kind == "segment":
            segs.append({"t": t, "video": rest.split()[-1], "updates": [], "questions": [], "pos": []})
            continue
        if not segs:
            continue
        s = segs[-1]
        if kind == "caption":
            c = re.match(r"(\d+) ([\d.]+) (\d+) (.*)", rest)
            m = re.match(r"\s*</(Observe|Summary)>\s*(.*)", c.group(4))
            s.setdefault("captions", []).append({"t": t, "second": int(c.group(1)), "seconds": float(c.group(2)),
                                                 "kind": m.group(1).lower() if m else "silence",
                                                 "text": m.group(2).strip() if m else ""})
            continue
        if kind == "update":
            u = re.match(r"(\d+) ([\d.]+) (\d+) (.*)", rest)
            s["updates"].append({"t": t, "second": int(u.group(1)), "seconds": float(u.group(2)),
                                 "tokens": int(u.group(3)), "answer": u.group(4)})
        elif kind == "type" and take["kind"] == "qwen":
            s["prompt"], s["prompt_t"] = rest, t
        elif kind in ("ask", "type"):
            s["questions"].append({"text": rest, "type_t": t})
        elif kind == "send":
            q = s["questions"][-1]
            q["send_t"], q["send_pos"] = t, float(rest.split()[-1])
        elif kind == "answered":
            q = next((q for q in s["questions"] if "send_t" in q and "answer" not in q), None)
            if q is None:
                continue
            a = re.match(r"ttft ([\d.]+) total ([\d.]+) (.*?) \| (.*)", rest)
            q.update(ttft=float(a.group(1)), total=float(a.group(2)), answer=a.group(4).strip(),
                     stats=a.group(3), done_t=t)
        elif kind == "timing":
            q = [q for q in s["questions"] if "answer" in q][-1]
            m = re.match(r"memory (\d+) events", rest)
            q["memory"] = int(m.group(1)) if m else 0
            m = re.search(r"(\d+) prompt tokens", rest)
            q["prompt_tokens"] = int(m.group(1)) if m else None
        elif kind == "context":
            q = [q for q in s["questions"] if "answer" in q][-1]
            m = re.match(r"(\d+) (\d+) (\d+) tokens (\d+) frames(?: (\d+) behind (\w+))?", rest)
            q["context"] = tuple(int(x) for x in m.groups()[:4])
            q["behind"] = int(m.group(5) or 0)
            q["stream"] = m.group(6) or "caption"
        elif kind == "recalled":
            q = [q for q in s["questions"] if "answer" in q][-1]
            q["recalled"] = [float(x) for x in rest.split("|")[0].split()]
        elif kind in ("start", "play", "pause"):
            s.setdefault(kind, t)
    # playback positions, per segment
    for i, s in enumerate(segs):
        end = segs[i + 1]["t"] if i + 1 < len(segs) else 1e12
        s["end_t"] = end
        s["pos"] = [p for p in pos if s["t"] <= p[0] < end]
    return segs


class Playback:
    """The source position the phone shows at recording time t (from the 4 Hz position log)."""

    def __init__(self, samples):
        # keep the samples once playback has started (the position resets while the video is loaded)
        self.s = samples
        self.t = [p[0] for p in samples]

    def at(self, t):
        i = bisect_right(self.t, t) - 1
        if i < 0:
            return self.s[0][1], self.s[0][2], self.s[0][3]
        t0, p0, r0, on0 = self.s[i]
        if i + 1 >= len(self.s):
            return p0 + ((t - t0) * r0 if on0 else 0), r0, on0
        t1, p1, _, _ = self.s[i + 1]
        if not on0 or p1 < p0:
            return p0, r0, on0
        return p0 + (p1 - p0) * min(1.0, (t - t0) / max(1e-6, t1 - t0)), r0, on0


# ---------------------------------------------------------------- drawing

def gradient(w, h, c0, c1):
    a = np.linspace(0, 1, h)[:, None, None]
    g = np.array(c0)[None, None, :] * (1 - a) + np.array(c1)[None, None, :] * a
    return Image.fromarray(np.repeat(g, w, axis=1).astype(np.uint8))


_bg = None


def background():
    global _bg
    if _bg is None:
        _bg = gradient(W, H, BG0, BG1)
    return _bg.copy()


def shadow(im, box, radius, blur=18, alpha=40):
    x0, y0, x1, y1 = [round(v) for v in box]
    pad = 3 * blur
    w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad + 6
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([pad, pad + 6, pad + x1 - x0, pad + 6 + y1 - y0], radius, fill=alpha)
    m = m.filter(ImageFilter.GaussianBlur(blur))
    im.paste(Image.new("RGB", (w, h), (30, 41, 59)), (x0 - pad, y0 - pad), m)


def top_bar(d, right):
    d.rounded_rectangle([56, 34, 92, 70], 9, fill=SECTION["proactive"])
    d.polygon([(68, 43), (68, 61), (83, 52)], fill=(255, 255, 255))
    d.text((106, 33), "EfficientStream", font=font(30, "Bold"), fill=INK)
    f = font(22, "Medium")
    d.text((W - 64 - d.textlength(right, font=f), 40), right, font=f, fill=DIM)


def section_header(d, key, name, subtitle, y=100):
    c = SECTION[key]
    d.ellipse([58, y + 13, 76, y + 31], fill=c)
    f = font(34, "Bold")
    d.text((90, y), name, font=f, fill=c)
    x = 90 + d.textlength(name, font=f)
    if subtitle:
        d.text((x + 14, y), "—  " + subtitle, font=font(34, "Semibold"), fill=INK)


def pill(d, x, y, text, fg, bg, f, pad=12, h=34, anchor="left"):
    tw = d.textlength(text, font=f)
    if anchor == "right":
        x -= tw + 2 * pad
    d.rounded_rectangle([x, y, x + tw + 2 * pad, y + h], h // 2, fill=bg)
    d.text((x + pad, y + (h - f.size) / 2 - 2), text, font=f, fill=fg)
    return x + tw + 2 * pad


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    return m


def fit(frame_wh, box_wh):
    fw, fh = frame_wh
    bw, bh = box_wh
    s = min(bw / fw, bh / fh)
    return round(fw * s) // 2 * 2, round(fh * s) // 2 * 2


class Frames:
    """Decoded frames of a video from `start`, read in order (ffmpeg pipe), scaled to w x h."""

    def __init__(self, path, start, w, h, fps=None, duration=None):
        self.w, self.h = w, h
        vf = (f"fps={fps}," if fps else "") + f"scale={w}:{h}:flags=bicubic,format=rgb24"
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0, start):.3f}", *(["-t", f"{duration:.3f}"] if duration else []),
               "-i", path, "-vf", vf, "-f", "rawvideo", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.rate = fps or float(eval(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=avg_frame_rate", "-of", "csv=p=0", path]).strip()))
        self.start = max(0, start)
        self.n = -1
        self.cur = None

    def read(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) < self.w * self.h * 3:
            return False
        self.cur = np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)
        self.n += 1
        return True

    def at(self, t):
        """The frame showing at source time t (t never decreases between calls)."""
        while self.cur is None or self.start + (self.n + 1) / self.rate <= t + 1e-6:
            if not self.read():
                break
        return self.cur

    def close(self):
        self.p.stdout.close()
        self.p.kill()


def size_of(path):
    w, h = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "csv=p=0", path]).strip().split(",")[:2]
    return int(w), int(h)


def grab(path, t, w):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.2f}", "-i", path, "-frames:v", "1",
                          "-vf", f"scale={w}:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    import io
    return Image.open(io.BytesIO(out.stdout)).convert("RGB")


# ---------------------------------------------------------------- cards

def card(im, box, fill=CARD, outline=LINE, radius=18, shade=True):
    if shade:
        shadow(im, box, radius, blur=14, alpha=26)
    ImageDraw.Draw(im).rounded_rectangle(box, radius, fill=fill, outline=outline, width=2)


def text_block(d, x, y, text, f, fill, width, max_lines=None, line_h=None):
    lines = wrap(d, text, f, width)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + "…"
    lh = line_h or round(f.size * 1.28)
    for line in lines:
        d.text((x, y), line, font=f, fill=fill)
        y += lh
    return y


def question_card(im, y, label, text, cursor=False, dim=False, draw=True):
    d = ImageDraw.Draw(im if draw else Image.new("RGB", (8, 8)))
    f = font(25, "Semibold")
    lines = wrap(d, text, f, CW - 48) or [""]
    h = 52 + len(lines) * 32 + 12
    if not draw:
        return y + h
    card(im, [CX, y, CX + CW, y + h], fill=Q_BG, outline=Q_LINE)
    d.text((CX + 20, y + 16), label, font=font(15, "Bold"), fill=(180, 83, 9))
    yy = y + 46
    for line in lines:
        d.text((CX + 20, yy), line, font=f, fill=Q_INK if not dim else (146, 100, 60))
        yy += 32
    if cursor:
        x = CX + 22 + d.textlength(lines[-1], font=f)
        d.rectangle([x, yy - 30, x + 2, yy - 4], fill=Q_INK)
    return y + h


def split_first(text):
    """The first sentence and the rest."""
    m = re.match(r"(.+?[.!?])\s+(.*)", text, re.S)
    return (m.group(1), m.group(2)) if m else (text, "")


def answer_card(im, y, status, status_color, text, sub=None, max_lines=8, faded=False, draw=True):
    """The model's answer: large when short; otherwise the first sentence in bold and the rest in regular weight."""
    d = ImageDraw.Draw(im if draw else Image.new("RGB", (8, 8)))
    big = len(text) <= 60
    head, rest = (text, "") if big else split_first(text)
    fh = font(30, "Bold") if big else font(23, "Bold")
    fr = font(21, "Regular")
    lines = [(l, fh) for l in wrap(d, head, fh, CW - 40)] if text else []
    lines += [(l, fr) for l in wrap(d, rest, fr, CW - 40)] if rest else []
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = (lines[-1][0].rstrip(".,") + "…", lines[-1][1])
    hs = [round(f.size * 1.3) for _, f in lines] or [36]
    subs = [sub] if isinstance(sub, str) else (sub or [])
    h = 56 + sum(hs) + 30 * len(subs) + (4 if subs else 0) + 8
    if not draw:
        return y + h
    card(im, [CX, y, CX + CW, y + h])
    d.text((CX + 20, y + 16), "EfficientStream", font=font(17, "Bold"), fill=INK)
    pill(d, CX + CW - 16, y + 12, status, (255, 255, 255), status_color, font(14, "Bold"), pad=10, h=26,
         anchor="right")
    yy = y + 50
    for (line, f), lh in zip(lines, hs):
        d.text((CX + 20, yy), line, font=f, fill=(DIM if faded else INK) if f is fh else (51, 65, 85))
        yy += lh
    fs = font(17, "Medium")
    for k, line in enumerate(subs):
        # one line each: cut long ones (an earlier long response) to the card's width
        if d.textlength(line, font=fs) > CW - 40:
            while line and d.textlength(line + "…”", font=fs) > CW - 40:
                line = line[:-1]
            line = line.rstrip(" ,.") + "…" + ("”" if "“" in line else "")
        d.text((CX + 20, yy + 6 + 30 * k), line, font=fs, fill=DIM if k == 0 else SECTION["all"])
    return y + h


# ---------------------------------------------------------------- one example

def phone_bezel(im):
    b = 10
    shadow(im, [PH_X - b, PH_Y - b, PH_X + PH_W + b, PH_Y + PH_H + b], PH_R + b, blur=20, alpha=50)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([PH_X - b, PH_Y - b, PH_X + PH_W + b, PH_Y + PH_H + b], PH_R + b, fill=(28, 31, 38),
                        outline=(80, 86, 100), width=2)
    lab = "The phone, live (screen recording)"
    f = font(18, "Medium")
    d.text((PH_X + PH_W / 2 - d.textlength(lab, font=f) / 2, PH_Y + PH_H + 22), lab, font=f, fill=DIM)


class Example:
    def __init__(self, sec, sc, seg, take_path, base, cfg):
        self.sec, self.sc, self.seg, self.cfg = sec, sc, seg, cfg
        self.key = sec["key"]
        self.take = take_path
        self.src = str(base / sc["source"])
        first_on = next((i for i, p in enumerate(seg["pos"]) if p[3]), 0)
        self.pb = Playback(seg["pos"][first_on:])
        self.prompt = seg.get("prompt")
        self.qs = [q for q in seg["questions"] if "send_t" in q]
        self.updates = seg["updates"]
        self.captions = seg.get("captions")
        # the stretch of the recording to show
        first = self.pb.s[0][0] if self.pb.s else seg["t"]
        if self.prompt:
            a = seg["prompt_t"] - 0.6
        elif sc.get("from") is not None:
            a = self.rec_time(sc["from"])
        else:
            a = first - 0.3
        if sc.get("to") is not None:
            b = self.rec_time(sc["to"])
        elif self.qs and "done_t" in self.qs[-1] and not self.prompt:
            b = self.qs[-1]["done_t"] + sc.get("hold", 3.0)
        else:
            b = seg.get("pause", first + 30) + sc.get("tail", 1.0)
        self.a, self.b = a, min(b, seg["end_t"] - 0.1)
        # stretches fast-forwarded in the edit: [[from, to, speed]] in source seconds (labelled on screen)
        self.ff = [(self.rec_time(f0), self.rec_time(f1), sp) for f0, f1, sp in sc.get("ff", [])]
        dur = float(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
                         "-of", "csv=p=0", self.src]).strip() or 1e9)
        self.clip = (sc.get("clip_from", self.pb.at(max(a, first))[0]), min(dur, sc.get("clip_to", self.pb.at(b)[0])))
        sw, sh = size_of(self.src)
        self.fw, self.fh = fit((sw, sh), (VW, VH))
        self.thumbs = {}

    def rec_time(self, pos):
        for t, p, _, on in self.pb.s:
            if on and p >= pos:
                return t
        return self.pb.s[-1][0]

    def thumb(self, t, w=150):
        if (t, w) not in self.thumbs:
            im = grab(self.src, t + 0.05, w)
            self.thumbs[t, w] = im.resize((w, round(w * self.fh / self.fw)))
        return self.thumbs[t, w]

    def static(self):
        im = background()
        d = ImageDraw.Draw(im)
        top_bar(d, self.cfg["top_right"])
        section_header(d, self.key, self.sec["name"], self.sc["title"])
        d.text((90, 150), self.sc["model"], font=font(19, "Medium"), fill=DIM)
        shadow(im, [VX, VY, VX + VW, VY + VH], 16, blur=20, alpha=46)
        ImageDraw.Draw(im).rounded_rectangle([VX, VY, VX + VW, VY + VH], 16, fill=(10, 12, 18))
        phone_bezel(im)
        return im

    # what is visible at recording time t
    def state(self, t):
        pos, rate, playing = self.pb.at(t)
        st = {"pos": pos, "rate": rate}
        if self.prompt:
            started = self.seg.get("start", 1e12)
            if t < started:
                n = len(self.prompt)
                frac = (t - self.seg["prompt_t"]) / max(0.2, started - 0.35 - self.seg["prompt_t"])
                st["typing"] = self.prompt[:max(0, min(n, round(frac * n)))]
            st["updates"] = [u for u in self.updates if u["t"] <= t]
        qs = []
        for q in self.qs:
            if t < q["type_t"]:
                continue
            e = {"text": q["text"], "pos": q["send_pos"]}
            if t < q["send_t"]:
                n = len(q["text"])
                frac = (t - q["type_t"]) / max(0.2, q["send_t"] - 0.35 - q["type_t"])
                e["typing"] = q["text"][:max(0, min(n, round(frac * n)))]
            elif "answer" in q:
                first = q["send_t"] + q["ttft"]
                done = q["send_t"] + q["total"]
                if t >= done:
                    e["answer"], e["done"] = q["answer"], True
                elif t >= first:
                    words = q["answer"].split()
                    k = max(1, round(len(words) * (t - first) / max(0.05, done - first)))
                    e["answer"] = " ".join(words[:k])
                e["q"] = q
            qs.append(e)
        st["qs"] = qs
        return st

    def draw_dynamic(self, im, st, t):
        d = ImageDraw.Draw(im)
        c = SECTION[self.key]
        # badges over the video
        pill(d, VX + 16, VY + 16, f"SOURCE {mmss_app(st['pos'])}", (255, 255, 255), (15, 23, 42),
             font(17, "Semibold", mono=True), pad=12, h=32)
        r = st["rate"] * st.get("edit", 1.0)
        pill(d, VX + VW - 16, VY + 16, f"{r:.0f}×", (255, 255, 255), (15, 23, 42) if r <= 1.01 else (217, 119, 6),
             font(18, "Bold"), pad=12, h=32, anchor="right")
        if st.get("edit", 1.0) > 1.01:
            self.ff_chip(im, st["edit"])
        # progress bar
        c0, c1 = self.clip
        by = VY + VH + 22
        d.rounded_rectangle([VX, by, VX + VW, by + 8], 4, fill=FAINT)
        frac = min(1, max(0, (st["pos"] - c0) / max(1e-6, c1 - c0)))
        d.rounded_rectangle([VX, by, VX + max(8, VW * frac), by + 8], 4, fill=c)
        d.text((VX, by + 16), mmss(c0), font=font(16, "Medium"), fill=DIM)
        e = mmss(c1)
        d.text((VX + VW - d.textlength(e, font=font(16, "Medium")), by + 16), e, font=font(16, "Medium"), fill=DIM)
        for q in st["qs"]:
            if "typing" in q:
                continue
            x = VX + VW * min(1, max(0, (q["pos"] - c0) / max(1e-6, c1 - c0)))
            d.ellipse([x - 8, by - 4, x + 8, by + 12], fill=Q_LINE, outline=(180, 83, 9), width=2)
        for u in st.get("updates", []):
            if u["answer"].startswith("</Response>"):
                x = VX + VW * min(1, max(0, (u["second"] + 0.5 - c0) / max(1e-6, c1 - c0)))
                d.rounded_rectangle([x - 4, by - 8, x + 4, by + 16], 3, fill=GREEN)
        y = by + 52
        if self.captions is not None:
            if self.prompt:
                y = self.draw_checks(im, st, y, compact=True)
            self.draw_captions(im, st, t, y)
        elif self.prompt:
            self.draw_checks(im, st, y)
        elif self.key == "memory":
            self.draw_recall(im, st, y)
        else:
            self.draw_note(im, st, y)
        self.draw_column(im, st, t)

    def draw_captions(self, im, st, t, y):
        """The caption memory as it is written (one decision per second): its newest entries."""
        d = ImageDraw.Draw(im)
        caps = [c for c in self.captions if c["t"] <= t]
        written = [c for c in caps if c["kind"] != "silence"]
        c = SECTION["memory"]
        d.text((VX, y), "Memory: captions the model writes as it watches", font=font(20, "Semibold"), fill=c)
        if caps:
            s = f"second {mmss(caps[-1]['second'])} {'written' if caps[-1]['kind'] != 'silence' else 'silent'}" \
                f"  ·  {len(written)} captions so far"
            d.text((VX + VW - d.textlength(s, font=font(17, "Medium")), y + 3), s, font=font(17, "Medium"), fill=DIM)
        y += 34
        bottom = 1052
        if not written:
            d.text((VX, y), "Watching… nothing worth writing down yet.", font=font(19, "Medium"), fill=DIM)
            return
        f, fb = font(19, "Regular"), font(19, "Semibold")
        lines = []
        for e in reversed(written):
            text = re.sub(r"^\[[^\]]*\]\s*", "", e["text"])
            head = (f"Summary {mmss(e['second'])}" if e["kind"] == "summary" else mmss(e["second"]))
            lines.append((head, text, e["kind"], t - e["t"]))
        for head, text, kind, age in lines:
            hw = 92 if kind != "summary" else 150
            body = wrap(d, text, f, VW - hw - 10)
            if len(body) > 2:
                body = body[:2]
                body[-1] = body[-1].rstrip(".,") + "…"
            if y + 26 * len(body) > bottom:
                break
            col = INK if age < 3 else (71, 85, 105)
            if age < 1.0:
                d.rounded_rectangle([VX - 8, y - 3, VX + VW, y + 26 * len(body) - 1], 6, fill=(220, 252, 231))
            d.text((VX, y), head, font=fb, fill=c)
            for k, line in enumerate(body):
                d.text((VX + hw, y + 26 * k), line, font=fb if kind == "summary" else f, fill=col)
            y += 26 * len(body) + 4

    def draw_checks(self, im, st, y, compact=False):
        d = ImageDraw.Draw(im)
        ups = st.get("updates", [])
        d.text((VX, y), "Watching: one check per second of video", font=font(20, "Semibold"), fill=INK)
        if ups:
            silent = [u["seconds"] for u in ups if not u["answer"].startswith("</Response>")]
            s = f"{len(ups)} checks · {sum(silent) / max(1, len(silent)):.2f} s per silent check"
            d.text((VX + VW - d.textlength(s, font=font(18, "Medium")), y + 2), s, font=font(18, "Medium"), fill=DIM)
        y += 32 if compact else 36
        n = max(1, int(round(self.clip[1] - self.clip[0])))
        gap = 4 if n <= 40 else 1
        bw = min(28, (VW - gap * (n - 1)) / n)
        bh = 18 if compact else 30
        for i in range(n):
            x = VX + i * (bw + gap)
            sec = int(self.clip[0]) + i
            u = next((u for u in ups if u["second"] == sec), None)
            col = (GREEN if u["answer"].startswith("</Response>") else SILENT) if u else (226, 232, 240)
            d.rounded_rectangle([x, y, x + bw, y + bh], 3 if compact else 5, fill=col)
        if compact:
            return y + bh + 18
        y += 44
        legend = [((148, 163, 184), "silent"), (GREEN, "spoke up")]
        x = VX
        for col, lab in legend:
            d.rounded_rectangle([x, y + 3, x + 16, y + 19], 4, fill=col)
            d.text((x + 24, y), lab, font=font(17, "Medium"), fill=DIM)
            x += 24 + d.textlength(lab, font=font(17, "Medium")) + 28

    def draw_recall(self, im, st, y):
        d = ImageDraw.Draw(im)
        q = next((q for q in reversed(st["qs"]) if q.get("done")), None)
        if not q or not q["q"].get("recalled"):
            stored = int(max(0.0, st["pos"] - self.clip[0]) // 2) + 1
            d.text((VX, y), f"Memory: {stored} frames stored so far", font=font(20, "Semibold"), fill=INK)
            d.text((VX, y + 32), "One frame every 2 s of video, with its MobileCLIP embedding. A question brings back "
                                 "the stored frames closest to it.", font=font(18, "Medium"), fill=DIM)
            return
        d.text((VX, y), "Recalled from memory: the stored frames closest to the question",
               font=font(20, "Semibold"), fill=INK)
        x = VX
        for rt in q["q"]["recalled"][:self.sc.get("recall_n", 4)]:
            th = self.thumb(rt, 150)
            tw, thh = th.size
            m = rounded_mask(tw, thh, 8)
            im.paste(th, (x, y + 36), m)
            ImageDraw.Draw(im).rounded_rectangle([x, y + 36, x + tw, y + 36 + thh], 8, outline=SECTION["memory"],
                                                 width=3)
            pill(ImageDraw.Draw(im), x + 6, y + 42, f"EARLIER · {mmss(rt)}", (255, 255, 255), (15, 23, 42),
                 font(13, "Bold"), pad=7, h=22)
            x += tw + 14

    def draw_note(self, im, st, y):
        d = ImageDraw.Draw(im)
        q = next((q for q in reversed(st["qs"]) if "typing" not in q), None)
        if not q:
            d.text((VX, y), "The model sees one frame per second as the video plays; a question looks at the last "
                            "16 seconds.", font=font(20, "Semibold"), fill=INK)
            return
        qq = q["q"]
        last = int(qq["send_pos"])
        n = int(re.search(r"(\d+) frames", qq.get("stats", "")).group(1)) if "frames" in qq.get("stats", "") else 16
        times = list(range(last - n + 1, last + 1))
        d.text((VX, y), f"What the answer looked at: the last {n} seconds, one frame per second "
                        f"({mmss(times[0])}–{mmss(times[-1])})", font=font(20, "Semibold"), fill=INK)
        tw, gap = 118, 6
        th = round(tw * self.fh / self.fw)
        for i, ft in enumerate(times):
            r, c = divmod(i, 8)
            x, yy = VX + c * (tw + gap), y + 36 + r * (th + gap)
            im.paste(self.thumb(ft, tw), (x, yy), rounded_mask(tw, th, 6))

    def draw_column(self, im, st, t):
        y = VY
        if self.prompt:
            typing = st.get("typing")
            if typing is not None:
                y = question_card(im, y, "TYPING WHAT TO WATCH FOR…", typing, cursor=True) + 14
            else:
                y = question_card(im, y, f"WATCHING FOR · told at {mmss(self.clip[0])}", self.prompt) + 14
                ups = st.get("updates", [])
                rs = [u for u in ups if u["answer"].startswith("</Response>")]
                busy = bool(st["qs"])  # questions too: keep the watch short
                if rs:
                    r = rs[-1]
                    txt = r["answer"][len("</Response>"):].strip()
                    sub = [f"second {mmss(r['second'])} of video · answered in {r['seconds']:.1f} s"]
                    if len(rs) > 1:
                        sub.append("earlier: " + "  ·  ".join(
                            f"{mmss(e['second'])} “{e['answer'][len('</Response>'):].strip()}”" for e in rs[-4:-1]))
                    y = answer_card(im, y, f"SPOKE UP · {mmss(r['second'])}" + (f" · #{len(rs)}" if len(rs) > 1 else ""),
                                    GREEN, txt, sub=sub, max_lines=3 if busy else 8) + 14
                elif ups:
                    u = ups[-1]
                    y = answer_card(im, y, "SILENT", SILENT, "Watching… nothing yet.",
                                    sub=f"second {mmss(u['second'])} checked in {u['seconds']:.2f} s") + 14
                else:
                    y = answer_card(im, y, "WATCHING", SECTION["proactive"], "Watching…") + 14
        qs = st["qs"][-1:] if self.prompt else st["qs"][-2:]
        if not self.prompt and not qs:
            y = question_card(im, y, "QUESTION", "Waiting for a question…", dim=True) + 14
            if self.key == "memory":
                answer_card(im, y, "REMEMBERING", SECTION["memory"],
                            "Writing captions as it watches…" if self.captions is not None
                            else "Watching and storing frames…", faded=True)
            else:
                answer_card(im, y, "WATCHING", (15, 23, 42), "Watching…", faded=True)
            return

        def layout(y, draw, short):
            for i, q in enumerate(qs):
                if "typing" in q:
                    y = question_card(im, y, "TYPING A QUESTION…", q["typing"], cursor=True, draw=draw) + 14
                    continue
                y = question_card(im, y, f"QUESTION · asked at {mmss_app(q['pos'])}", q["text"], draw=draw) + 14
                if "answer" in q:
                    qq = q["q"]
                    sub = None
                    if q.get("done"):
                        sub = [f"first words {qq['ttft']:.2f} s · full answer {qq['total']:.1f} s"]
                        if qq.get("context"):
                            a, b, tok, fr = qq["context"]
                            sub.append(f"looked at the video from {mmss(a)} to {mmss(b + 1)}")
                            if qq.get("memory"):
                                sub.append(f"plus {qq['memory']} captions it wrote earlier")
                    y = answer_card(im, y, "ANSWER" if self.key != "memory" else "FROM MEMORY",
                                    GREEN if self.key != "memory" else SECTION["memory"], q["answer"], sub=sub,
                                    max_lines=3 if short and i < len(qs) - 1 else 9, draw=draw) + 14
                else:
                    y = answer_card(im, y, "THINKING", (217, 119, 6), "…", draw=draw) + 14
            return y

        layout(y, True, layout(y, False, False) > COL_BOTTOM)

    def ff_chip(self, im, speed):
        """Over the video and the phone: this stretch is fast-forwarded in the edit."""
        d = ImageDraw.Draw(im)
        f = font(22, "Bold")
        label = f"Sped up {speed:.0f}× for viewing" + (
            "  ·  still silent, checking every second" if self.key == "proactive" else "  ·  the model keeps watching")
        tw = d.textlength(label, font=f)
        w = tw + 86
        x, y = VX + (VW - w) / 2, VY + VH - 70
        d.rounded_rectangle([x, y, x + w, y + 48], 24, fill=(217, 119, 6))
        cx, cy = x + 34, y + 24
        for dx in (-11, 3):
            d.polygon([(cx + dx, cy - 10), (cx + dx + 13, cy), (cx + dx, cy + 10)], fill=(255, 255, 255))
        d.text((x + 62, y + 10), label, font=f, fill=(255, 255, 255))

    def edit_speed(self, t):
        return next((sp for f0, f1, sp in self.ff if f0 <= t < f1), 1.0)

    def render(self, enc, total_frames_cb=None):
        base = self.static()
        times = []
        t = self.a
        while t < self.b:
            times.append(t)
            t += self.edit_speed(t) / FPS
        n = max(1, len(times))
        p0 = self.pb.at(self.a)[0]
        src = Frames(self.src, p0 - 0.5, self.fw, self.fh)
        ph = Frames(self.take, self.a, PH_W, PH_H, fps=FPS, duration=self.b - self.a + 1)
        pmask = rounded_mask(PH_W, PH_H, PH_R)
        vmask = rounded_mask(VW, VH, 16)
        cache_key, cache_im = None, None
        for t in times:
            st = self.state(t)
            st["edit"] = self.edit_speed(t)
            im = base.copy()
            # the source frame, letterboxed in the video box
            fr = src.at(st["pos"])
            if fr is not None:
                box = Image.new("RGB", (VW, VH), (10, 12, 18))
                box.paste(Image.fromarray(fr), ((VW - self.fw) // 2, (VH - self.fh) // 2))
                im.paste(box, (VX, VY), vmask)
            pf = ph.at(t)
            if pf is not None:
                im.paste(Image.fromarray(pf), (PH_X, PH_Y), pmask)
            self.draw_dynamic(im, st, t)
            enc.stdin.write(im.tobytes())
        src.close()
        ph.close()
        return n


# ---------------------------------------------------------------- cards between examples

def title_card(cfg):
    im = background()
    d = ImageDraw.Draw(im)
    top_bar(d, cfg["top_right"])
    if cfg.get("paper_title"):
        f = font(64, "Bold")
        lines = wrap(d, cfg["paper_title"], f, 1500)
        y = 400 - len(lines) * 84 / 2
        for line in lines:
            d.text((W / 2 - d.textlength(line, font=f) / 2, y), line, font=f, fill=INK)
            y += 84
        y += 40
    else:
        t = cfg["title"]
        f = font(96, "Bold")
        d.text((W / 2 - d.textlength(t, font=f) / 2, 330), t, font=f, fill=INK)
        y = 470
    for line, size, col in cfg["title_lines"]:
        f = font(size, "Semibold")
        d.text((W / 2 - d.textlength(line, font=f) / 2, y), line, font=f, fill=tuple(col))
        y += size + 22
    f = font(22, "Medium")
    s = cfg["title_footer"]
    d.text((W / 2 - d.textlength(s, font=f) / 2, H - 110), s, font=f, fill=DIM)
    return im


def mini_timeline(d, x, y, w, key, k=1.0):
    """A video as a bar, with when the question is asked (?) and where the answer comes from."""
    c = SECTION[key]
    h = round(14 * k)
    d.rounded_rectangle([x, y, x + w, y + h], h // 2, fill=FAINT)
    if key == "all":
        # one stream, a question at the beginning, in the middle and at the end
        d.rounded_rectangle([x, y, x + w, y + h], h // 2, fill=c)
        r, cy = 17 * k, y + h / 2
        f, fl = font(round(22 * k), "Bold"), font(round(17 * k), "Medium")
        for at, lab in ((0.04, "watch for…"), (0.33, "what now?"), (0.93, "what earlier?")):
            qx = x + w * at
            d.ellipse([qx - r, cy - r, qx + r, cy + r], fill=Q_LINE, outline=(180, 83, 9), width=max(2, round(3 * k)))
            d.text((qx - d.textlength("?", font=f) / 2, cy - 15 * k), "?", font=f, fill=(120, 53, 15))
            tw = d.textlength(lab, font=fl)
            d.text((min(max(qx - tw / 2, x), x + w - tw), cy + r + 8 * k), lab, font=fl, fill=DIM)
        return
    q_at, a_at = {"proactive": (0.05, 0.68), "perception": (0.5, 0.5), "memory": (0.95, 0.22)}[key]
    qx, ax = x + w * q_at, x + w * a_at
    cy = y + h / 2
    if key == "proactive":
        d.rounded_rectangle([x, y, qx, y + h], h // 2, fill=c)
        d.line([(qx + 20 * k, cy - 26 * k), (ax - 20 * k, cy - 26 * k)], fill=c, width=max(2, round(3 * k)))
        d.polygon([(ax - 20 * k, cy - 33 * k), (ax - 6 * k, cy - 26 * k), (ax - 20 * k, cy - 19 * k)], fill=c)
    elif key == "perception":
        d.rounded_rectangle([x, y, qx, y + h], h // 2, fill=c)
    else:
        d.rounded_rectangle([x, y, qx, y + h], h // 2, fill=c)
        # an arc from the question back to the earlier moment, its head touching that moment's dot
        p0, p2 = (qx, cy - 22 * k), (ax, cy - 22 * k)
        p1 = ((qx + ax) / 2, cy - 110 * k)
        pts = [((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
                (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]) for u in [i / 60 for i in range(61)]]
        head = 16 * k
        tx, ty = p2[0] - p1[0], p2[1] - p1[1]
        n = (tx * tx + ty * ty) ** 0.5
        tx, ty = tx / n, ty / n
        bx, by = p2[0] - tx * head, p2[1] - ty * head
        d.line([q for q in pts if (q[0] - p2[0]) ** 2 + (q[1] - p2[1]) ** 2 > (head * 0.8) ** 2], fill=c,
               width=max(2, round(3 * k)), joint="curve")
        d.polygon([p2, (bx - ty * head * 0.55, by + tx * head * 0.55), (bx + ty * head * 0.55, by - tx * head * 0.55)],
                  fill=c)
    r = 17 * k
    # the answer's moment
    d.ellipse([ax - r, cy - r, ax + r, cy + r], fill=GREEN if key != "memory" else (255, 255, 255),
              outline=GREEN if key != "memory" else c, width=max(2, round(4 * k)))
    # the question
    if key != "perception":
        d.ellipse([qx - r, cy - r, qx + r, cy + r], fill=Q_LINE, outline=(180, 83, 9), width=max(2, round(3 * k)))
        f = font(round(22 * k), "Bold")
        d.text((qx - d.textlength("?", font=f) / 2, cy - 15 * k), "?", font=f, fill=(120, 53, 15))
    else:
        d.ellipse([qx - r, cy - r, qx + r, cy + r], fill=Q_LINE, outline=GREEN, width=max(2, round(4 * k)))
        f = font(round(22 * k), "Bold")
        d.text((qx - d.textlength("?", font=f) / 2, cy - 15 * k), "?", font=f, fill=(120, 53, 15))
    f = font(round(17 * k), "Medium")
    labels = {"proactive": [(qx, "asked"), (ax, "answered when it happens")],
              "perception": [(qx, "asked, and answered about now")],
              "memory": [(ax, "the moment it recalls"), (qx, "asked")]}[key]
    for lx, lab in labels:
        tw = d.textlength(lab, font=f)
        lx = min(max(lx - tw / 2, x), x + w - tw)
        d.text((lx, cy + r + 8 * k), lab, font=f, fill=DIM)


def overview_card(cfg, highlight=None):
    im = background()
    d = ImageDraw.Draw(im)
    top_bar(d, cfg["top_right"])
    t = cfg["overview_title"]
    f = font(46, "Bold")
    d.text((W / 2 - d.textlength(t, font=f) / 2, 200), t, font=f, fill=INK)
    secs = [s for s in cfg["sections"] if s["key"] in ("proactive", "perception", "memory")]
    n = len(secs)
    cw, gap = 520, 40
    x0 = (W - n * cw - (n - 1) * gap) / 2
    for i, s in enumerate(secs):
        x = x0 + i * (cw + gap)
        c = SECTION[s["key"]]
        box = [x, 310, x + cw, 870]
        card(im, box)
        d = ImageDraw.Draw(im)
        d.ellipse([x + 36, 346, x + 92, 402], fill=c)
        d.text((x + 64 - d.textlength(str(i + 1), font=font(30, "Bold")) / 2, 353), str(i + 1), font=font(30, "Bold"),
               fill=(255, 255, 255))
        d.text((x + 112, 348), s["name"], font=font(40, "Bold"), fill=c)
        d.text((x + 36, 434), s["when"], font=font(28, "Semibold"), fill=INK)
        mini_timeline(d, x + 56, 550, cw - 112, s["key"])
        text_block(d, x + 36, 650, s["blurb"], font(24, "Regular"), (71, 85, 105), cw - 72)
        if s["model"]:
            d.text((x + 36, 820), s["model"], font=font(19, "Medium"), fill=DIM)
    return im


def section_card(cfg, sec):
    im = background()
    d = ImageDraw.Draw(im)
    top_bar(d, cfg["top_right"])
    c = SECTION[sec["key"]]
    f = font(76, "Bold")
    tw = d.textlength(sec["name"], font=f)
    x = W / 2 - (tw + 62) / 2
    d.ellipse([x, 300, x + 40, 340], fill=c)
    d.text((x + 62, 270), sec["name"], font=f, fill=c)
    f2 = font(34, "Semibold")
    d.text((W / 2 - d.textlength(sec["when"], font=f2) / 2, 380), sec["when"], font=f2, fill=INK)
    mini_timeline(d, W / 2 - 400, 520, 800, sec["key"], k=1.4)
    f3 = font(28)
    y = 660
    for line in wrap(d, sec["blurb"], f3, 1400):
        d.text((W / 2 - d.textlength(line, font=f3) / 2, y), line, font=f3, fill=(71, 85, 105))
        y += 40
    f4 = font(22, "Medium")
    if sec["model"]:
        m = "Model: " + sec["model"]
        d.text((W / 2 - d.textlength(m, font=f4) / 2, y + 30), m, font=f4, fill=DIM)
    return im


def outro_card(cfg, stats):
    im = background()
    d = ImageDraw.Draw(im)
    top_bar(d, cfg["top_right"])
    f = font(56, "Bold")
    t = cfg["outro_title"]
    d.text((W / 2 - d.textlength(t, font=f) / 2, 230), t, font=f, fill=INK)
    y = 350
    for name, value in stats:
        f1, f2 = font(30, "Medium"), font(30, "Bold")
        w1 = d.textlength(name, font=f1)
        d.text((W / 2 - 30 - w1, y), name, font=f1, fill=DIM)
        d.text((W / 2 + 10, y), value, font=f2, fill=INK)
        y += 58
    f = font(40, "Bold")
    t = "Thanks for watching"
    d.text((W / 2 - d.textlength(t, font=f) / 2, y + 70), t, font=f, fill=SECTION["proactive"])
    f = font(22, "Medium")
    s = cfg["title_footer"]
    d.text((W / 2 - d.textlength(s, font=f) / 2, H - 110), s, font=f, fill=DIM)
    return im


def measured(examples):
    """The timings shown at the end, averaged over the examples in the video (from the app's logs)."""
    def avg(xs):
        return sum(xs) / max(1, len(xs))
    out = []
    ups = [u for e in examples if e.key == "proactive" for u in e.updates if e.a <= u["t"] <= e.b]
    silent = [u["seconds"] for u in ups if not u["answer"].startswith("</Response>")]
    spoke = [u for u in ups if u["answer"].startswith("</Response>")]
    caps = [c for e in examples if e.key != "all" and e.captions for c in e.captions if e.a <= c["t"] <= e.b]
    if silent:
        out.append(("Watching: one check per second of video", f"{avg(silent):.2f} s per check"))
    if spoke:
        out.append(("When it speaks up", f"{avg([u['seconds'] for u in spoke]):.1f} s "
                                         "for a short alert"))
    if caps:
        out.append(("Memory captions, one decision per second", f"{avg([c['seconds'] for c in caps if c['kind'] == 'silence']):.2f} s "
                                                              f"silent, {avg([c['seconds'] for c in caps if c['kind'] != 'silence']):.1f} s writing"))
    qs = [q for e in examples for q in e.qs if "ttft" in q and q.get("context")]
    if qs:
        out.append(("A question → first words of the answer",
                    f"{avg([q['ttft'] for q in qs]):.2f} s (full answer {avg([q['total'] for q in qs]):.1f} s)"))
    return out


def still(enc, im, seconds):
    b = im.convert("RGB").tobytes()
    n = round(seconds * FPS)
    for _ in range(n):
        enc.stdin.write(b)
    return n


def main(cfg_path, out):
    cfg = json.loads(Path(cfg_path).read_text())
    base = Path(os.environ.get("ES_DEMO", "."))
    takes = {k: read_take(base, v, v.get("offset", cfg.get("offset", 0.0))) for k, v in cfg["takes"].items()}
    only = os.environ.get("ONLY")  # render one example: "section:index"
    plan = []
    for sec in cfg["sections"]:
        exs = []
        for ei, sc in enumerate(sec["examples"]):
            seg = takes[sc["take"]][sc["segment"]]
            exs.append(Example(sec, sc, seg, str(base / cfg["takes"][sc["take"]]["video"]), base, cfg))
        plan.append((sec, exs))
    stats = measured([e for _, exs in plan for e in exs])
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    if not only:
        n += still(enc, title_card(cfg), cfg.get("title_seconds", 4.0))
        n += still(enc, overview_card(cfg), cfg.get("overview_seconds", 6.0))
    for sec, exs in plan:
        if not only:
            n += still(enc, section_card(cfg, sec), cfg.get("section_seconds", 3.0))
        for ei, ex in enumerate(exs):
            if only and only != f"{sec['key']}:{ei}":
                continue
            k = ex.render(enc)
            print(f"{sec['key']} {ei}: {ex.sc['title']}: {k / FPS:.1f} s (recording {ex.a:.1f}-{ex.b:.1f}, "
                  f"source {ex.clip[0]:.1f}-{ex.clip[1]:.1f})", flush=True)
            n += k
    if not only:
        n += still(enc, outro_card(cfg, stats), cfg.get("outro_seconds", 6.0))
    enc.stdin.close()
    enc.wait()
    print(f"wrote {out}: {n / FPS:.1f} s")
    for name, value in stats:
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
