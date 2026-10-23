"""ST (results/HYBRID_RULES.md): FOLIO-style focused semantic memory (arXiv 2607.13298), training-free, built for our streaming setup.

Writer loop, per video, every 8-s segment (causal: a segment is written only from its own frames and the memory before it):
  keyframes  1-fps probe frames of the segment; keep first + last, and add the middle + the highest-change frame when the change is high
  writer     a VLM turns the keyframes + the current focus list into JSON entity records (state, location, relations, visible text) and
             event records (action, participants, changed entities, time)
  merge      rule-based identity matching on normalised names / aliases (+ category) appends records to entity chains
  focus      per-entity score p <- clip(g * p + a . phi+ - b . phi-) with fixed coefficients; levels focus / support / context set how much
             detail the writer is asked for in the next segment
Query time (render): content words of the question + options are matched against entity names / aliases and records; the top entities
with their state chains and the matching events, in time order, restricted to [question start, question end], become the memory text."""
import json
import re

import numpy as np

SEG = 8.0
G, A_VIS, A_RE, A_ST, A_LOC, A_EV, B_GONE, B_STATIC = 0.7, 0.15, 0.2, 0.25, 0.15, 0.25, 0.1, 0.1
N_FOCUS, P_FOCUS, P_SUPPORT, MAX_LISTED = 6, 0.4, 0.2, 12
CATS = ("person", "object", "animal", "vehicle", "place", "text", "other")
STOP = set("""a an the is are was were be been being am do does did done to of in on at by for with from into onto as and or but not no yes
what which who whom whose when where why how this that these those there here it its they them their he him his she her you your i me my we
our us can could would should will shall may might must has have had having than then so if about over under after before during while
up down out off again once just only also very too more most some any each every all both either neither other another such same own
video frame frames scene shown show shows seen see currently now moment time clip camera person's""".split())


def clean_name(s):
    """Drop a copied writing level ("focus: ...") or category tag ("... [person]") from a name."""
    s = re.sub(r"^\s*((focus|support|context)\s*:\s*)+", "", str(s or ""), flags=re.I)
    return re.sub(r"(\s*\[[^\]]*\])+\s*$", "", s).strip()


def norm(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", clean_name(s).lower())
    s = re.sub(r"^(a|an|the) ", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def words(s):
    out = set()
    for w in norm(s).split():
        if w in STOP or len(w) < 2:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


# ---------------------------------------------------------------- writer side
WRITER_SYSTEM = ("You write memory records for a streaming video assistant. Report only what is visible in the given frames; never guess "
                 "beyond them. Answer with JSON only.")


def writer_prompt(t0, t1, times, listed, video=False):
    lines = []
    for e in listed:  # one JSON object per entity, so the writer does not copy level or category into the name
        last = e["records"][-1] if e["records"] else {}
        lines.append(json.dumps(dict(name=e["name"], category=e["category"], level=e["level"], last_state=last.get("state", ""),
                                     last_location=last.get("location", "")), ensure_ascii=False))
    known = "\n".join(lines) if lines else "none yet"
    src = (f"This video clip covers {t0:.0f} s to {t1:.0f} s of the video, two frames per second." if video else
           f"These frames are from {t0:.0f} s to {t1:.0f} s of the video (frames at {', '.join(f'{t:.0f} s' for t in times)}).")
    return (f"{src}\n"
            f"Entities already in memory (name, category, writing level, last state and location):\n{known}\n\n"
            "Write one JSON object with two lists.\n"
            '"entities": one item per entity visible in these frames (people, objects, animals, vehicles, places, on-screen text): '
            '{"name": short canonical name, "aliases": [other names], "category": one of person/object/animal/vehicle/place/text/other, '
            '"state": what it is doing or what condition it is in, "location": where it is, "relations": [short relations to other '
            'entities], "text": visible text on it or ""}. "name" is a plain name only (no level, no category). Reuse the exact names above for the same entities. Give full detail for focus '
            "entities; for support entities give state and location; for any other entity give only name, category and a short state.\n"
            '"events": actions or changes that happen in these frames, in order, one item per occurrence (if an action repeats, list each '
            'repetition; use specific verbs for what is done to what, not generic ones such as moving or talking): {"action": short verb phrase, "participants": [entity names], "changed": [entities whose state or location '
            'changed], "time": seconds}.\n'
            "Return only the JSON object, compact on one line (no indentation), omitting empty fields.")


def keyframes(probe, low_thr=6.0):
    """probe: [(t, PIL)] 1-fps frames of a segment. Boundaries always; the middle + the highest-change frame when the largest frame-to-frame
    change (mean abs grey difference at 64 px, 0-255) reaches low_thr. Returns (indices, max change)."""
    if len(probe) <= 2:
        return list(range(len(probe))), 0.0
    g = [np.asarray(im.convert("L").resize((64, 36)), np.float32) for _, im in probe]
    ch = [float(np.abs(a - b).mean()) for a, b in zip(g[:-1], g[1:])]
    idx = {0, len(probe) - 1}
    if max(ch) >= low_thr:
        idx |= {len(probe) // 2, int(np.argmax(ch)) + 1}
    return sorted(idx), max(ch)


def _repair(t):
    """A truncated JSON object: cut after the last complete list item and close the open brackets."""
    stack, in_str, esc, best = [], False, False, None
    for i, c in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack:
                break
            stack.pop()
            if c == "}" and stack and stack[-1] == "]":
                best = (i, list(stack))
    if best is None:
        return None
    i, st = best
    return t[:i + 1] + "".join(reversed(st))


def parse_json(txt):
    s = txt.find("{")
    if s < 0:
        return None
    t = txt[s:txt.rfind("}") + 1] if txt.rfind("}") > s else txt[s:]
    for cand in (t, re.sub(r",\s*([}\]])", r"\1", t), _repair(txt[s:])):
        if not cand:
            continue
        try:
            d = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            return d
    return None


class Memory:
    def __init__(self):
        self.entities, self.events, self.index, self.seg = [], [], {}, 0

    def match(self, name, aliases, cat):
        for k in [norm(name)] + [norm(a) for a in aliases or []]:
            i = self.index.get(k)
            if i is not None and (cat in ("other", "") or self.entities[i]["category"] in ("other", cat)):
                return i
        return None

    def listed(self):
        es = [e for e in self.entities if e["level"] in ("focus", "support")]
        return sorted(es, key=lambda e: -e["p"])[:MAX_LISTED]

    def update(self, t0, t1, d):
        """Merge one segment's writer output; then the focus update."""
        self.seg += 1
        seen, changed_names = set(), set()
        for ev in (d.get("events") or []) if isinstance(d.get("events"), list) else []:
            if not isinstance(ev, dict) or not ev.get("action"):
                continue
            try:
                t = float(ev.get("time", t0))
            except (TypeError, ValueError):
                t = t0
            parts = [clean_name(x) for x in ev.get("participants") or [] if x][:6]
            chg = [clean_name(x) for x in ev.get("changed") or [] if x][:6]
            self.events.append(dict(t=min(max(t, t0), t1), t0=t0, t1=t1, action=str(ev["action"])[:120], participants=parts, changed=chg))
            changed_names |= {norm(x) for x in parts + chg}
        for en in (d.get("entities") or []) if isinstance(d.get("entities"), list) else []:
            if not isinstance(en, dict) or not en.get("name"):
                continue
            name = clean_name(en["name"])[:60]
            aliases = [clean_name(a)[:60] for a in en.get("aliases") or [] if a and norm(a) not in ("", "focus", "support", "context")][:4]
            if not name:
                continue
            cat = str(en.get("category") or "other").lower()
            cat = cat if cat in CATS else "other"
            i = self.match(name, aliases, cat)
            if i is None:
                i = len(self.entities)
                self.entities.append(dict(name=name, aliases=[], category=cat, records=[], p=0.3, level="support", last_seen=None, first_seen=t1))
            e = self.entities[i]
            for k in [name] + aliases:
                if norm(k) and norm(k) not in self.index:
                    self.index[norm(k)] = i
                if norm(k) != norm(e["name"]) and norm(k) not in {norm(a) for a in e["aliases"]}:
                    e["aliases"].append(k)
            rel = en.get("relations") or []
            rec = dict(t0=t0, t1=t1, state=str(en.get("state") or "")[:160], location=str(en.get("location") or "")[:100],
                       relations=[str(x)[:80] for x in rel][:4] if isinstance(rel, list) else [str(rel)[:80]],
                       text=str(en.get("text") or "")[:120])
            prev = e["records"][-1] if e["records"] else None
            e["_re"] = e["last_seen"] is not None and e["last_seen"] < t0 - 1e-6
            e["_st"] = prev is not None and norm(prev["state"]) != norm(rec["state"])
            e["_loc"] = prev is not None and rec["location"] and norm(prev["location"]) != norm(rec["location"])
            e["records"].append(rec)
            e["last_seen"] = t1
            seen.add(i)
        for i, e in enumerate(self.entities):
            vis = i in seen
            ev = bool(({norm(e["name"])} | {norm(a) for a in e["aliases"]}) & changed_names)
            pos = (A_VIS * vis + A_RE * (vis and e.get("_re", False)) + A_ST * (vis and e.get("_st", False)) +
                   A_LOC * (vis and bool(e.get("_loc", False))) + A_EV * bool(ev))
            static = vis and not (e.get("_st") or e.get("_loc") or ev)
            e["p"] = float(np.clip(G * e["p"] + pos - B_GONE * (not vis) - B_STATIC * static, 0, 1))
            for k in ("_re", "_st", "_loc"):
                e.pop(k, None)
        order = sorted(range(len(self.entities)), key=lambda i: -self.entities[i]["p"])
        for r, i in enumerate(order):
            p = self.entities[i]["p"]
            self.entities[i]["level"] = "focus" if r < N_FOCUS and p >= P_FOCUS else "support" if p >= P_SUPPORT else "context"

    def dump(self):
        return dict(entities=[{k: v for k, v in e.items()} for e in self.entities], events=self.events)


# ---------------------------------------------------------------- query side
def _spans(records):
    """Collapse consecutive records with the same state and location into one time span."""
    out = []
    for r in records:
        key = (norm(r["state"]), norm(r["location"]))
        if out and out[-1][0] == key and r["t0"] <= out[-1][1]["t1"] + 1e-6:
            out[-1][1]["t1"] = r["t1"]
            out[-1][1]["text"] = out[-1][1]["text"] or r["text"]
        else:
            out.append((key, dict(r)))
    return [x for _, x in out]


def render(mem, q_start, q_end, question, options=None, k_ent=8, k_ev=24):
    """Memory text for one question: entities and events written inside [q_start, q_end]."""
    opts = list(options.values()) if isinstance(options, dict) else list(options or [])
    terms = words(question) | set().union(*[words(o) for o in opts]) if opts else words(question)
    ents = []
    for e in mem["entities"]:
        recs = [r for r in e["records"] if r["t1"] <= q_end + 1e-6 and r["t0"] >= q_start - SEG]
        if not recs:
            continue
        s = 2 * len(words(e["name"]) & terms) + len(set().union(*[words(a) for a in e["aliases"]]) & terms if e["aliases"] else set())
        s += 0.5 * len(set().union(*[words(r["state"]) | words(r["text"]) for r in recs]) & terms)
        ents.append((s, e, recs))
    evs = [v for v in mem["events"] if v["t1"] <= q_end + 1e-6 and v["t0"] >= q_start - SEG]
    sel = [x for x in sorted(ents, key=lambda x: -x[0]) if x[0] > 0][:k_ent]
    names = {norm(e["name"]) for _, e, _ in sel} | {norm(a) for _, e, _ in sel for a in e["aliases"]}
    def ev_score(v):
        return 2 * len(words(v["action"]) & terms) + len({norm(x) for x in v["participants"] + v["changed"]} & names)
    chosen = [v for v in evs if ev_score(v) > 0]
    fallback = not sel and not chosen
    if fallback:  # nothing matched: the most recent events and the entities seen in the last segment
        chosen = evs[-k_ev:]
        last = max((r["t1"] for _, _, rs in ents for r in rs), default=None)
        sel = [x for x in ents if x[2][-1]["t1"] == last][:k_ent]
    chosen = sorted(sorted(chosen, key=lambda v: -ev_score(v))[:k_ev], key=lambda v: v["t"])
    lines = ["<LONG_TERM_VIDEO_MEMORY>"]
    if sel:
        lines.append("Entities (state over time):")
        for _, e, recs in sel:
            spans = []
            for r in _spans(recs)[-8:]:
                bits = "; ".join(x for x in (r["state"], f"location: {r['location']}" if r["location"] else "", f"text: {r['text']}" if r["text"] else "") if x)
                spans.append(f"{r['t0']:.0f}-{r['t1']:.0f}s {bits}")
            alias = f" (also: {', '.join(e['aliases'][:2])})" if e["aliases"] else ""
            lines.append(f"- {e['name']}{alias} [{e['category']}]: " + " | ".join(spans))
    if chosen:
        lines.append("Events (in order):")
        for v in chosen:
            who = ", ".join(v["participants"][:3])
            lines.append(f"- {v['t']:.0f}s: {v['action']}" + (f" ({who})" if who else ""))
    if len(lines) == 1:
        lines.append("None for this question.")
    lines.append("</LONG_TERM_VIDEO_MEMORY>")
    return "\n".join(lines), dict(n_ent=len(sel), n_ev=len(chosen), fallback=int(fallback))
