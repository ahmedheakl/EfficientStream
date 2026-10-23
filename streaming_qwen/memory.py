"""The caption memory: what the caption stream wrote while the video played, as timed events, and the text block a
question reads them in. Details come from </Observe> answers; a </Summary> answer covers its segment, from the segment's
first detail to the second it was written."""
import re

from .prompts import MEMORY_POSTAMBLE, MEMORY_PREAMBLE


class CaptionMemory:
    def __init__(self):
        self.events, self._first_observe, self._fallback = [], None, 0.0

    def add(self, answer, a, b):
        """One second's answer for [a, b): adds an event for </Observe> or </Summary> with text, nothing otherwise."""
        tag, _, text = answer.strip().partition(">")
        tag, text = tag + ">", text.strip()
        if tag == "</Observe>" and text:
            if self._first_observe is None:
                self._first_observe = a
            self.events.append(dict(kind="observe", start=a, end=b, text=text))
        elif tag == "</Summary>" and text:
            start = self._first_observe if self._first_observe is not None else self._fallback
            self.events.append(dict(kind="summary", start=min(start, b), end=b, text=text))
            self._fallback, self._first_observe = b, None

    def before(self, t):
        """The events that ended by second t."""
        return [e for e in self.events if e["end"] <= t]


def caption_body(text):
    """A caption without its leading time stamps, e.g. "[12s] [0s-20s] text" -> "text"."""
    return re.sub(r"^(\[[^\]]*\]\s*)+", "", text).strip()


def compact(events, time, keep_recent=60.0):
    """The memory a question gets: a summary replaces the details it covers, except details that end in the last
    `keep_recent` seconds before `time`, and identical details are merged into one event spanning their times."""
    sums = [e for e in events if e["kind"] == "summary" and e["text"]]
    inside = lambda e: any(s["start"] <= e["start"] and e["end"] <= s["end"] for s in sums)
    kept = [e for e in events if not (e["kind"] == "observe" and inside(e) and e["end"] < time - keep_recent)]
    out, seen = [], {}
    for e in kept:
        if e["kind"] != "observe":
            out.append(e)
            continue
        body = caption_body(e["text"])
        if body in seen:
            m = seen[body]
            m["start"], m["end"], m["text"] = min(m["start"], e["start"]), max(m["end"], e["end"]), body
        else:
            seen[body] = dict(e)
            out.append(seen[body])
    return sorted(out, key=lambda e: (e["end"], e["kind"]))


def memory_text(events):
    """The memory block of a question's prompt: summaries, then details, each "[a s-b s] text"."""
    t = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0").rstrip(".")
    line = lambda e: f"[{t(e['start'])}s-{t(e['end'])}s] {e['text']}"
    return "\n".join([MEMORY_PREAMBLE, "<SEMANTIC_SUMMARIES>", *[line(e) for e in events if e["kind"] == "summary"],
                      "</SEMANTIC_SUMMARIES>", "<EVENT_CAPTIONS>", *[line(e) for e in events if e["kind"] == "observe"],
                      "</EVENT_CAPTIONS>", MEMORY_POSTAMBLE])
