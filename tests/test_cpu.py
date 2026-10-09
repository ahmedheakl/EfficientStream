"""CPU tests: no model weights needed.  python -m pytest tests/"""
import numpy as np
from PIL import Image

from efficientstream.frames import BoundedStore, StoredFrame, fit_long_side, format_time, grid_size, letterbox
from efficientstream.index import top_k_in_time_order
from efficientstream.prompt import FINAL_VIEW, MC_INSTRUCTION, OE_HEAD, question_block


def reference_store(times, budget):
    """The store as one pass over a list of times (the evaluated implementation)."""
    keep = []
    for i in range(len(times)):
        keep.append(i)
        if len(keep) > budget:
            hi = len(keep) - 1
            if hi <= 1:
                keep.pop(1)
                continue
            gaps = [times[keep[j + 1]] - times[keep[j - 1]] for j in range(1, hi)]
            keep.pop(1 + int(np.argmin(gaps)))
    return keep


def test_store_matches_reference_and_stays_bounded():
    rng = np.random.default_rng(0)
    for budget in (1, 2, 5, 83):
        times = np.cumsum(rng.uniform(0.5, 3.0, 400)).tolist()
        s = BoundedStore(budget)
        for t in times:
            s.add(StoredFrame(t, None, np.zeros(2, np.float32)))
            assert len(s) <= budget
        assert [f.time for f in s.frames] == [times[i] for i in reference_store(times, budget)]
        if budget >= 2:
            assert s.frames[0].time == times[0] and s.frames[-1].time == times[-1]


def test_store_rejects_out_of_order():
    s = BoundedStore(4)
    s.add(StoredFrame(2.0, None, None))
    try:
        s.add(StoredFrame(1.0, None, None))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_grid_size():
    assert grid_size(1920, 1080, 256 * 32 * 32) == (672, 384)
    w, h = grid_size(640, 360, 256 * 32 * 32)
    assert (w, h) == (640, 352) and w % 32 == 0 and h % 32 == 0
    for wh in ((4000, 3000), (3000, 4000), (100, 50)):
        w, h = grid_size(*wh, 262144)
        assert w * h <= 262144 and w % 32 == 0 and h % 32 == 0


def test_letterbox_and_long_side():
    im = Image.new("RGB", (1920, 1080), (255, 0, 0))
    sq = letterbox(im, 448)
    assert sq.size == (448, 448)
    assert sq.getpixel((0, 0)) == (0, 0, 0) and sq.getpixel((224, 224)) == (255, 0, 0)
    assert fit_long_side(im, 896).size == (896, 504)
    assert fit_long_side(Image.new("RGB", (480, 640)), 896).size == (672, 896)


def test_time_format():
    assert format_time(417.4) == "00:06:57" and format_time(3725) == "01:02:05" and format_time(417.4, "s") == "417s"


def test_top_k_is_in_time_order_and_stable():
    e = np.eye(5, dtype=np.float32)
    q = np.array([0.1, 0.9, 0.0, 0.8, 0.0], np.float32)
    assert top_k_in_time_order(e, q, 2) == [1, 3]
    assert top_k_in_time_order(e, q, 10) == [0, 1, 2, 3, 4]
    assert top_k_in_time_order(np.ones((4, 2), np.float32), np.ones(2, np.float32), 2) == [0, 1]   # ties: oldest first


def test_question_block():
    mc = question_block("Where is the cup?", ["A. table", "B. floor"])
    assert mc == "\nWhere is the cup?\nA. table\nB. floor\n\n" + FINAL_VIEW + "\n" + MC_INSTRUCTION
    oe = question_block("What is he doing?", None)
    assert oe.startswith("\n" + OE_HEAD + "\nWhat is he doing?\n\n" + FINAL_VIEW + "\n")
