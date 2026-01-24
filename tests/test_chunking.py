"""Minimal unit tests for chunking logic.

These tests are intentionally lightweight and do not require torch or the
SAM-Audio model.

Run with:
    pytest -q

(Only needed if you want to maintain this as a package.)
"""

from patchbay_backend.chunking import decide_chunking, make_chunk_plan


def test_no_chunking_below_threshold_when_no_params():
    enable, max_len_s, ov_s = decide_chunking(duration_s=10.0, max_len_s=None, overlap_s=None)
    assert enable is False
    assert ov_s == 0.0


def test_no_auto_chunking_above_threshold_when_no_params():
    """Requested behavior: no auto-chunking for long audio unless max_len_s is set."""
    enable, max_len_s, ov_s = decide_chunking(duration_s=31.0, max_len_s=None, overlap_s=None)
    assert enable is False
    assert ov_s == 0.0


def test_overlap_without_max_len_is_ignored():
    """Requested behavior: overlap_s without max_len_s must not enable chunking."""
    enable, max_len_s, ov_s = decide_chunking(duration_s=10.0, max_len_s=None, overlap_s=1.0)
    assert enable is False
    assert ov_s == 0.0


def test_make_plan_valid():
    plan = make_chunk_plan(total_samples=48000 * 40, sr=48000, duration_s=40.0, max_len_s=10.0, overlap_s=1.0)
    assert plan.enable is True
    assert plan.chunk_len == 48000 * 10
    assert plan.overlap == 48000
    assert plan.step == 48000 * 9
