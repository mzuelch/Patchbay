"""Minimal unit tests for anchor parsing and strict mapping.

These tests do not require torch.
"""

import pytest

from patchbay_backend.anchors import parse_anchors_cli, validate_and_clip_anchors, strict_chunk_anchors


def test_parse_anchors_cli():
    anchors = parse_anchors_cli([['+', '0', '1.5'], ['-', '2', '3']])
    assert anchors == [('+', 0.0, 1.5), ('-', 2.0, 3.0)]


def test_validate_and_clip():
    anchors = validate_and_clip_anchors([('+', -1.0, 2.0), ('-', 9.0, 12.0)], total_dur_s=10.0)
    assert anchors == [('+', 0.0, 2.0), ('-', 9.0, 10.0)]


def test_strict_chunk_anchors_intersection():
    global_anchors = [('+', 1.0, 5.0)]
    # chunk 3..7 -> intersection is 3..5 -> relative 0..2
    chunk = strict_chunk_anchors(global_anchors, chunk_start_s=3.0, chunk_end_s=7.0)
    assert chunk == [('+', 0.0, 2.0)]


def test_parse_rejects_invalid():
    with pytest.raises(ValueError):
        parse_anchors_cli([['*', '0', '1']])
