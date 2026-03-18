"""Tests for app.indexer.segmenter — transcript segment merging."""

import pytest

from app.indexer.segmenter import merge_segments


def _seg(text: str, start: float, end: float) -> dict:
    """Shorthand for building a Whisper-style segment dict."""
    return {"text": text, "start": start, "end": end}


class TestEdgeCases:
    """Edge cases that must not raise exceptions."""

    def test_empty_input_returns_empty_list(self):
        assert merge_segments([]) == []

    def test_single_short_segment_emitted_as_one_chunk(self):
        segs = [_seg("Hello.", 0.0, 3.0)]
        result = merge_segments(segs, chunk_duration=15.0)
        assert len(result) == 1
        assert result[0]["text"] == "Hello."

    def test_single_long_segment_emitted_as_one_chunk(self):
        """A single segment longer than chunk_duration is always one chunk."""
        segs = [_seg("Very long monologue.", 0.0, 120.0)]
        result = merge_segments(segs, chunk_duration=15.0)
        assert len(result) == 1


class TestChunking:
    """Tests for the core accumulation logic."""

    def test_texts_are_joined_with_space(self):
        segs = [_seg("Hello", 0.0, 5.0), _seg("world.", 5.0, 9.0)]
        result = merge_segments(segs, chunk_duration=20.0)
        assert result[0]["text"] == "Hello world."

    def test_exact_boundary_triggers_new_chunk(self):
        """When elapsed time == chunk_duration the chunk must close."""
        segs = [
            _seg("First sentence.", 0.0, 10.0),
            _seg("Second sentence.", 10.0, 20.0),
        ]
        result = merge_segments(segs, chunk_duration=10.0)
        assert len(result) == 2

    def test_chunks_cover_correct_time_ranges(self):
        segs = [
            _seg("intro", 0.0, 8.0),
            _seg("topic", 8.0, 16.0),
            _seg("outro", 16.0, 20.0),
        ]
        result = merge_segments(segs, chunk_duration=10.0)
        assert result[0]["start"] == 0.0
        assert result[-1]["end"] == 20.0

    def test_last_partial_chunk_is_always_emitted(self):
        """Even when the final segment does not reach chunk_duration."""
        segs = [
            _seg("A long sentence.", 0.0, 12.0),
            _seg("B", 12.0, 14.0),  # 2-second leftover
        ]
        result = merge_segments(segs, chunk_duration=10.0)
        assert result[-1]["text"] == "B"

    def test_six_segments_produce_two_chunks_at_15s(self):
        # Sentence boundary after segment 2 (elapsed=15s) triggers first chunk.
        segs = [_seg(f"w{i}{'.' if i == 2 else ''}", float(i * 5), float((i + 1) * 5)) for i in range(6)]
        result = merge_segments(segs, chunk_duration=15.0)
        assert len(result) == 2

    def test_chunk_duration_respected_for_many_segments(self):
        # 30 segments of 2s; sentence boundary every 8 segments (at 16s) → 4 chunks.
        segs = [_seg(f"s{i}{'.' if (i + 1) % 8 == 0 else ''}", i * 2.0, (i + 1) * 2.0) for i in range(30)]
        result = merge_segments(segs, chunk_duration=15.0)
        assert len(result) == 4


class TestOutputStructure:
    """Every chunk must have exactly the expected keys."""

    def test_output_dicts_have_required_keys(self):
        segs = [_seg("text", 0.0, 5.0)]
        result = merge_segments(segs)
        assert set(result[0].keys()) == {"text", "start", "end"}

    def test_start_and_end_are_floats(self):
        segs = [_seg("text", 0.0, 5.0)]
        result = merge_segments(segs)
        assert isinstance(result[0]["start"], float)
        assert isinstance(result[0]["end"], float)

    def test_end_is_greater_than_or_equal_to_start(self):
        segs = [_seg(f"s{i}", i * 5.0, (i + 1) * 5.0) for i in range(4)]
        for chunk in merge_segments(segs, chunk_duration=15.0):
            assert chunk["end"] >= chunk["start"]
