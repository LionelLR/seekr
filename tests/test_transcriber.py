"""Tests for app.indexer.transcriber — Whisper transcription wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from app.indexer.transcriber import Transcriber


def _make_whisper_result(segments):
    """Build a fake Whisper transcription result dict."""
    return {"segments": segments, "text": " ".join(s["text"] for s in segments)}


class TestLazyLoading:
    """The Whisper model must not be loaded until transcribe() is called."""

    def test_model_not_loaded_on_init(self):
        tr = Transcriber("base")
        assert tr._model is None

    def test_model_loaded_on_first_access(self):
        with patch("app.indexer.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = MagicMock()
            tr = Transcriber("small")
            _ = tr.model  # trigger load
            mock_whisper.load_model.assert_called_once_with("small")

    def test_model_only_loaded_once(self):
        with patch("app.indexer.transcriber.whisper") as mock_whisper:
            mock_whisper.load_model.return_value = MagicMock()
            tr = Transcriber("tiny")
            _ = tr.model
            _ = tr.model
            assert mock_whisper.load_model.call_count == 1


class TestTranscribe:
    """Tests for the transcribe() method output format and filtering."""

    @pytest.fixture
    def patched_transcriber(self):
        """Return a Transcriber whose whisper.load_model is mocked."""
        with patch("app.indexer.transcriber.whisper") as mock_whisper:
            mock_model = MagicMock()
            mock_whisper.load_model.return_value = mock_model
            tr = Transcriber("base")
            yield tr, mock_model

    def test_returns_list_of_dicts(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = _make_whisper_result(
            [{"text": " Hello world", "start": 0.0, "end": 2.5}]
        )
        result = tr.transcribe(sample_wav_path)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_segment_has_text_start_end_keys(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = _make_whisper_result(
            [{"text": " Hi", "start": 0.0, "end": 1.0}]
        )
        result = tr.transcribe(sample_wav_path)
        assert set(result[0].keys()) == {"text", "start", "end"}

    def test_leading_whitespace_stripped(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = _make_whisper_result(
            [{"text": "  spaces around  ", "start": 0.0, "end": 2.0}]
        )
        result = tr.transcribe(sample_wav_path)
        assert result[0]["text"] == "spaces around"

    def test_empty_segments_filtered_out(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = _make_whisper_result(
            [
                {"text": "   ", "start": 0.0, "end": 1.0},  # whitespace-only
                {"text": "real text", "start": 1.0, "end": 3.0},
            ]
        )
        result = tr.transcribe(sample_wav_path)
        assert len(result) == 1
        assert result[0]["text"] == "real text"

    def test_empty_transcript_returns_empty_list(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = {"segments": []}
        result = tr.transcribe(sample_wav_path)
        assert result == []

    def test_timestamps_preserved(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = _make_whisper_result(
            [{"text": "check timestamp", "start": 12.5, "end": 17.3}]
        )
        result = tr.transcribe(sample_wav_path)
        assert result[0]["start"] == 12.5
        assert result[0]["end"] == 17.3

    def test_multiple_segments_returned(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        raw = [
            {"text": f" segment {i}", "start": float(i * 5), "end": float((i + 1) * 5)}
            for i in range(5)
        ]
        mock_model.transcribe.return_value = _make_whisper_result(raw)
        result = tr.transcribe(sample_wav_path)
        assert len(result) == 5

    def test_verbose_false_passed_to_whisper(self, patched_transcriber, sample_wav_path):
        tr, mock_model = patched_transcriber
        mock_model.transcribe.return_value = {"segments": []}
        tr.transcribe(sample_wav_path)
        _, kwargs = mock_model.transcribe.call_args
        assert kwargs.get("verbose") is False
