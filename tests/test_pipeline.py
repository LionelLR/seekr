"""Tests for app.indexer.pipeline — end-to-end indexing orchestration.

All external I/O (yt-dlp downloads, Whisper transcription) is mocked so
that the tests run without ffmpeg, network access, or API keys.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.vector_store import VectorStore
from app.indexer.pipeline import IndexingPipeline
from app.indexer.transcriber import Transcriber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_data_dir, mock_embedder, transcriber=None, chunk_duration=10.0):
    """Convenience factory for IndexingPipeline with default mocks."""
    vs = VectorStore(tmp_data_dir)
    tr = transcriber or MagicMock(spec=Transcriber)
    return IndexingPipeline(
        mock_embedder, vs, tr, chunk_duration=chunk_duration, data_dir=tmp_data_dir
    )


SAMPLE_SEGMENTS = [
    {"text": "Introduction to the topic.", "start": 0.0, "end": 5.0},
    {"text": "First key concept explained.", "start": 5.0, "end": 10.0},
    {"text": "Second key concept explored.", "start": 10.0, "end": 15.0},
]


# ---------------------------------------------------------------------------
# URL ID hashing
# ---------------------------------------------------------------------------


class TestUrlId:
    def test_same_url_produces_same_id(self, pipeline):
        url = "https://youtube.com/watch?v=abc"
        assert pipeline._url_id(url) == pipeline._url_id(url)

    def test_different_urls_produce_different_ids(self, pipeline):
        assert pipeline._url_id("https://a.com") != pipeline._url_id("https://b.com")

    def test_id_is_16_hex_chars(self, pipeline):
        uid = pipeline._url_id("https://example.com")
        assert len(uid) == 16
        assert all(c in "0123456789abcdef" for c in uid)


# ---------------------------------------------------------------------------
# Already-indexed guard
# ---------------------------------------------------------------------------


class TestAlreadyIndexed:
    def test_is_indexed_false_for_new_url(self, pipeline):
        assert not pipeline.is_indexed("https://new.example.com/v")

    def test_is_indexed_true_after_adding_hash(self, pipeline):
        url = "https://example.com/v"
        pipeline._processed.add(pipeline._url_id(url))
        assert pipeline.is_indexed(url)

    def test_process_url_returns_already_indexed_for_known_url(
        self, tmp_data_dir, mock_embedder
    ):
        pl = _make_pipeline(tmp_data_dir, mock_embedder)
        url = "https://youtube.com/watch?v=known"
        pl._processed.add(pl._url_id(url))
        result = pl.process_url(url)
        assert result["status"] == "already_indexed"
        assert result["chunks"] == 0


# ---------------------------------------------------------------------------
# Successful YouTube / direct URL flow
# ---------------------------------------------------------------------------


class TestProcessUrlSuccess:
    @pytest.fixture
    def patched_direct(self):
        """Patch all I/O so process_url runs without network or ffmpeg."""
        with (
            patch("app.indexer.pipeline.get_video_info") as mock_info,
            patch("app.indexer.pipeline.download_audio") as mock_dl,
            patch("app.indexer.pipeline.get_platform", return_value="direct"),
            patch("app.indexer.pipeline.is_local_file", return_value=False),
        ):
            mock_info.return_value = {
                "title": "Test Video",
                "thumbnail": "https://img.example.com/thumb.jpg",
                "url": "https://example.com/video.mp4",
            }
            mock_dl.return_value = "/tmp/fake_audio.mp3"
            yield

    def test_returns_success_status(
        self, tmp_data_dir, mock_embedder, mock_transcriber, patched_direct
    ):
        mock_transcriber.transcribe.return_value = SAMPLE_SEGMENTS
        pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_transcriber, chunk_duration=10.0)
        result = pl.process_url("https://example.com/video.mp4")
        assert result["status"] == "success"

    def test_chunks_count_matches_segmenter_output(
        self, tmp_data_dir, mock_embedder, mock_transcriber, patched_direct
    ):
        mock_transcriber.transcribe.return_value = SAMPLE_SEGMENTS
        # 3 segments × 5s = 15s → two chunks at 10s
        pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_transcriber, chunk_duration=10.0)
        result = pl.process_url("https://example.com/video.mp4")
        assert result["chunks"] == 2

    def test_url_is_marked_as_processed(
        self, tmp_data_dir, mock_embedder, mock_transcriber, patched_direct
    ):
        mock_transcriber.transcribe.return_value = SAMPLE_SEGMENTS
        pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_transcriber)
        url = "https://example.com/video.mp4"
        pl.process_url(url)
        assert pl.is_indexed(url)

    def test_metadata_stored_in_vector_store(
        self, tmp_data_dir, mock_embedder, mock_transcriber, patched_direct
    ):
        mock_transcriber.transcribe.return_value = SAMPLE_SEGMENTS
        pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_transcriber, chunk_duration=10.0)
        pl.process_url("https://example.com/video.mp4")
        assert pl.vector_store.total_segments >= 1

    def test_on_progress_callback_called(
        self, tmp_data_dir, mock_embedder, mock_transcriber, patched_direct
    ):
        mock_transcriber.transcribe.return_value = SAMPLE_SEGMENTS
        pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_transcriber)
        messages = []
        pl.process_url("https://example.com/video.mp4", on_progress=messages.append)
        assert len(messages) > 0

    def test_idempotent_second_call_skips(
        self, tmp_data_dir, mock_embedder, mock_transcriber, patched_direct
    ):
        mock_transcriber.transcribe.return_value = SAMPLE_SEGMENTS
        pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_transcriber)
        url = "https://example.com/video.mp4"
        pl.process_url(url)
        result2 = pl.process_url(url)
        assert result2["status"] == "already_indexed"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestProcessUrlErrors:
    def test_download_failure_returns_error(self, tmp_data_dir, mock_embedder):
        with (
            patch("app.indexer.pipeline.get_video_info") as mock_info,
            patch("app.indexer.pipeline.download_audio", return_value=None),
            patch("app.indexer.pipeline.get_platform", return_value="direct"),
            patch("app.indexer.pipeline.is_local_file", return_value=False),
        ):
            mock_info.return_value = {"title": "Broken", "thumbnail": "", "url": "http://bad.com"}
            pl = _make_pipeline(tmp_data_dir, mock_embedder)
            result = pl.process_url("http://bad.com/v.mp4")
        assert result["status"] == "error"
        assert "chunks" in result

    def test_empty_transcription_returns_error(self, tmp_data_dir, mock_embedder):
        mock_tr = MagicMock(spec=Transcriber)
        mock_tr.transcribe.return_value = []  # nothing transcribed

        with (
            patch("app.indexer.pipeline.get_video_info") as mock_info,
            patch("app.indexer.pipeline.download_audio", return_value="/tmp/audio.mp3"),
            patch("app.indexer.pipeline.get_platform", return_value="direct"),
            patch("app.indexer.pipeline.is_local_file", return_value=False),
        ):
            mock_info.return_value = {"title": "Silent", "thumbnail": "", "url": "http://s.com"}
            pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_tr)
            result = pl.process_url("http://s.com/silent.mp4")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Social platform path (NEEDS_LOCAL_SERVING)
# ---------------------------------------------------------------------------


class TestSocialPlatformPath:
    def test_linkedin_uses_download_video(self, tmp_data_dir, mock_embedder):
        mock_tr = MagicMock(spec=Transcriber)
        mock_tr.transcribe.return_value = SAMPLE_SEGMENTS

        with (
            patch("app.indexer.pipeline.get_video_info") as mock_info,
            patch("app.indexer.pipeline.download_video") as mock_dl_vid,
            patch("app.indexer.pipeline.get_platform", return_value="linkedin"),
            patch("app.indexer.pipeline.is_local_file", return_value=False),
        ):
            mock_info.return_value = {
                "title": "LinkedIn Post",
                "thumbnail": "",
                "url": "https://linkedin.com/posts/...",
            }
            mock_dl_vid.return_value = str(Path(tmp_data_dir) / "vid.mp4")
            # create the fake file so the pipeline finds it
            Path(tmp_data_dir).mkdir(parents=True, exist_ok=True)
            (Path(tmp_data_dir) / "vid.mp4").write_bytes(b"\x00")

            pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_tr)
            result = pl.process_url("https://linkedin.com/posts/...")
        assert result["status"] == "success"
        mock_dl_vid.assert_called_once()

    def test_social_served_url_uses_api_endpoint(self, tmp_data_dir, mock_embedder):
        mock_tr = MagicMock(spec=Transcriber)
        mock_tr.transcribe.return_value = SAMPLE_SEGMENTS

        with (
            patch("app.indexer.pipeline.get_video_info") as mock_info,
            patch("app.indexer.pipeline.download_video") as mock_dl_vid,
            patch("app.indexer.pipeline.get_platform", return_value="tiktok"),
            patch("app.indexer.pipeline.is_local_file", return_value=False),
        ):
            mock_info.return_value = {"title": "TikTok", "thumbnail": "", "url": "https://tiktok.com/..."}
            uid = IndexingPipeline(
                mock_embedder, VectorStore(tmp_data_dir), mock_tr, data_dir=tmp_data_dir
            )._url_id("https://tiktok.com/...")
            fake_path = str(Path(tmp_data_dir) / f"{uid}.mp4")
            Path(tmp_data_dir).mkdir(parents=True, exist_ok=True)
            Path(fake_path).write_bytes(b"\x00")
            mock_dl_vid.return_value = fake_path

            pl = _make_pipeline(tmp_data_dir, mock_embedder, mock_tr)
            pl.process_url("https://tiktok.com/...")
            # The stored video_url must point to the local serving endpoint
            stored = pl.vector_store.metadata[0]["video_url"]
        assert stored.startswith("/api/local-video/")


# ---------------------------------------------------------------------------
# process_directory
# ---------------------------------------------------------------------------


class TestProcessDirectory:
    def test_finds_video_files(
        self, tmp_data_dir, mock_embedder, sample_video_dir
    ):
        pl = _make_pipeline(tmp_data_dir, mock_embedder)
        with patch.object(pl, "process_url", return_value={"status": "success", "chunks": 1}) as mock_pu:
            results = pl.process_directory(str(sample_video_dir))
        assert len(results) == 2
        assert mock_pu.call_count == 2

    def test_returns_list_of_result_dicts(
        self, tmp_data_dir, mock_embedder, sample_video_dir
    ):
        pl = _make_pipeline(tmp_data_dir, mock_embedder)
        with patch.object(pl, "process_url", return_value={"status": "success", "chunks": 3}):
            results = pl.process_directory(str(sample_video_dir))
        assert all("status" in r for r in results)

    def test_ignores_non_video_files(self, tmp_data_dir, mock_embedder, tmp_path):
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        (mixed_dir / "notes.txt").write_text("not a video")
        (mixed_dir / "clip.mp4").write_bytes(b"\x00")

        pl = _make_pipeline(tmp_data_dir, mock_embedder)
        with patch.object(pl, "process_url", return_value={"status": "success", "chunks": 1}) as mock_pu:
            pl.process_directory(str(mixed_dir))
        assert mock_pu.call_count == 1

    def test_indexed_count_property(self, pipeline):
        assert pipeline.indexed_count == 0
        pipeline._processed.add("abc123")
        assert pipeline.indexed_count == 1
