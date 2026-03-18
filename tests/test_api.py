"""Tests for app.api.routes — FastAPI endpoint behaviour.

Uses Starlette's synchronous TestClient.  All external services
(embedder, pipeline, vector store) are provided by the ``api_client``
fixture defined in conftest.py, which injects mock dependencies directly
into the ``app.api.routes`` module-level globals.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.vector_store import VectorStore


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------


class TestStatusEndpoint:
    def test_returns_200(self, api_client):
        response = api_client.get("/api/status")
        assert response.status_code == 200

    def test_empty_index_returns_zero_segments(self, api_client):
        data = api_client.get("/api/status").json()
        assert data["total_segments"] == 0

    def test_empty_index_returns_zero_indexed_videos(self, api_client):
        data = api_client.get("/api/status").json()
        assert data["indexed_videos"] == 0

    def test_videos_list_is_empty_initially(self, api_client):
        data = api_client.get("/api/status").json()
        assert data["videos"] == []

    def test_reflects_added_segments(self, api_client, mock_embedder):
        from app.api import routes

        routes._pipeline.vector_store.add(
            [mock_embedder.embed("hello")],
            [{"text": "hello", "video_url": "https://youtube.com/v=abc", "start": 0.0, "end": 5.0,
              "video_title": "T", "video_thumbnail": ""}],
        )
        data = api_client.get("/api/status").json()
        assert data["total_segments"] == 1


# ---------------------------------------------------------------------------
# POST /api/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_empty_index_returns_200_and_empty_results(self, api_client):
        response = api_client.post("/api/search", json={"query": "gradient descent"})
        assert response.status_code == 200
        assert response.json()["results"] == []

    def test_echoes_query_in_response(self, api_client):
        data = api_client.post("/api/search", json={"query": "my query"}).json()
        assert data["query"] == "my query"

    def test_returns_result_with_correct_fields(self, api_client, mock_embedder):
        from app.api import routes

        vec = mock_embedder.embed("machine learning fundamentals")
        routes._pipeline.vector_store.add(
            [vec],
            [{
                "text": "machine learning fundamentals",
                "video_url": "https://youtube.com/watch?v=ml",
                "video_title": "ML 101",
                "video_thumbnail": "",
                "start": 30.0,
                "end": 45.0,
            }],
        )
        results = api_client.post(
            "/api/search", json={"query": "machine learning fundamentals", "top_k": 1}
        ).json()["results"]
        assert len(results) == 1
        r = results[0]
        assert r["video_title"] == "ML 101"
        assert r["start"] == 30.0
        assert r["end"] == 45.0
        assert 0.0 <= r["score"] <= 1.0

    def test_top_k_limits_results(self, api_client, mock_embedder):
        from app.api import routes

        for i in range(8):
            routes._pipeline.vector_store.add(
                [mock_embedder.embed(f"topic {i}")],
                [{
                    "text": f"topic {i}",
                    "video_url": "https://youtube.com/watch?v=x",
                    "video_title": "V",
                    "video_thumbnail": "",
                    "start": float(i * 10),
                    "end": float(i * 10 + 10),
                }],
            )
        results = api_client.post(
            "/api/search", json={"query": "topic", "top_k": 3}
        ).json()["results"]
        assert len(results) <= 3

    def test_score_is_between_zero_and_one(self, api_client, mock_embedder):
        from app.api import routes

        vec = mock_embedder.embed("neural network")
        routes._pipeline.vector_store.add(
            [vec],
            [{"text": "neural network", "video_url": "u", "video_title": "T",
              "video_thumbnail": "", "start": 0.0, "end": 5.0}],
        )
        results = api_client.post(
            "/api/search", json={"query": "neural network"}
        ).json()["results"]
        assert all(0.0 <= r["score"] <= 1.0 + 1e-6 for r in results)

    def test_results_ordered_by_descending_score(self, api_client, mock_embedder):
        from app.api import routes

        # Add multiple segments; the one matching the query closely should rank first
        target_text = "backpropagation algorithm explained"
        for text in ("random noise text A", target_text, "random noise text B"):
            routes._pipeline.vector_store.add(
                [mock_embedder.embed(text)],
                [{"text": text, "video_url": "u", "video_title": "T",
                  "video_thumbnail": "", "start": 0.0, "end": 5.0}],
            )
        results = api_client.post(
            "/api/search", json={"query": target_text, "top_k": 3}
        ).json()["results"]
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# POST /api/admin/extract
# ---------------------------------------------------------------------------


def _video_entry(url: str, title: str = "Test Video", duration: float = 120.0) -> dict:
    """Helper: build a dict matching the shape returned by extract_page_videos."""
    return {"url": url, "title": title, "thumbnail": "https://img/thumb.jpg", "duration": duration}


class TestExtractEndpoint:
    def test_returns_200(self, api_client):
        entry = _video_entry("https://youtube.com/watch?v=abc")
        with patch("app.api.routes.extract_page_videos", return_value=[entry]):
            response = api_client.post("/api/admin/extract", json={"url": "https://youtube.com/playlist?list=xyz"})
        assert response.status_code == 200

    def test_returns_found_urls(self, api_client):
        entries = [
            _video_entry("https://youtube.com/watch?v=a", "Video A"),
            _video_entry("https://youtube.com/watch?v=b", "Video B"),
        ]
        with patch("app.api.routes.extract_page_videos", return_value=entries):
            data = api_client.post(
                "/api/admin/extract", json={"url": "https://youtube.com/playlist?list=P"}
            ).json()
        assert data["urls"] == [e["url"] for e in entries]

    def test_returns_video_objects_with_titles(self, api_client):
        entry = _video_entry("https://youtube.com/watch?v=x", "My Lecture", duration=300.0)
        with patch("app.api.routes.extract_page_videos", return_value=[entry]):
            data = api_client.post(
                "/api/admin/extract", json={"url": "https://youtube.com/watch?v=x"}
            ).json()
        assert data["videos"][0]["title"] == "My Lecture"
        assert data["videos"][0]["duration"] == 300.0

    def test_message_contains_count(self, api_client):
        entries = [_video_entry(f"https://youtube.com/watch?v={i}") for i in range(3)]
        with patch("app.api.routes.extract_page_videos", return_value=entries):
            data = api_client.post(
                "/api/admin/extract", json={"url": "https://youtube.com/channel/x"}
            ).json()
        assert "3" in data["message"]

    def test_single_url_returned_for_single_video(self, api_client):
        url = "https://youtube.com/watch?v=single"
        with patch("app.api.routes.extract_page_videos", return_value=[_video_entry(url, "Single Video")]):
            data = api_client.post("/api/admin/extract", json={"url": url}).json()
        assert len(data["urls"]) == 1

    def test_enrichment_triggered_for_id_like_title(self, api_client):
        """Entries whose title looks like a raw YouTube video ID should be enriched."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        entry = {"url": url, "title": "dQw4w9WgXcQ", "thumbnail": "", "duration": 0}
        enriched_info = {"title": "Rick Astley - Never Gonna Give You Up", "thumbnail": "t.jpg", "duration": 213}
        with patch("app.api.routes.extract_page_videos", return_value=[entry]), \
             patch("app.api.routes.get_video_info", return_value=enriched_info):
            data = api_client.post("/api/admin/extract", json={"url": url}).json()
        assert data["videos"][0]["title"] == "Rick Astley - Never Gonna Give You Up"


# ---------------------------------------------------------------------------
# POST /api/admin/index-url  (SSE streaming)
# ---------------------------------------------------------------------------


class TestIndexUrlEndpoint:
    def test_returns_200(self, api_client):
        from app.api import routes

        routes._pipeline.process_url = MagicMock(
            return_value={"status": "success", "url": "https://example.com/v.mp4",
                          "title": "Test", "chunks": 5}
        )
        response = api_client.post(
            "/api/admin/index-url", json={"url": "https://example.com/v.mp4"}
        )
        assert response.status_code == 200

    def test_response_contains_done_sentinel(self, api_client):
        from app.api import routes

        routes._pipeline.process_url = MagicMock(
            return_value={"status": "success", "url": "u", "title": "T", "chunks": 3}
        )
        response = api_client.post("/api/admin/index-url", json={"url": "u"})
        assert "__DONE__" in response.text

    def test_already_indexed_message_in_stream(self, api_client):
        from app.api import routes

        routes._pipeline.process_url = MagicMock(
            return_value={"status": "already_indexed", "url": "u", "chunks": 0}
        )
        response = api_client.post("/api/admin/index-url", json={"url": "u"})
        assert "already" in response.text.lower() or "indexed" in response.text.lower()

    def test_error_status_message_in_stream(self, api_client):
        from app.api import routes

        routes._pipeline.process_url = MagicMock(
            return_value={"status": "error", "url": "u", "error": "Download failed", "chunks": 0}
        )
        response = api_client.post("/api/admin/index-url", json={"url": "u"})
        assert "error" in response.text.lower() or "download" in response.text.lower()

    def test_content_type_is_event_stream(self, api_client):
        from app.api import routes

        routes._pipeline.process_url = MagicMock(
            return_value={"status": "success", "url": "u", "title": "T", "chunks": 1}
        )
        response = api_client.post("/api/admin/index-url", json={"url": "u"})
        assert "text/event-stream" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# GET /api/local-video/{filename}
# ---------------------------------------------------------------------------


class TestLocalVideoEndpoint:
    def test_returns_404_for_missing_file(self, api_client):
        response = api_client.get("/api/local-video/nonexistent_video.mp4")
        assert response.status_code == 404

    def test_serves_existing_file(self, api_client, tmp_path):
        from app.api import routes
        from app.core.config import Config

        # Temporarily point the config data_dir to tmp_path
        original_cfg = routes._cfg
        try:
            routes._cfg = Config(data_dir=str(tmp_path))
            local_dir = tmp_path / "local_videos"
            local_dir.mkdir()
            video_file = local_dir / "test_video.mp4"
            video_file.write_bytes(b"fake mp4 content")

            response = api_client.get("/api/local-video/test_video.mp4")
            assert response.status_code == 200
        finally:
            routes._cfg = original_cfg
