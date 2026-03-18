"""Tests for app.core.vector_store — FAISS-backed persistence and search."""

import numpy as np
import pytest

from app.core.vector_store import VectorStore


def _make_meta(text: str, url: str = "https://example.com/v") -> dict:
    """Helper to build a minimal metadata dict."""
    return {"text": text, "video_url": url, "start": 0.0, "end": 5.0}


class TestEmptyStore:
    """Behaviour of a freshly created, empty VectorStore."""

    def test_total_segments_is_zero(self, vector_store: VectorStore):
        assert vector_store.total_segments == 0

    def test_unique_videos_is_empty(self, vector_store: VectorStore):
        assert vector_store.unique_videos == []

    def test_search_returns_empty_list(self, vector_store: VectorStore):
        query = [0.1] * 128
        results = vector_store.search(query, top_k=3)
        assert results == []


class TestAddAndSearch:
    """Tests for add() and search() with real FAISS vectors."""

    def test_add_increments_total_segments(
        self, vector_store: VectorStore, mock_embedder
    ):
        vector_store.add(
            [mock_embedder.embed("hello")],
            [_make_meta("hello")],
        )
        assert vector_store.total_segments == 1

    def test_search_returns_correct_metadata(
        self, vector_store: VectorStore, mock_embedder
    ):
        vec = mock_embedder.embed("gradient descent explanation")
        vector_store.add([vec], [_make_meta("gradient descent explanation")])

        results = vector_store.search(vec, top_k=1)
        assert len(results) == 1
        score, meta = results[0]
        assert meta["text"] == "gradient descent explanation"

    def test_search_scores_are_between_zero_and_one(
        self, vector_store: VectorStore, mock_embedder
    ):
        for i in range(5):
            vector_store.add(
                [mock_embedder.embed(f"segment {i}")],
                [_make_meta(f"segment {i}")],
            )
        results = vector_store.search(mock_embedder.embed("segment 2"), top_k=3)
        for score, _ in results:
            assert 0.0 <= score <= 1.0 + 1e-6  # allow tiny floating-point slack

    def test_top_k_limits_result_count(
        self, vector_store: VectorStore, mock_embedder
    ):
        for i in range(10):
            vector_store.add(
                [mock_embedder.embed(f"doc {i}")],
                [_make_meta(f"doc {i}")],
            )
        results = vector_store.search(mock_embedder.embed("doc"), top_k=4)
        assert len(results) == 4

    def test_top_k_capped_at_total_vectors(
        self, vector_store: VectorStore, mock_embedder
    ):
        """Requesting more results than there are vectors should not raise."""
        vector_store.add(
            [mock_embedder.embed("only one")],
            [_make_meta("only one")],
        )
        results = vector_store.search(mock_embedder.embed("only one"), top_k=100)
        assert len(results) == 1

    def test_best_match_is_first(self, vector_store: VectorStore, mock_embedder):
        """The segment most similar to the query should rank first."""
        target_text = "backpropagation in neural networks"
        target_vec = mock_embedder.embed(target_text)

        for label in ("random topic A", "random topic B", target_text):
            vector_store.add(
                [mock_embedder.embed(label)],
                [_make_meta(label)],
            )

        results = vector_store.search(target_vec, top_k=3)
        assert results[0][1]["text"] == target_text

    def test_batch_add(self, vector_store: VectorStore, mock_embedder):
        texts = ["alpha", "beta", "gamma"]
        embeddings = mock_embedder.embed_batch(texts)
        metadata = [_make_meta(t) for t in texts]
        vector_store.add(embeddings, metadata)
        assert vector_store.total_segments == 3


class TestPersistence:
    """Tests that FAISS index and metadata survive a reload."""

    def test_reload_restores_total_segments(
        self, tmp_data_dir: str, mock_embedder
    ):
        vs1 = VectorStore(tmp_data_dir)
        vs1.add([mock_embedder.embed("hello")], [_make_meta("hello")])

        vs2 = VectorStore(tmp_data_dir)  # fresh instance, loads from disk
        assert vs2.total_segments == 1

    def test_reload_restores_metadata(self, tmp_data_dir: str, mock_embedder):
        vs1 = VectorStore(tmp_data_dir)
        vs1.add(
            [mock_embedder.embed("important segment")],
            [_make_meta("important segment", url="https://youtube.com/watch?v=xyz")],
        )

        vs2 = VectorStore(tmp_data_dir)
        assert vs2.metadata[0]["text"] == "important segment"
        assert vs2.metadata[0]["video_url"] == "https://youtube.com/watch?v=xyz"

    def test_reload_allows_search(self, tmp_data_dir: str, mock_embedder):
        vec = mock_embedder.embed("persisted text")
        VectorStore(tmp_data_dir).add([vec], [_make_meta("persisted text")])

        vs2 = VectorStore(tmp_data_dir)
        results = vs2.search(vec, top_k=1)
        assert results[0][1]["text"] == "persisted text"


class TestUniqueVideos:
    """Tests for the unique_videos property."""

    def test_deduplicates_same_url(self, vector_store: VectorStore, mock_embedder):
        for _ in range(3):
            vector_store.add(
                [mock_embedder.embed("x")],
                [_make_meta("x", url="https://youtube.com/watch?v=same")],
            )
        assert vector_store.unique_videos == ["https://youtube.com/watch?v=same"]

    def test_preserves_insertion_order(
        self, vector_store: VectorStore, mock_embedder
    ):
        for url in ("url1", "url2", "url3"):
            vector_store.add(
                [mock_embedder.embed(url)],
                [_make_meta(url, url=url)],
            )
        assert vector_store.unique_videos == ["url1", "url2", "url3"]


class TestClearByUrl:
    """Tests for the clear_by_url() method."""

    def test_removes_matching_metadata(
        self, vector_store: VectorStore, mock_embedder
    ):
        vector_store.add(
            [mock_embedder.embed("a"), mock_embedder.embed("b")],
            [_make_meta("a", url="url1"), _make_meta("b", url="url2")],
        )
        removed = vector_store.clear_by_url("url1")
        assert removed == 1
        assert vector_store.total_segments == 1
        assert vector_store.metadata[0]["text"] == "b"

    def test_returns_zero_for_unknown_url(
        self, vector_store: VectorStore, mock_embedder
    ):
        vector_store.add([mock_embedder.embed("a")], [_make_meta("a")])
        removed = vector_store.clear_by_url("https://nonexistent.com/v")
        assert removed == 0
        assert vector_store.total_segments == 1

    def test_clears_all_segments_for_url(
        self, vector_store: VectorStore, mock_embedder
    ):
        for i in range(4):
            vector_store.add(
                [mock_embedder.embed(f"seg {i}")],
                [_make_meta(f"seg {i}", url="target_url")],
            )
        vector_store.add([mock_embedder.embed("keep")], [_make_meta("keep", url="other")])

        removed = vector_store.clear_by_url("target_url")
        assert removed == 4
        assert vector_store.total_segments == 1
