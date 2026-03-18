"""Shared pytest fixtures for the Seekr test suite.

Fixtures are organised from smallest to largest:

- Primitive helpers (``tmp_data_dir``, ``sample_wav_path``)
- Fake/mock services (``mock_embedder``, ``vector_store``)
- Composite fixtures for API tests (``api_client``)
"""

import struct
import wave
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.testclient import TestClient

from app.core.config import Config
from app.core.embeddings import EmbeddingProvider
from app.core.vector_store import VectorStore
from app.indexer.pipeline import IndexingPipeline
from app.indexer.transcriber import Transcriber


# ---------------------------------------------------------------------------
# Directories & files
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> str:
    """Return a temporary directory path suitable for a VectorStore."""
    return str(tmp_path / "data")


@pytest.fixture
def sample_wav_path(tmp_path: Path) -> str:
    """Create a minimal 1-second silent WAV file and return its path.

    The file is valid enough for path-passing tests; it does **not**
    contain actual speech.  Whisper is always mocked before any real
    transcription would occur.
    """
    wav_path = tmp_path / "sample.wav"
    n_frames = 16_000  # 1 second at 16 kHz
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return str(wav_path)


@pytest.fixture
def sample_video_dir(tmp_path: Path) -> Path:
    """Create a directory containing two dummy MP4 files (zero-byte stubs)."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    for name in ("lecture_01.mp4", "lecture_02.mp4"):
        (video_dir / name).write_bytes(b"\x00" * 16)
    return video_dir


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


SAMPLE_SEGMENTS = [
    {"text": "Welcome to this machine learning course.", "start": 0.0, "end": 5.0},
    {"text": "Today we discuss gradient descent.", "start": 5.0, "end": 10.0},
    {"text": "Gradient descent minimises the loss function.", "start": 10.0, "end": 15.0},
    {"text": "We adjust weights using the gradient.", "start": 15.0, "end": 20.0},
    {"text": "Backpropagation computes the gradients.", "start": 20.0, "end": 25.0},
    {"text": "The learning rate controls the step size.", "start": 25.0, "end": 30.0},
]

SAMPLE_QUERIES = [
    "what is gradient descent?",
    "how does backpropagation work?",
    "what is the learning rate?",
    "neural network training",
]


@pytest.fixture
def sample_segments() -> List[Dict[str, Any]]:
    """Return a small list of Whisper-style transcript segments."""
    return list(SAMPLE_SEGMENTS)


# ---------------------------------------------------------------------------
# Mock embedding provider
# ---------------------------------------------------------------------------


class _DeterministicEmbedder(EmbeddingProvider):
    """A fast, deterministic embedding provider for testing.

    Produces reproducible 128-dimensional float vectors derived from a
    seeded ``numpy`` RNG.  Vectors for the same text are always identical,
    ensuring consistent similarity rankings across test runs.
    """

    DIM = 128

    def embed(self, text: str) -> List[float]:
        seed = abs(hash(text)) % (2**31)
        rng = np.random.default_rng(seed)
        return rng.random(self.DIM).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self.DIM


@pytest.fixture
def mock_embedder() -> _DeterministicEmbedder:
    """Return a deterministic, dependency-free embedding provider."""
    return _DeterministicEmbedder()


# ---------------------------------------------------------------------------
# Core services
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_store(tmp_data_dir: str) -> VectorStore:
    """Return a fresh, empty VectorStore backed by a temp directory."""
    return VectorStore(tmp_data_dir)


@pytest.fixture
def populated_vector_store(
    tmp_data_dir: str,
    mock_embedder: _DeterministicEmbedder,
) -> VectorStore:
    """Return a VectorStore pre-loaded with SAMPLE_SEGMENTS data."""
    vs = VectorStore(tmp_data_dir)
    texts = [s["text"] for s in SAMPLE_SEGMENTS]
    embeddings = mock_embedder.embed_batch(texts)
    metadata = [
        {
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"],
            "video_url": "https://youtube.com/watch?v=ml_course",
            "video_title": "ML Course Episode 1",
            "video_thumbnail": "",
            "platform": "youtube",
        }
        for seg in SAMPLE_SEGMENTS
    ]
    vs.add(embeddings, metadata)
    return vs


@pytest.fixture
def mock_transcriber() -> MagicMock:
    """Return a MagicMock transcriber that returns SAMPLE_SEGMENTS."""
    tr = MagicMock(spec=Transcriber)
    tr.transcribe.return_value = list(SAMPLE_SEGMENTS)
    return tr


@pytest.fixture
def pipeline(
    tmp_data_dir: str,
    mock_embedder: _DeterministicEmbedder,
    mock_transcriber: MagicMock,
) -> IndexingPipeline:
    """Return an IndexingPipeline wired with mock/in-memory dependencies."""
    vs = VectorStore(tmp_data_dir)
    return IndexingPipeline(
        mock_embedder,
        vs,
        mock_transcriber,
        chunk_duration=10.0,
        data_dir=tmp_data_dir,
    )


# ---------------------------------------------------------------------------
# API test client
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(
    tmp_data_dir: str,
    mock_embedder: _DeterministicEmbedder,
) -> TestClient:
    """Return a Starlette TestClient backed by a minimal FastAPI app.

    Services are wired with the deterministic mock embedder and an empty
    in-memory VectorStore.  The ``app.api.routes`` module-level globals
    are patched before the TestClient is created.
    """
    from app.api import routes
    from app.core.search import SearchEngine

    vs = VectorStore(tmp_data_dir)
    cfg = Config(data_dir=tmp_data_dir)
    search_engine = SearchEngine(mock_embedder, vs, top_k=3)
    mock_tr = MagicMock(spec=Transcriber)
    mock_tr.transcribe.return_value = list(SAMPLE_SEGMENTS)
    pl = IndexingPipeline(
        mock_embedder, vs, mock_tr, chunk_duration=10.0, data_dir=tmp_data_dir
    )

    routes._cfg = cfg
    routes._search_engine = search_engine
    routes._pipeline = pl

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(routes.router, prefix="/api")

    return TestClient(app, raise_server_exceptions=True)
