"""Seekr test suite.

Tests are organised by layer:

- ``test_config.py`` — configuration loading
- ``test_embeddings.py`` — embedding providers (mocked API calls)
- ``test_vector_store.py`` — FAISS vector store
- ``test_segmenter.py`` — transcript segment merging
- ``test_transcriber.py`` — Whisper transcription wrapper (mocked)
- ``test_pipeline.py`` — end-to-end indexing pipeline (mocked)
- ``test_api.py`` — FastAPI route handlers (in-process test client)

Run with::

    pytest tests/ -v
"""
