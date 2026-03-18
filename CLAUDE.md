# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with uv (recommended)
uv sync
uv run seekr serve

# Or with pip
pip install -e .
seekr serve

# Start the web server (http://localhost:8000)
seekr serve --config config.yaml --port 8000 --reload   # dev mode

# Index local video files
seekr index ./videos/

# Add and index a remote video (YouTube, LinkedIn, Instagram, TikTok, MP4, playlist…)
seekr add-source https://youtube.com/watch?v=...
seekr add-source https://www.linkedin.com/posts/...
seekr add-source https://www.tiktok.com/@user/video/...
seekr add-source https://youtube.com/playlist?list=...  --extract-only  # preview only
```

Set API keys before running:
```bash
export OPENAI_API_KEY=sk-...    # required if embedding_provider: openai
export GEMINI_API_KEY=...       # required if embedding_provider: gemini
```

## Architecture

The app is split into four layers that are intentionally decoupled:

### Core (`app/core/`)
- `config.py` — `Config` dataclass loaded from `config.yaml`, overridden by env vars
- `embeddings.py` — Abstract `EmbeddingProvider` base class; `OpenAIEmbedding` and `GeminiEmbedding` implement it. Swap providers by changing `embedding_provider` in `config.yaml`
- `vector_store.py` — FAISS `IndexFlatIP` with L2-normalized vectors (cosine similarity). Persists to `data/faiss.index` + `data/metadata.json`. Metadata includes `video_url`, `start`, `end`, `text`, `video_title`, `video_thumbnail`
- `search.py` — Thin wrapper: embed query → FAISS search → return ranked metadata dicts with `score`

### Indexer (`app/indexer/`)
- `downloader.py` — yt-dlp wrappers for metadata, audio download (→ MP3), and extracting video URLs from playlist/channel pages
- `transcriber.py` — Lazy-loaded Whisper model; returns `[{text, start, end}]` dicts
- `segmenter.py` — `merge_segments()` collapses short Whisper segments into configurable-duration chunks
- `pipeline.py` — `IndexingPipeline` orchestrates the full flow: metadata → download audio → transcribe → chunk → embed → store. Tracks processed URLs by SHA-256 hash in `data/processed.json` to skip re-indexing. Accepts an `on_progress` callback used for SSE streaming

### API (`app/api/`)
- `routes.py` — FastAPI router. The `POST /api/admin/index-url` endpoint streams progress as SSE using `asyncio.Queue` + `loop.run_in_executor` (pipeline runs in a thread pool; progress messages are pushed via `loop.call_soon_threadsafe`)
- `models.py` — Pydantic v2 request/response models
- `routes._cfg`, `routes._search_engine`, `routes._pipeline` — module-level globals injected by `server.py` at startup

### Server & CLI
- `app/server.py` — `create_app(config_path)` factory wires all services together; also exports a module-level `app` for `uvicorn app.server:app`
- `main.py` — Typer CLI with `serve`, `index`, `add-source` commands; each command lazily imports and builds the pipeline

### Frontend (`app/ui/`)
- `index.html` — Search UI. Detects video type: YouTube URLs → iframe embed with `?start={sec}&autoplay=1`; MP4/WebM → HTML5 `<video>` with `currentTime` + `play()`. Result cards show timestamp badge, confidence %, and transcript snippet
- `admin.html` — Admin UI. Reads the SSE `index-url` response as a `ReadableStream` to stream progress into a terminal-style log. Supports indexing one or all extracted URLs sequentially

## Key Data Flow

```
User query
  → POST /api/search
  → SearchEngine.search()
  → OpenAIEmbedding.embed(query)
  → VectorStore.search() → FAISS top-k
  → Return [{video_url, start, text, score}]
  → Frontend seeks video to `start` seconds
```

```
Admin adds URL
  → POST /api/admin/index-url  (SSE stream)
  → IndexingPipeline.process_url()
  → yt-dlp download audio → Whisper transcribe
  → merge_segments() → embed_batch() → VectorStore.add()
```

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `embedding_provider` | `openai` | `openai` or `gemini` |
| `whisper_model` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `chunk_duration` | `120` | Seconds per indexed segment |
| `top_k` | `3` | Search results returned |
| `data_dir` | `./data` | Where FAISS index and files are stored |
| `cookies_from_browser` | `""` | Browser for cookie extraction (`firefox`, `chrome`, …) — needed for LinkedIn/Instagram/TikTok |
| `chat_model` | `""` | LLM model for RAG answers. Leave empty for plain search. E.g. `gpt-4o-mini` (OpenAI) or `gemini-2.0-flash` (Gemini) |

## Adding a New Embedding Provider

1. Subclass `EmbeddingProvider` in `app/core/embeddings.py`
2. Implement `embed()`, `embed_batch()`, and the `dimension` property
3. Add a branch in `get_embedding_provider()` for the new provider name
4. Set `embedding_provider: your_provider` in `config.yaml`
