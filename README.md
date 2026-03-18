# Seekr

Semantic search inside videos, audio, and documents. Ask a question and jump to the exact timestamp where the answer is spoken — or find the page in a PDF where it's written.

Seekr transcribes media with Whisper, splits transcripts into chunks, embeds them locally (or via OpenAI/Gemini), stores them in a FAISS index, and lets you search them via a clean web UI. Supports YouTube, LinkedIn, Instagram, TikTok, direct video/audio files, and PDFs.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** — required by Whisper and yt-dlp for audio/video processing
- An **OpenAI API key** or **Google Gemini API key** — or use the free **local** provider (no key needed, recommended to start)

---

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it once, then:

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the project
cd /path/to/seekr

# Create the virtual environment and install all dependencies
uv sync

# Copy the example config and edit it
cp config.example.yaml config.yaml

# Run the server (no API key needed with the default local embedding provider)
uv run seekr serve

# Or activate the venv and use seekr directly
source .venv/bin/activate
seekr serve
```

### Using pip

```bash
cd /path/to/seekr
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
seekr serve
```

---

## Arch Linux Setup

```bash
# System dependencies
sudo pacman -S python python-pip ffmpeg

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or via pacman (AUR):
# yay -S uv

# Clone and set up
cd /path/to/seekr
uv sync
cp config.example.yaml config.yaml

# Start the server
uv run seekr serve
```

If you plan to use LinkedIn/Instagram/TikTok and want cookie extraction from your browser:

```bash
# Firefox is the most reliable for yt-dlp cookie extraction on Arch
sudo pacman -S firefox

# Then in config.yaml:
# cookies_from_browser: firefox
```

> **Tip:** On Arch, `openai-whisper` depends on `torch`. If you have an NVIDIA GPU, install `python-pytorch-cuda` from the repos for faster transcription. For CPU-only, `uv sync` pulls the standard CPU build automatically.

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit it. The `config.yaml` file is gitignored and will never be committed.

```yaml
# Embedding: "local" runs on your CPU — no API key needed (default)
embedding_provider: local
local_embedding_model: all-MiniLM-L6-v2   # any sentence-transformers model

# AI chat answers (RAG) — optional; leave empty for plain vector search
chat_model: ''          # e.g. gpt-4o-mini or gemini-2.0-flash

whisper_model: base     # tiny | base | small | medium | large
chunk_duration: 120     # seconds per indexed chunk
top_k: 3                # default number of search results
data_dir: ./data        # where to store the FAISS index and files

# For LinkedIn, Instagram, TikTok — use your browser's session
# Options: chrome | firefox | chromium | safari | edge | brave | opera
cookies_from_browser: ""
```

Set API keys via environment variables (they take precedence over `config.yaml`):

```bash
export OPENAI_API_KEY="sk-..."
# or:
export GEMINI_API_KEY="AIza..."
```

#### Free option — no API key required

The default `embedding_provider: local` runs entirely on your CPU. Seekr downloads a small model (~80 MB) on first run and embeds everything locally — no account or internet connection needed after that.

---

## Supported Content Types

| Type | Examples |
|------|---------|
| YouTube | Videos, playlists, channels |
| TikTok | Public and private videos |
| Instagram | Reels, posts |
| LinkedIn | Posts with video |
| Direct video file | `.mp4`, `.webm`, `.avi`, `.mov`, `.mkv` |
| Direct audio file | `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.opus` |
| PDF | Local files via the Admin panel |

For platforms that require login, Seekr uses yt-dlp's cookie extraction to read your active browser session. Set `cookies_from_browser` to your browser name in `config.yaml`, make sure you are logged in to the platform in that browser, then index away.

---

## CLI Usage

### Start the web server

```bash
seekr serve
seekr serve --host 127.0.0.1 --port 8080 --reload   # dev mode
```

Open `http://localhost:8000` to search and `http://localhost:8000/admin` to manage sources.

### Index a local directory

```bash
seekr index /path/to/videos/
```

Supported formats: `.mp4`, `.webm`, `.avi`, `.mov`, `.mkv`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.opus`, `.pdf`

### Add a remote video source

```bash
# Single video
seekr add-source "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Playlist or channel
seekr add-source "https://www.youtube.com/playlist?list=PLxxxxxxxx"

# LinkedIn post (requires cookies_from_browser set in config.yaml)
seekr add-source "https://www.linkedin.com/posts/..."

# TikTok video
seekr add-source "https://www.tiktok.com/@user/video/..."

# Instagram reel
seekr add-source "https://www.instagram.com/reel/..."

# Preview URLs without indexing
seekr add-source "https://www.youtube.com/playlist?list=PLxxxxxxxx" --extract-only
```

---

## Web UI

### Search (`/`)

- Type a question in the search bar
- Results show the title, timestamp, transcript snippet, and confidence score
- Click a result card to load the media player at that exact moment
- YouTube embeds via iframe; LinkedIn/Instagram/TikTok/local files use the HTML5 player
- PDFs open at the matching page

### Admin (`/admin`)

- Paste any URL and click **Extract** to discover all videos in a playlist/channel/profile (shows titles, thumbnails, and durations)
- Click **Index Now** to index a URL directly, or **Index** per-video after extracting
- **Index All** processes all extracted URLs sequentially
- Upload a local **PDF** or **audio/video file** directly from the panel
- Choose between **Audio only** (faster, iframe player) or **Video MP4** (in-app HTML5 player) mode
- Progress streams live in the log with percentage indicators and elapsed time
- A warning banner appears when a social platform URL is detected, reminding you to configure `cookies_from_browser`

---

## How It Works

```
User query
  → POST /api/search
  → embed query (local / OpenAI / Gemini)
  → FAISS cosine similarity search
  → return top-k [{title, url, start, text, score}]
  → frontend seeks player to `start` seconds
```

**Indexing pipeline:**

1. **Identify** — detect content type: YouTube, social video, local video/audio, or PDF
2. **Download** — yt-dlp fetches audio (YouTube) or full video (social platforms)
3. **Extract text** — Whisper transcribes audio/video into timestamped segments; PDFs are parsed directly
4. **Chunk** — segments are merged into configurable-length chunks (default 120 s)
5. **Embed** — chunk text is embedded (local `all-MiniLM-L6-v2`, OpenAI `text-embedding-3-small`, or Gemini)
6. **Store** — embeddings + metadata saved in a FAISS flat index on disk
7. **Deduplicate** — processed URLs tracked by SHA-256 hash; re-indexing the same URL is skipped

---

## Data Layout

```
data/
├── faiss.index        # FAISS vector index
├── metadata.json      # Per-chunk metadata (text, timestamps, source info)
├── processed.json     # Hashes of already-indexed sources (avoids re-indexing)
├── audio_tmp/         # Temporary audio files (auto-cleaned after transcription)
└── local_videos/      # Downloaded social-platform videos + local file copies
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/status` | Index statistics (chunk count, vector dimensions) |
| `POST` | `/api/search` | Semantic search `{query, top_k}` |
| `POST` | `/api/chat` | RAG chat answer `{query, top_k}` (requires `chat_model` set) |
| `POST` | `/api/admin/extract` | Extract video list from a playlist/channel/profile `{url}` |
| `POST` | `/api/admin/index-url` | Index a URL with SSE progress stream `{url, download_video}` |
| `POST` | `/api/admin/upload-pdf` | Upload and index a PDF file |
| `POST` | `/api/admin/upload-audio` | Upload and index a local audio/video file |
| `GET`  | `/api/admin/settings` | Read current config (sanitised — no API keys) |
| `GET`  | `/api/local-video/{filename}` | Serve a locally stored video file |
| `GET`  | `/api/local-audio/{filename}` | Serve a locally stored audio file |
| `GET`  | `/api/local-pdf/{filename}` | Serve a locally stored PDF file |
