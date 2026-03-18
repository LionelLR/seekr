"""FastAPI route handlers for the Seekr REST API."""

import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Cookie, Form, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse

from .models import (
    AddSourceRequest,
    DeleteVideoRequest,
    ExtractedVideo,
    ExtractResponse,
    IndexStatusResponse,
    LoginRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SegmentItem,
    SegmentListResponse,
    SegmentUpdateRequest,
    VideoInfo,
)
from ..indexer.downloader import extract_page_videos, get_video_info

router = APIRouter()

# Populated by server.py on startup
_cfg = None
_search_engine = None
_pipeline = None
_chat_provider = None
_admin_password: str = ""
_admin_token: str = ""   # SHA-256 of password; empty when no password set


def _make_token(password: str) -> str:
    return hashlib.sha256(f"seekr:{password}".encode()).hexdigest()


def _check_admin(request: Request) -> None:
    """Raise 401 if admin password is set and the request has no valid session cookie."""
    if not _admin_password:
        return
    token = request.cookies.get("seekr_session", "")
    if token != _admin_token:
        raise HTTPException(401, "Admin authentication required")


def _services_ready() -> bool:
    return _search_engine is not None and _pipeline is not None


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.get("/admin/auth-status")
def auth_status() -> dict:
    """Return whether admin password protection is enabled."""
    return {"protected": bool(_admin_password)}


@router.post("/admin/login")
def admin_login(req: LoginRequest, response: Response) -> dict:
    if not _admin_password:
        return {"ok": True}
    if req.password != _admin_password:
        raise HTTPException(401, "Wrong password")
    response.set_cookie(
        "seekr_session", _admin_token,
        httponly=True, samesite="strict", max_age=86400 * 30,
    )
    return {"ok": True}


@router.post("/admin/logout")
def admin_logout(response: Response) -> dict:
    response.delete_cookie("seekr_session")
    return {"ok": True}


# ── Status & Search ───────────────────────────────────────────────────────────

@router.get("/status", response_model=IndexStatusResponse)
def status() -> IndexStatusResponse:
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    vs = _pipeline.vector_store
    provider = _cfg.embedding_provider if _cfg else ""
    emb_model = (
        _cfg.local_embedding_model if provider == "local"
        else provider  # "openai" or "gemini"
    ) if _cfg else ""
    return IndexStatusResponse(
        total_segments=vs.total_segments,
        indexed_videos=_pipeline.indexed_count,
        videos=vs.unique_videos,
        video_list=[VideoInfo(**v) for v in vs.video_list],
        ai_enabled=_chat_provider is not None,
        chat_model=_cfg.chat_model if _cfg else "",
        embedding_provider=provider,
        embedding_model=emb_model,
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    if _pipeline.vector_store.total_segments == 0:
        return SearchResponse(query=req.query, results=[])
    results = _search_engine.search(req.query, req.top_k)
    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                video_url=r["video_url"],
                source_url=r.get("source_url", r["video_url"]),
                platform=r.get("platform", ""),
                video_title=r.get("video_title", ""),
                video_thumbnail=r.get("video_thumbnail", ""),
                text=r["text"],
                start=r["start"],
                end=r["end"],
                score=r["score"],
            )
            for r in results
        ],
    )


# ── AI chat (RAG) ─────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat_answer(req: SearchRequest) -> StreamingResponse:
    """RAG endpoint: vector search → LLM answer streamed as SSE.

    Event types sent over the stream:
    - ``sources``: JSON array of matching segments (sent first, before LLM starts).
    - ``token``:   One LLM response token.
    - ``error``:   Error message string.
    - ``done``:    Stream finished (no data).
    """
    import json
    import threading

    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    if not _chat_provider:
        raise HTTPException(400, "AI answers are not configured. Set chat_model in config.yaml.")
    if _pipeline.vector_store.total_segments == 0:
        raise HTTPException(400, "No videos indexed yet.")

    top_k = max(req.top_k, 5)  # use at least 5 sources for better context
    raw = _search_engine.search(req.query, top_k)
    sources = [
        {
            "video_url":       r["video_url"],
            "source_url":      r.get("source_url", r["video_url"]),
            "platform":        r.get("platform", ""),
            "video_title":     r.get("video_title", ""),
            "video_thumbnail": r.get("video_thumbnail", ""),
            "text":            r["text"],
            "start":           r["start"],
            "end":             r["end"],
            "score":           r["score"],
        }
        for r in raw
    ]

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_llm() -> None:
        try:
            for token in _chat_provider.stream_answer(req.query, sources):
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=run_llm, daemon=True).start()

    async def event_stream() -> AsyncGenerator[str, None]:
        # Send sources first so the UI can render them while the LLM streams
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        while True:
            try:
                kind, data = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield "data: {\"type\":\"token\",\"text\":\"…\"}\n\n"
                continue
            if kind == "token":
                yield f"data: {json.dumps({'type': 'token', 'text': data})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'text': data})}\n\n"
                break
            else:
                yield "data: {\"type\":\"done\"}\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Admin: extract & index ────────────────────────────────────────────────────

_YT_ID_RE = __import__("re").compile(r"^[A-Za-z0-9_-]{11}$")


@router.post("/admin/extract", response_model=ExtractResponse)
async def extract_videos(req: AddSourceRequest, request: Request) -> ExtractResponse:
    """Extract video list from a playlist, channel, or single video URL.

    After flat extraction, entries whose title is missing or looks like a raw
    YouTube video ID are enriched via parallel ``get_video_info`` calls so the
    admin UI can display real titles, thumbnails, and durations.
    """
    _check_admin(request)
    loop = asyncio.get_event_loop()
    entries = await loop.run_in_executor(None, extract_page_videos, req.url)

    # Identify entries that need enrichment: title is absent, equals the URL,
    # or is a bare 11-char YouTube video ID returned by flat-mode extraction.
    cookies = _cfg.cookies_from_browser if _cfg else ""

    async def _enrich(entry: dict) -> dict:
        title = entry.get("title", "")
        needs_enrich = (
            not title
            or title == entry.get("url", "")
            or _YT_ID_RE.match(title)
        )
        if not needs_enrich:
            return entry
        info = await loop.run_in_executor(None, get_video_info, entry["url"], cookies)
        if "error" not in info:
            return {
                **entry,
                "title": info.get("title") or entry.get("title") or entry["url"],
                "thumbnail": entry.get("thumbnail") or info.get("thumbnail", ""),
                "duration": entry.get("duration") or info.get("duration", 0),
            }
        return entry

    enriched = await asyncio.gather(*[_enrich(e) for e in entries])
    videos = [ExtractedVideo(**e) for e in enriched]
    return ExtractResponse(
        videos=videos,
        urls=[v.url for v in videos],
        message=f"Found {len(videos)} video(s)",
    )


@router.post("/admin/index-url")
async def index_url(req: AddSourceRequest, request: Request) -> StreamingResponse:
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_progress(msg: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    async def run_pipeline():
        result = await loop.run_in_executor(
            None, lambda: _pipeline.process_url(req.url, on_progress, force_video=req.download_video)
        )
        await queue.put(None)
        return result

    task = asyncio.create_task(run_pipeline())

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                yield "data: still processing...\n\n"
                continue
            if msg is None:
                break
            yield f"data: {msg}\n\n"

        result = await task
        status = result.get("status", "unknown")
        chunks = result.get("chunks", 0)
        title = result.get("title", "")
        if status == "success":
            yield f"data: Done! {chunks} chunks indexed for \"{title}\"\n\n"
        elif status == "already_indexed":
            yield f"data: Already indexed: {req.url}\n\n"
        else:
            error = result.get("error", "Unknown error")
            yield f"data: Error: {error}\n\n"
        yield "data: __DONE__\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Admin: video management ───────────────────────────────────────────────────

@router.post("/admin/cleanup-media")
def cleanup_media(request: Request) -> dict:
    """Delete downloaded audio/video files from disk while keeping the index."""
    _check_admin(request)
    if not _cfg:
        raise HTTPException(503, "Not initialized")
    deleted = 0
    freed = 0
    for subdir in ("audio_tmp", "local_videos"):
        d = Path(_cfg.data_dir) / subdir
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
                    deleted += 1
    return {"deleted_files": deleted, "freed_bytes": freed}


@router.post("/admin/reset-index")
def reset_index(request: Request) -> dict:
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    _pipeline.reset()
    return {"ok": True}


@router.delete("/admin/video")
def delete_video(req: DeleteVideoRequest, request: Request) -> dict:
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    removed = _pipeline.vector_store.delete_by_url(req.video_url)
    _pipeline.remove_url(req.video_url)
    return {"removed_segments": removed, "video_url": req.video_url}


@router.get("/admin/segments", response_model=SegmentListResponse)
def get_segments(video_url: str, request: Request) -> SegmentListResponse:
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    pairs = _pipeline.vector_store.get_by_url(video_url)
    return SegmentListResponse(
        video_url=video_url,
        segments=[
            SegmentItem(idx=i, text=m["text"], start=m["start"], end=m["end"])
            for i, m in pairs
        ],
    )


@router.put("/admin/segment/{idx}")
def update_segment(idx: int, req: SegmentUpdateRequest, request: Request) -> dict:
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    new_embedding = None
    if req.re_embed and req.text:
        new_embedding = _pipeline.embedder.embed(req.text)
    ok = _pipeline.vector_store.update_segment(
        idx, text=req.text, start=req.start, end=req.end, new_embedding=new_embedding
    )
    if not ok:
        raise HTTPException(404, "Segment not found")
    return {"ok": True, "idx": idx}


@router.delete("/admin/segment/{idx}")
def delete_segment(idx: int, request: Request) -> dict:
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")
    ok = _pipeline.vector_store.delete_segment(idx)
    if not ok:
        raise HTTPException(404, "Segment not found")
    return {"ok": True}


# ── Voice transcription (server-side Whisper fallback) ───────────────────────

@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = "",
) -> dict:
    """Transcribe uploaded audio using the server's Whisper model.

    Accepts any audio format supported by Whisper/ffmpeg (WebM, OGG, WAV, MP4…).
    Pass ``language`` (ISO 639-1 code, e.g. ``"fr"``) to skip auto-detection
    and force transcription in that language.
    Returns the concatenated transcript text.
    """
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")

    data = await audio.read()
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    lang = language.strip() or None
    try:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, lambda: _pipeline.transcriber.transcribe(tmp_path, language=lang)
        )
        text = " ".join(s["text"] for s in segments).strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"text": text}


# ── Settings (read / write config.yaml) ──────────────────────────────────────

_SETTINGS_FIELDS = {
    # field: (label, needs_restart)
    "embedding_provider":   ("Embedding provider",    True),
    "local_embedding_model":("Local embedding model", True),
    "chat_model":           ("Chat / AI model",       False),
    "whisper_model":        ("Whisper model",         True),
    "top_k":                ("Default top-k results", False),
    "chunk_duration":       ("Chunk duration (s)",    True),
    "openai_api_key":       ("OpenAI API key",        False),
    "gemini_api_key":       ("Gemini API key",        False),
    "cookies_from_browser": ("Cookies from browser",  True),
    "vocabulary":           ("Transcription vocabulary", False),
}

def _mask(key: str, val: str) -> str:
    """Return a masked representation of an API key for display."""
    if not val:
        return ""
    if key in ("openai_api_key", "gemini_api_key"):
        return val[:6] + "••••••••" if len(val) > 6 else "••••••••"
    return val


@router.get("/admin/settings")
def get_settings(request: Request) -> dict:
    """Return current config values (API keys are masked)."""
    _check_admin(request)
    if not _cfg:
        raise HTTPException(503, "Not initialized")
    return {
        "embedding_provider":    _cfg.embedding_provider,
        "local_embedding_model": _cfg.local_embedding_model,
        "chat_model":            _cfg.chat_model,
        "whisper_model":         _cfg.whisper_model,
        "top_k":                 _cfg.top_k,
        "chunk_duration":        _cfg.chunk_duration,
        "openai_api_key":        _mask("openai_api_key", _cfg.openai_api_key),
        "gemini_api_key":        _mask("gemini_api_key", _cfg.gemini_api_key),
        "cookies_from_browser":  _cfg.cookies_from_browser,
        "vocabulary":            _cfg.vocabulary,
    }


@router.post("/admin/settings")
def save_settings(body: dict, request: Request) -> dict:
    """Persist settings to config.yaml and hot-reload what can be changed live."""
    import yaml
    import os
    _check_admin(request)
    if not _cfg:
        raise HTTPException(503, "Not initialized")

    config_path = os.environ.get("SEEKR_CONFIG", "config.yaml")

    # Load existing YAML (preserve keys we don't manage)
    existing: dict = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            existing = yaml.safe_load(f) or {}

    needs_restart = False
    for field, (_, restart) in _SETTINGS_FIELDS.items():
        if field not in body:
            continue
        val = body[field]
        # Skip blank API key submissions — means "keep existing"
        if field in ("openai_api_key", "gemini_api_key") and (not val or "•" in str(val)):
            continue
        # Type coercion
        if field in ("top_k",):
            try:
                val = int(val)
            except (ValueError, TypeError):
                continue
        if field in ("chunk_duration",):
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
        if field == "vocabulary":
            # Accept a list from JSON or a newline-separated string from the UI
            if isinstance(val, str):
                val = [v.strip() for v in val.splitlines() if v.strip()]
            elif not isinstance(val, list):
                continue
        existing[field] = val
        setattr(_cfg, field, val)
        if restart:
            needs_restart = True

    # Write back to config.yaml
    with open(config_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

    # Hot-reload what we can without a restart
    # 1. API keys → env vars
    if _cfg.openai_api_key:
        os.environ["OPENAI_API_KEY"] = _cfg.openai_api_key
    if _cfg.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = _cfg.gemini_api_key

    # 2. chat_model → swap provider
    global _chat_provider
    from ..core.chat import get_chat_provider
    _chat_provider = get_chat_provider(_cfg)

    # 3. top_k → update search engine
    if _search_engine:
        _search_engine.top_k = _cfg.top_k

    # 4. vocabulary → update transcriber initial_prompt (hot-reload, no restart needed)
    if _pipeline:
        _pipeline.transcriber.initial_prompt = ", ".join(_cfg.vocabulary)

    return {"ok": True, "needs_restart": needs_restart}


# ── File upload indexing ──────────────────────────────────────────────────────

@router.post("/admin/index-file")
async def index_file(request: Request, file: UploadFile = File(...)) -> StreamingResponse:
    """Index an uploaded PDF or audio file with SSE progress streaming."""
    _check_admin(request)
    if not _services_ready():
        raise HTTPException(503, "Services not initialized")

    import tempfile
    suffix = Path(file.filename or "upload").suffix.lower()
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    original_name = file.filename or "upload"
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_progress(msg: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    async def run_pipeline():
        result = await loop.run_in_executor(
            None, lambda: _pipeline.process_url(tmp_path, on_progress)
        )
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        await queue.put(None)
        return result

    task = asyncio.create_task(run_pipeline())

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=300.0)
            except asyncio.TimeoutError:
                yield "data: still processing...\n\n"
                continue
            if msg is None:
                break
            yield f"data: {msg}\n\n"

        result = await task
        status = result.get("status", "unknown")
        chunks = result.get("chunks", 0)
        title = result.get("title", original_name)
        if status == "success":
            yield f"data: Done! {chunks} chunks indexed for \"{title}\"\n\n"
        elif status == "already_indexed":
            yield f"data: Already indexed: {original_name}\n\n"
        else:
            error = result.get("error", "Unknown error")
            yield f"data: Error: {error}\n\n"
        yield "data: __DONE__\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Static file serving ───────────────────────────────────────────────────────

@router.get("/local-video/{filename}")
def serve_local_video(filename: str) -> FileResponse:
    if not _cfg:
        raise HTTPException(503, "Not initialized")
    path = Path(_cfg.data_dir) / "local_videos" / filename
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(str(path))


@router.get("/local-pdf/{filename}")
def serve_local_pdf(filename: str) -> FileResponse:
    if not _cfg:
        raise HTTPException(503, "Not initialized")
    path = Path(_cfg.data_dir) / "local_pdfs" / filename
    if not path.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(str(path), media_type="application/pdf")


@router.get("/local-audio/{filename}")
def serve_local_audio(filename: str) -> FileResponse:
    if not _cfg:
        raise HTTPException(503, "Not initialized")
    path = Path(_cfg.data_dir) / "local_audio" / filename
    if not path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(str(path))
