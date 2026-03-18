"""End-to-end video indexing pipeline.

The :class:`IndexingPipeline` orchestrates the full workflow for a single
video URL:

1. Fetch video metadata (title, thumbnail, …)
2. Download audio (YouTube / direct) **or** the full video
   (LinkedIn / Instagram / TikTok)
3. Transcribe with Whisper
4. Merge short segments into configurable-duration chunks
5. Generate embeddings for each chunk
6. Store vectors + metadata in the FAISS index

Already-indexed URLs are tracked by a SHA-256 hash written to
``data/processed.json`` so that re-running the pipeline is idempotent.
"""

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .downloader import (
    NEEDS_LOCAL_SERVING,
    download_audio,
    download_file,
    download_video,
    get_platform,
    get_video_info,
    is_audio_file,
    is_local_file,
    is_pdf_file,
)
from .pdf_extractor import extract_pdf_segments, get_pdf_info
from .segmenter import merge_segments
from .transcriber import Transcriber
from ..core.embeddings import EmbeddingProvider
from ..core.vector_store import VectorStore


class IndexingPipeline:
    """Orchestrates downloading, transcription, embedding, and storage.

    A single :class:`IndexingPipeline` instance is shared across the web
    server and the CLI.  It is safe to call :meth:`process_url` from a
    background thread (the SSE endpoint does this via
    ``loop.run_in_executor``).

    Platform routing:

    - **YouTube / direct URLs** — audio-only download via yt-dlp, audio
      file deleted after transcription.
    - **LinkedIn / Instagram / TikTok** — full MP4 download, file kept
      in ``data/local_videos/`` and served via the ``/api/local-video/``
      endpoint.
    - **Local files** — copied to ``data/local_videos/`` and served with
      the same endpoint.

    Args:
        embedder: Embedding provider for generating chunk vectors.
        vector_store: FAISS store where embeddings are persisted.
        transcriber: Whisper-based transcriber for audio/video files.
        chunk_duration: Target chunk length in seconds passed to
            :func:`~app.indexer.segmenter.merge_segments`.
        data_dir: Root directory for all pipeline artefacts.
        cookies_from_browser: Browser name for yt-dlp cookie extraction,
            forwarded to all downloader functions.

    Example:
        >>> pipeline = IndexingPipeline(embedder, vs, transcriber)
        >>> result = pipeline.process_url("https://youtu.be/abc")
        >>> result["status"]
        'success'
        >>> result["chunks"]
        12
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        transcriber: Transcriber,
        chunk_duration: float = 120.0,
        min_chunk_duration: float = 10.0,
        data_dir: str = "./data",
        cookies_from_browser: str = "",
    ) -> None:
        """Initialise the pipeline and load the set of already-processed URLs.

        Args:
            embedder: Embedding backend.
            vector_store: Persistent FAISS store.
            transcriber: Whisper transcription wrapper.
            chunk_duration: Maximum seconds per chunk; split at sentence
                boundaries after this duration.
            min_chunk_duration: Minimum seconds before splitting at a
                question boundary.
            data_dir: Filesystem root for index, audio temp files, and
                downloaded videos.
            cookies_from_browser: Browser for yt-dlp cookie extraction.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.transcriber = transcriber
        self.chunk_duration = chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.cookies_from_browser = cookies_from_browser
        self.data_dir = Path(data_dir)
        self._audio_dir = self.data_dir / "audio_tmp"
        self._local_videos_dir = self.data_dir / "local_videos"
        self._local_pdfs_dir = self.data_dir / "local_pdfs"
        self._local_audio_dir = self.data_dir / "local_audio"
        self._processed_path = self.data_dir / "processed.json"
        self._processed: set = self._load_processed()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_processed(self) -> set:
        """Load the set of processed URL hashes from disk.

        Returns:
            A Python ``set`` of hex-encoded SHA-256 prefix strings.
            Returns an empty set if the file does not exist yet.
        """
        if self._processed_path.exists():
            with open(self._processed_path) as f:
                return set(json.load(f))
        return set()

    def _save_processed(self) -> None:
        """Persist the current set of processed URL hashes to disk."""
        with open(self._processed_path, "w") as f:
            json.dump(list(self._processed), f)

    def _url_id(self, url: str) -> str:
        """Compute a short, stable identifier for *url*.

        Uses the first 16 hex characters of a SHA-256 digest to avoid
        filename-length issues while remaining collision-resistant enough
        for typical workloads.

        Args:
            url: Any URL or filesystem path string.

        Returns:
            A 16-character lowercase hex string.
        """
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _transcribe_with_heartbeat(
        self,
        media_path: str,
        log: Callable[[str], None],
        start_pct: int,
        end_pct: int,
    ) -> List[Dict]:
        """Run Whisper transcription with periodic progress heartbeats.

        Since Whisper has no built-in progress callback, a background thread
        emits ``[N%] Transcribing… (Xs elapsed)`` messages every 4 seconds
        while the transcription blocks.

        Args:
            media_path: Path to the audio or video file.
            log: Progress callback to emit heartbeat messages.
            start_pct: Percentage to show at the start of transcription.
            end_pct: Percentage ceiling (never exceeded during heartbeats).

        Returns:
            Raw Whisper segment list from :meth:`Transcriber.transcribe`.
        """
        result: List[Dict] = []
        exc: List[BaseException] = []
        done = threading.Event()

        pct_range = end_pct - start_pct

        def _heartbeat() -> None:
            start = time.monotonic()
            tick = 0
            while not done.wait(4.0):
                tick += 1
                elapsed = int(time.monotonic() - start)
                # Creep slowly from start_pct toward (end_pct - 5) so the
                # final jump to end_pct comes when transcription actually ends.
                pct = min(start_pct + (pct_range * tick) // 12, end_pct - 5)
                log(f"[{pct}%] Transcribing… ({elapsed}s elapsed)")

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        try:
            result = self.transcriber.transcribe(media_path)
        except Exception as e:  # noqa: BLE001
            exc.append(e)
        finally:
            done.set()
            t.join(timeout=1.0)

        if exc:
            raise exc[0]
        return result

    def is_indexed(self, url: str) -> bool:
        """Return ``True`` if *url* has already been indexed.

        Args:
            url: Video URL to check.

        Returns:
            ``True`` when the URL's hash is present in the processed set.
        """
        return self._url_id(url) in self._processed

    def remove_url(self, url: str) -> None:
        """Remove a URL from the processed set so it can be re-indexed."""
        uid = self._url_id(url)
        self._processed.discard(uid)
        self._save_processed()

    def reset(self) -> None:
        """Wipe all indexed data: vectors, metadata, and processed URL log."""
        self.vector_store.reset()
        self._processed = set()
        if self._processed_path.exists():
            self._processed_path.unlink()

    def process_url(
        self,
        url: str,
        on_progress: Optional[Callable[[str], None]] = None,
        force_video: bool = False,
    ) -> Dict[str, Any]:
        """Run the full indexing pipeline for a single video URL.

        Steps:

        1. Return early with ``status="already_indexed"`` if the URL was
           processed before.
        2. Fetch video metadata (title, thumbnail).
        3. Download audio (YouTube) or full video (social platforms /
           local files).
        4. Transcribe with Whisper.
        5. Merge transcript segments into fixed-duration chunks.
        6. Generate embeddings for each chunk via the configured provider.
        7. Store embeddings + metadata in FAISS.
        8. Record the URL as processed and clean up temporary audio files.

        Args:
            url: Video URL or local filesystem path to index.
            on_progress: Optional callback invoked with a human-readable
                status string at each pipeline stage.  Used by the SSE
                endpoint to stream progress to the browser.

        Returns:
            A result dict containing:

            - ``status``: ``"success"``, ``"already_indexed"``, or
              ``"error"``.
            - ``url``: The input URL.
            - ``title``: Video title (empty on error).
            - ``chunks``: Number of chunks indexed (0 on error or skip).
            - ``error``: Error message (only present when
              ``status == "error"``).

        Example:
            >>> result = pipeline.process_url(
            ...     "https://youtu.be/abc",
            ...     on_progress=print,
            ... )
            >>> result["status"]
            'success'
        """
        uid = self._url_id(url)

        if uid in self._processed:
            return {"status": "already_indexed", "url": url, "chunks": 0}

        def log(msg: str) -> None:
            print(f"[Seekr] {msg}")
            if on_progress:
                on_progress(msg)

        local = is_local_file(url)
        clean_path = url.replace("file://", "")

        # ── PDF handling ───────────────────────────────────────────────────────
        if is_pdf_file(clean_path):
            if local:
                pdf_path = clean_path
                self._local_pdfs_dir.mkdir(parents=True, exist_ok=True)
                dest = self._local_pdfs_dir / f"{uid}.pdf"
                shutil.copy2(pdf_path, dest)
                served_url = f"/api/local-pdf/{uid}.pdf"
            else:
                log("[5%] Downloading PDF...")
                self._local_pdfs_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = download_file(url, str(self._local_pdfs_dir))
                if not pdf_path:
                    return {"status": "error", "url": url, "error": "PDF download failed", "chunks": 0}
                served_url = url

            log("[20%] Extracting text from PDF...")
            info = get_pdf_info(pdf_path)
            chunks = extract_pdf_segments(pdf_path)
            if not chunks:
                return {"status": "error", "url": url, "error": "No text could be extracted from PDF", "chunks": 0}

            log(f"[60%] Generating embeddings for {len(chunks)} chunks...")
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_batch(texts)

            metadata = [
                {
                    "text": chunk["text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "video_url": served_url,
                    "video_title": info.get("title", Path(clean_path).stem),
                    "video_thumbnail": "",
                    "source_url": url,
                    "platform": "pdf",
                }
                for chunk in chunks
            ]
            self.vector_store.add(embeddings, metadata)
            self._processed.add(uid)
            self._save_processed()
            log(f"[100%] Done! {len(chunks)} chunks indexed.")
            return {"status": "success", "url": url, "title": info.get("title", ""), "chunks": len(chunks)}

        # ── Direct audio file handling ─────────────────────────────────────────
        if is_audio_file(clean_path) and (local or get_platform(url) == "direct"):
            title = Path(clean_path).stem
            if local:
                self._local_audio_dir.mkdir(parents=True, exist_ok=True)
                src = Path(clean_path)
                dest = self._local_audio_dir / f"{uid}{src.suffix}"
                shutil.copy2(src, dest)
                served_url = f"/api/local-audio/{uid}{src.suffix}"
                audio_path = clean_path
            else:
                log("[5%] Downloading audio file...")
                self._local_audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = download_file(url, str(self._audio_dir))
                if not audio_path:
                    return {"status": "error", "url": url, "error": "Audio download failed", "chunks": 0}
                served_url = url

            log("[15%] Transcribing audio…")
            raw_segments = self._transcribe_with_heartbeat(audio_path, log, 15, 75)

            if not local:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

            if not raw_segments:
                return {"status": "error", "url": url, "error": "Transcription returned no segments", "chunks": 0}

            log("[75%] Splitting transcript into semantic chunks...")
            chunks = merge_segments(raw_segments, self.chunk_duration, self.min_chunk_duration)
            log(f"[80%] Generating embeddings for {len(chunks)} chunks...")
            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_batch(texts)

            metadata = [
                {
                    "text": chunk["text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "video_url": served_url,
                    "video_title": title,
                    "video_thumbnail": "",
                    "source_url": url,
                    "platform": "audio",
                }
                for chunk in chunks
            ]
            self.vector_store.add(embeddings, metadata)
            self._processed.add(uid)
            self._save_processed()
            log(f"[100%] Done! {len(chunks)} chunks indexed.")
            return {"status": "success", "url": url, "title": title, "chunks": len(chunks)}

        # ── Video handling (existing logic) ────────────────────────────────────
        platform = get_platform(url) if not local else "local"

        # 1. Metadata
        log("[5%] Fetching video metadata...")
        info = get_video_info(url, self.cookies_from_browser)

        # 2. Download + transcribe — strategy depends on platform
        served_url = url

        if local:
            # Copy to served directory so the browser can access it
            self._local_videos_dir.mkdir(parents=True, exist_ok=True)
            src = Path(clean_path)
            dest = self._local_videos_dir / f"{uid}{src.suffix}"
            shutil.copy2(src, dest)
            served_url = f"/api/local-video/{uid}{src.suffix}"
            log("[20%] Transcribing local file…")
            raw_segments = self._transcribe_with_heartbeat(str(src), log, 20, 75)

        elif platform in NEEDS_LOCAL_SERVING:
            # LinkedIn / Instagram / TikTok — full video required for
            # playback, and Whisper can transcribe MP4 directly via ffmpeg
            log(f"[10%] Downloading video from {platform} (this may take a while)...")
            self._local_videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = download_video(
                url, str(self._local_videos_dir), uid, self.cookies_from_browser
            )
            if not video_path:
                return {
                    "status": "error",
                    "url": url,
                    "error": (
                        f"Video download failed for {platform}. "
                        "If the content requires login, set "
                        "cookies_from_browser in config.yaml."
                    ),
                    "chunks": 0,
                }
            served_url = f"/api/local-video/{uid}.mp4"
            log("[40%] Transcribing video…")
            raw_segments = self._transcribe_with_heartbeat(video_path, log, 40, 75)

        else:
            if force_video:
                # Download full video for local serving and playback in browser
                log("[10%] Downloading video (this may take a while)...")
                self._local_videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = download_video(
                    url, str(self._local_videos_dir), uid, self.cookies_from_browser
                )
                if not video_path:
                    return {"status": "error", "url": url, "error": "Video download failed", "chunks": 0}
                served_url = f"/api/local-video/{uid}.mp4"
                log("[40%] Transcribing video…")
                raw_segments = self._transcribe_with_heartbeat(video_path, log, 40, 75)
            else:
                # Audio-only download — faster and uses less disk space
                log("[10%] Downloading audio...")
                self._audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = download_audio(url, str(self._audio_dir), self.cookies_from_browser)
                if not audio_path:
                    return {"status": "error", "url": url, "error": "Audio download failed", "chunks": 0}
                log("[35%] Transcribing audio…")
                raw_segments = self._transcribe_with_heartbeat(audio_path, log, 35, 75)
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        if not raw_segments:
            return {
                "status": "error",
                "url": url,
                "error": "Transcription returned no segments",
                "chunks": 0,
            }

        # 3. Chunk
        log("[75%] Splitting transcript into semantic chunks...")
        chunks = merge_segments(raw_segments, self.chunk_duration, self.min_chunk_duration)

        # 4. Embed
        log(f"[80%] Generating embeddings for {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # 5. Store
        metadata = [
            {
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"],
                "video_url": served_url,
                "video_title": info.get("title", ""),
                "video_thumbnail": info.get("thumbnail", ""),
                "source_url": url,
                "platform": platform,
            }
            for chunk in chunks
        ]
        self.vector_store.add(embeddings, metadata)
        self._processed.add(uid)
        self._save_processed()

        log(f"[100%] Done! {len(chunks)} chunks indexed.")
        return {
            "status": "success",
            "url": url,
            "title": info.get("title", ""),
            "chunks": len(chunks),
        }

    def process_directory(
        self,
        directory: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Index all supported video/audio files found under *directory*.

        Walks the directory tree recursively and calls :meth:`process_url`
        for every file whose extension matches a known media format.

        Args:
            directory: Path to the root directory to scan.
            on_progress: Optional progress callback forwarded to each
                :meth:`process_url` call.

        Returns:
            List of result dicts, one per discovered media file.  See
            :meth:`process_url` for the dict structure.

        Example:
            >>> results = pipeline.process_directory("./lectures")
            >>> sum(1 for r in results if r["status"] == "success")
            5
        """
        exts = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".m4a",
                ".ogg", ".flac", ".aac", ".opus", ".pdf"}
        results = []
        for path in Path(directory).rglob("*"):
            if path.suffix.lower() in exts:
                result = self.process_url(str(path.absolute()), on_progress)
                results.append(result)
        return results

    @property
    def indexed_count(self) -> int:
        """Number of unique URLs that have been successfully indexed."""
        return len(self._processed)
