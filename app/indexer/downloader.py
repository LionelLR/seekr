"""Video and audio downloading utilities built on top of yt-dlp.

Supports YouTube, LinkedIn, Instagram, TikTok, and any direct media URL
that yt-dlp can handle.  The key distinction between platforms is whether
they can be *iframe-embedded* in a browser (YouTube can; social platforms
generally cannot), which determines the download strategy used by the
pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

# Platforms that cannot be embedded via iframe — the full video file must
# be downloaded locally and served through Seekr's own endpoint.
NEEDS_LOCAL_SERVING = {"linkedin", "instagram", "tiktok"}


def get_platform(url: str) -> str:
    """Identify the hosting platform from a URL.

    Args:
        url: Any video URL or page URL.

    Returns:
        One of ``"linkedin"``, ``"instagram"``, ``"tiktok"``,
        ``"youtube"``, or ``"direct"`` (fallback for unrecognised URLs).

    Example:
        >>> get_platform("https://www.youtube.com/watch?v=abc")
        'youtube'
        >>> get_platform("https://www.linkedin.com/posts/...")
        'linkedin'
        >>> get_platform("https://example.com/video.mp4")
        'direct'
    """
    u = url.lower()
    if "linkedin.com" in u:
        return "linkedin"
    if "instagram.com" in u:
        return "instagram"
    if "tiktok.com" in u:
        return "tiktok"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    return "direct"


def _base_opts(cookies_from_browser: str = "") -> dict:
    """Build a base yt-dlp options dict, optionally enabling cookie extraction.

    Args:
        cookies_from_browser: Browser name to extract cookies from (e.g.
            ``"firefox"``, ``"chrome"``).  An empty string disables cookie
            extraction.

    Returns:
        A yt-dlp options dict with ``quiet`` and ``no_warnings`` set, and
        ``cookiesfrombrowser`` populated when *cookies_from_browser* is
        non-empty.
    """
    opts: dict = {"quiet": True, "no_warnings": True}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def get_video_info(url: str, cookies_from_browser: str = "") -> Dict:
    """Fetch metadata for a video URL without downloading it.

    Makes a single yt-dlp ``extract_info`` call with ``download=False``
    to retrieve title, duration, thumbnail, and uploader.

    Args:
        url: Any video URL supported by yt-dlp.
        cookies_from_browser: Browser to read session cookies from.
            Required for platforms that enforce login.

    Returns:
        A dict with keys ``title``, ``duration`` (seconds), ``thumbnail``
        (URL or empty string), ``uploader``, ``url``, and ``platform``.
        On error, a partial dict with an ``"error"`` key is returned so
        that the pipeline can continue with a degraded title.

    Example:
        >>> info = get_video_info("https://youtu.be/dQw4w9WgXcQ")
        >>> info["title"]
        'Rick Astley - Never Gonna Give You Up ...'
    """
    try:
        with yt_dlp.YoutubeDL(_base_opts(cookies_from_browser)) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title") or Path(url).stem,
                "duration": info.get("duration", 0),
                "thumbnail": info.get("thumbnail", ""),
                "uploader": info.get("uploader", ""),
                "url": url,
                "platform": get_platform(url),
            }
    except Exception as e:
        return {
            "title": Path(url).stem,
            "duration": 0,
            "thumbnail": "",
            "url": url,
            "platform": get_platform(url),
            "error": str(e),
        }


def download_audio(
    url: str,
    output_dir: str,
    cookies_from_browser: str = "",
) -> Optional[str]:
    """Download the best available audio stream and convert it to MP3.

    Uses yt-dlp's ``FFmpegExtractAudio`` post-processor to produce a
    128 kbps MP3 file.  The output filename is derived from the video ID
    returned by yt-dlp.  Suitable for YouTube and other platforms where
    only audio is needed for transcription (the video is played via an
    embed or external link).

    Args:
        url: Video URL to download audio from.
        output_dir: Directory to write the MP3 file into.  Created if it
            does not exist.
        cookies_from_browser: Browser to read session cookies from.

    Returns:
        Absolute path to the downloaded ``.mp3`` file, or ``None`` if
        the download failed.

    Example:
        >>> path = download_audio("https://youtu.be/abc", "/tmp/audio")
        >>> path.endswith(".mp3")
        True
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outtmpl = os.path.join(output_dir, "%(id)s.%(ext)s")

    opts = {
        **_base_opts(cookies_from_browser),
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info:
            filename = ydl.prepare_filename(info)
            mp3 = os.path.splitext(filename)[0] + ".mp3"
            if os.path.exists(mp3):
                return mp3
    return None


def download_video(
    url: str,
    output_dir: str,
    video_id: str,
    cookies_from_browser: str = "",
) -> Optional[str]:
    """Download a full video file as MP4 for local serving and transcription.

    Used for platforms in :data:`NEEDS_LOCAL_SERVING` (LinkedIn, Instagram,
    TikTok) where iframe embedding is not available.  The file is written
    to ``{output_dir}/{video_id}.mp4``.  Whisper can transcribe MP4 files
    directly via ffmpeg, so no separate audio extraction step is needed.

    Args:
        url: Video URL to download.
        output_dir: Destination directory.  Created if it does not exist.
        video_id: Unique identifier used as the output filename stem.
        cookies_from_browser: Browser to read session cookies from.

    Returns:
        Absolute path to the downloaded ``.mp4`` file, or ``None`` if the
        download failed.

    Example:
        >>> path = download_video("https://tiktok.com/...", "/tmp/vids", "abc123")
        >>> path.endswith(".mp4")
        True
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_id}.mp4")

    opts = {
        **_base_opts(cookies_from_browser),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "nopart": True,  # Write directly to final file; avoids .part rename race
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    if os.path.exists(out_path):
        return out_path
    return None


def extract_page_videos(page_url: str, cookies_from_browser: str = "") -> List[Dict]:
    """Extract individual video URLs from a playlist, channel, or profile page.

    Uses yt-dlp's flat playlist extraction to discover video entries
    without downloading them.  Platform-specific URL reconstruction is
    applied for entries that only carry an ``id`` field.

    Falls back to returning ``[page_url]`` in all error cases so that the
    caller can still attempt to index the original URL directly.

    Args:
        page_url: URL of a YouTube playlist/channel, Instagram profile,
            TikTok user page, or any single video URL (which will be
            returned as-is in a single-element list).
        cookies_from_browser: Browser to read session cookies from.

    Returns:
        List of dicts with keys ``url``, ``title``, ``thumbnail``, ``duration``
        for each discovered video.  Never empty — always contains at least
        *page_url* as a fallback entry.

    Example:
        >>> entries = extract_page_videos("https://youtube.com/playlist?list=PL...")
        >>> len(entries) > 0
        True
    """
    platform = get_platform(page_url)

    opts = {
        **_base_opts(cookies_from_browser),
        "extract_flat": "in_playlist",
    }

    def _yt_thumbnail(video_id: str) -> str:
        """Return a reliable YouTube thumbnail URL from a video ID.

        Uses ``hqdefault.jpg`` which is always available (unlike
        ``maxresdefault.jpg`` which may be missing for older videos).
        """
        return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

    def _make_entry(url: str, entry: Dict, entry_platform: str = platform) -> Dict:
        title = entry.get("title") or ""
        vid_id = entry.get("id") or ""

        # Fall back to video ID then URL when title is absent
        if not title:
            title = vid_id or url

        # Thumbnail: flat-mode often returns None for YouTube.
        # Construct directly from the video ID — always reliable.
        thumbnail = entry.get("thumbnail") or ""
        if not thumbnail:
            thumbs = entry.get("thumbnails") or []
            if thumbs:
                thumbnail = thumbs[-1].get("url", "")
        if not thumbnail and entry_platform == "youtube" and vid_id:
            thumbnail = _yt_thumbnail(vid_id)

        duration = entry.get("duration") or 0
        return {"url": url, "title": title, "thumbnail": thumbnail, "duration": duration}

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(page_url, download=False)
            if not info:
                return [{"url": page_url, "title": page_url, "thumbnail": "", "duration": 0}]

            if "entries" in info:
                entries: List[Dict] = []
                uploader_id = info.get("uploader_id", "")
                for entry in (info.get("entries") or []):
                    if not entry:
                        continue
                    entry_url = entry.get("url") or entry.get("webpage_url") or ""
                    eid = entry.get("id") or ""
                    if not entry_url.startswith("http") and eid:
                        if platform == "youtube":
                            entry_url = f"https://www.youtube.com/watch?v={eid}"
                        elif platform == "instagram":
                            entry_url = f"https://www.instagram.com/p/{eid}/"
                        elif platform == "tiktok":
                            entry_url = f"https://www.tiktok.com/@{uploader_id}/video/{eid}"
                        else:
                            entry_url = page_url
                    if entry_url.startswith("http"):
                        entries.append(_make_entry(entry_url, entry))
                return entries or [{"url": page_url, "title": page_url, "thumbnail": "", "duration": 0}]

            # Single video — use top-level info (full metadata available here)
            return [_make_entry(page_url, info)]
    except Exception:
        return [{"url": page_url, "title": page_url, "thumbnail": "", "duration": 0}]


def is_local_file(url: str) -> bool:
    """Return ``True`` if *url* refers to a local filesystem path.

    Recognises paths starting with ``/``, ``./``, or the ``file://``
    scheme.

    Args:
        url: URL or filesystem path to test.

    Returns:
        ``True`` for local paths, ``False`` for remote URLs.

    Example:
        >>> is_local_file("/home/user/video.mp4")
        True
        >>> is_local_file("https://youtube.com/watch?v=abc")
        False
    """
    return url.startswith("/") or url.startswith("./") or url.startswith("file://")


# Audio file extensions handled by Whisper transcription directly
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma"}


def is_audio_file(url: str) -> bool:
    """Return ``True`` if *url* refers to a direct audio file.

    Checks the file extension after stripping any query string.

    Args:
        url: URL or filesystem path.

    Returns:
        ``True`` for recognised audio extensions.

    Example:
        >>> is_audio_file("/path/to/podcast.mp3")
        True
        >>> is_audio_file("https://example.com/talk.mp3?dl=1")
        True
    """
    return Path(url.split("?")[0]).suffix.lower() in AUDIO_EXTENSIONS


def is_pdf_file(url: str) -> bool:
    """Return ``True`` if *url* refers to a PDF file.

    Args:
        url: URL or filesystem path.

    Returns:
        ``True`` when the path ends in ``.pdf`` (case-insensitive).

    Example:
        >>> is_pdf_file("/docs/report.pdf")
        True
    """
    return Path(url.split("?")[0]).suffix.lower() == ".pdf"


def download_file(url: str, output_dir: str) -> Optional[str]:
    """Download a remote file to *output_dir* using urllib.

    Suitable for direct audio or PDF URLs that yt-dlp cannot handle.
    The filename is derived from the URL path.

    Args:
        url: Direct URL to the file.
        output_dir: Directory to write the file into.  Created if needed.

    Returns:
        Absolute path to the downloaded file, or ``None`` on failure.

    Example:
        >>> path = download_file("https://example.com/talk.mp3", "/tmp")
        >>> path.endswith(".mp3")
        True
    """
    import urllib.request

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?")[0]).name or "download"
    out_path = os.path.join(output_dir, filename)
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path if os.path.exists(out_path) else None
    except Exception:
        return None
