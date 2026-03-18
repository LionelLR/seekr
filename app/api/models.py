"""Pydantic request and response models for the Seekr API.

All models use Pydantic v2 validation.  Field defaults are chosen to
match the values in ``config.yaml`` so that a minimal JSON payload works
out of the box.
"""

from typing import List, Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    """Payload for ``POST /api/search``.

    Attributes:
        query: Natural-language question or phrase to search for.
        top_k: Maximum number of results to return.  Defaults to 3.

    Example:
        >>> req = SearchRequest(query="what is gradient descent?")
        >>> req.top_k
        3
    """

    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    """A single matching video segment returned by ``POST /api/search``.

    Attributes:
        video_url: Playable URL for the source video.  For YouTube this
            is the original watch URL; for social platforms it is the
            ``/api/local-video/`` served path.
        video_title: Human-readable title fetched from the platform at
            index time.
        video_thumbnail: Thumbnail image URL (may be empty).
        text: Transcript snippet for this segment.
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        score: Cosine similarity score in the range ``[0, 1]``.  Higher
            is more relevant.
    """

    video_url: str
    source_url: str = ""
    platform: str = ""
    video_title: str
    video_thumbnail: str
    text: str
    start: float
    end: float
    score: float


class SearchResponse(BaseModel):
    """Response envelope for ``POST /api/search``.

    Attributes:
        query: The original query string echoed back.
        results: Ordered list of matching segments (best match first).
    """

    query: str
    results: List[SearchResult]


class AddSourceRequest(BaseModel):
    """Payload shared by the ``/admin/extract`` and ``/admin/index-url`` endpoints.

    Attributes:
        url: Any URL supported by yt-dlp — single video, playlist,
            channel, or social media profile/post page.
        download_video: When True, download the full video file (MP4) instead
            of audio-only. The video is served locally for playback.
    """

    url: str
    download_video: bool = False


class ExtractedVideo(BaseModel):
    url: str
    title: str
    thumbnail: str
    duration: float


class ExtractResponse(BaseModel):
    """Response from ``POST /api/admin/extract``.

    Attributes:
        videos: Rich list of discovered videos with title, thumbnail, duration.
        urls: Plain URL list (kept for backwards compatibility).
        message: Human-readable summary, e.g. ``"Found 3 video(s)"``.
    """

    videos: List[ExtractedVideo]
    urls: List[str]  # kept for backwards compatibility
    message: str


class VideoInfo(BaseModel):
    url: str
    title: str
    thumbnail: str
    segment_count: int


class IndexStatusResponse(BaseModel):
    """Response from ``GET /api/status``.

    Attributes:
        total_segments: Total number of indexed chunks across all videos.
        indexed_videos: Number of distinct video URLs that have been
            processed (sourced from ``data/processed.json``).
        videos: Deduplicated list of ``video_url`` values stored in the
            FAISS metadata (in insertion order).
        video_list: One entry per unique video with title, thumbnail, and
            segment count.
    """

    total_segments: int
    indexed_videos: int
    videos: List[str]
    video_list: List[VideoInfo] = []
    ai_enabled: bool = False
    chat_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""


class DeleteVideoRequest(BaseModel):
    video_url: str


class SegmentItem(BaseModel):
    idx: int
    text: str
    start: float
    end: float


class SegmentListResponse(BaseModel):
    video_url: str
    segments: List[SegmentItem]


class SegmentUpdateRequest(BaseModel):
    text: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    re_embed: bool = False


class LoginRequest(BaseModel):
    password: str
