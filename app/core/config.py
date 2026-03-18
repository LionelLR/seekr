"""Application configuration management.

Loads settings from a YAML file and environment variables, exposing them
through a single :class:`Config` dataclass that is passed to every service.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    """Central configuration for Seekr.

    Values are loaded from ``config.yaml`` first, then overridden by
    environment variables where applicable.  All paths are relative to
    the working directory unless absolute paths are provided.

    Attributes:
        embedding_provider: Which embedding backend to use. One of
            ``"openai"``, ``"gemini"``, or ``"local"``.
        whisper_model: OpenAI Whisper model variant.  Larger models are
            more accurate but slower.  One of ``tiny``, ``base``,
            ``small``, ``medium``, ``large``.
        chunk_duration: Maximum length (seconds) of an indexed chunk.
            Chunks are split at sentence boundaries once this threshold
            is reached.  Questions and their answers are kept together,
            so real chunks may be shorter than this value.
        min_chunk_duration: Minimum seconds of content required before a
            chunk can be split at a question boundary.
        top_k: Default number of results returned by a search query.
        data_dir: Root directory for the FAISS index, metadata JSON,
            temporary audio files, and locally served videos.
        video_sources: Optional list of video URLs to index automatically
            on first server start.
        cookies_from_browser: Browser name from which yt-dlp extracts
            session cookies.  Required for LinkedIn, Instagram, and some
            TikTok content.  Leave empty to disable cookie extraction.
        openai_api_key: OpenAI API key.  Read from the ``OPENAI_API_KEY``
            environment variable if not set in the YAML file.
        gemini_api_key: Google Gemini API key.  Read from
            ``GEMINI_API_KEY`` if not set in the YAML file.
        admin_password: If non-empty, the admin panel requires this password.

    Example:
        >>> cfg = Config(embedding_provider="gemini", top_k=5)
        >>> cfg.top_k
        5
    """

    embedding_provider: str = "local"
    whisper_model: str = "base"
    chunk_duration: float = 120.0
    min_chunk_duration: float = 10.0
    top_k: int = 3
    data_dir: str = "./data"
    video_sources: List[str] = field(default_factory=list)
    # Browser to extract cookies from for authenticated platforms
    cookies_from_browser: str = ""
    local_embedding_model: str = "all-MiniLM-L6-v2"
    admin_password: str = ""
    # LLM model for RAG chat answers (e.g. "gpt-4o-mini", "gemini-2.0-flash").
    # Leave empty to disable AI answers and use plain vector search only.
    chat_model: str = ""
    # Custom vocabulary hints for Whisper transcription (proper names, brands, …).
    # These are passed as initial_prompt so Whisper biases toward these tokens.
    vocabulary: List[str] = field(default_factory=list)
    # Filled from env vars
    openai_api_key: str = ""
    gemini_api_key: str = ""


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file and environment variables.

    YAML keys map directly to :class:`Config` field names.  Unknown keys
    are silently ignored.  After the file is parsed, ``OPENAI_API_KEY``
    and ``GEMINI_API_KEY`` environment variables are applied on top,
    overriding any values set in the file.

    The :attr:`Config.data_dir` directory is created if it does not
    already exist.

    Args:
        path: Path to the YAML configuration file.  If the file does not
            exist the function returns a :class:`Config` with default
            values (plus any environment-variable overrides).

    Returns:
        A fully populated :class:`Config` instance.

    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-test"
        >>> cfg = load_config("/nonexistent/config.yaml")
        >>> cfg.openai_api_key
        'sk-test'
    """
    cfg = Config()
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    cfg.openai_api_key = os.getenv("OPENAI_API_KEY", cfg.openai_api_key)
    cfg.gemini_api_key = os.getenv("GEMINI_API_KEY", cfg.gemini_api_key)
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    return cfg
