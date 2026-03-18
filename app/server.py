"""FastAPI application factory.

:func:`create_app` wires together all services (config, embedder, vector
store, search engine, transcriber, pipeline), injects them into the route
module, and returns a configured :class:`~fastapi.FastAPI` instance.

A module-level ``app`` instance is also exported so that
``uvicorn app.server:app`` works without arguments.  The config path is
read from the ``SEEKR_CONFIG`` environment variable (defaults to
``config.yaml``).
"""

import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import routes as _routes
from .core.chat import get_chat_provider
from .core.config import load_config
from .core.embeddings import get_embedding_provider
from .core.search import SearchEngine
from .core.vector_store import VectorStore
from .indexer.pipeline import IndexingPipeline
from .indexer.transcriber import Transcriber


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure the Seekr FastAPI application.

    Loads the configuration, constructs all services, injects them into
    the ``app.api.routes`` module, registers middleware (CORS), mounts
    the API router, and adds routes for the static HTML UI.

    Args:
        config_path: Path to the YAML configuration file.  Passed
            directly to :func:`~app.core.config.load_config`.

    Returns:
        A fully configured :class:`~fastapi.FastAPI` instance ready to be
        served with Uvicorn.

    Example:
        >>> from app.server import create_app
        >>> application = create_app("config.yaml")
        >>> application.title
        'Seekr'
    """
    cfg = load_config(config_path)

    # Construct all services
    embedder = get_embedding_provider(cfg)
    vector_store = VectorStore(cfg.data_dir)
    search_engine = SearchEngine(embedder, vector_store, cfg.top_k)
    transcriber = Transcriber(cfg.whisper_model, initial_prompt=", ".join(cfg.vocabulary))
    pipeline = IndexingPipeline(
        embedder, vector_store, transcriber,
        cfg.chunk_duration, cfg.min_chunk_duration, cfg.data_dir, cfg.cookies_from_browser,
    )

    # Pre-warm Whisper in background so the first indexing request is not delayed
    threading.Thread(target=lambda: transcriber.model, daemon=True).start()

    # Inject into the routes module (avoids FastAPI DI boilerplate)
    _routes._cfg = cfg
    _routes._search_engine = search_engine
    _routes._pipeline = pipeline
    _routes._chat_provider = get_chat_provider(cfg)
    _routes._admin_password = cfg.admin_password
    if cfg.admin_password:
        _routes._admin_token = _routes._make_token(cfg.admin_password)

    app = FastAPI(title="Seekr", description="Search inside videos", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(_routes.router, prefix="/api")

    ui_dir = Path(__file__).parent / "ui"

    @app.get("/")
    def index() -> FileResponse:
        """Serve the main search UI (``index.html``)."""
        return FileResponse(str(ui_dir / "index.html"))

    @app.get("/admin")
    def admin() -> FileResponse:
        """Serve the admin UI (``admin.html``)."""
        return FileResponse(str(ui_dir / "admin.html"))

    return app


# Default instance for `uvicorn app.server:app`
_config_path = os.environ.get("SEEKR_CONFIG", "config.yaml")
app = create_app(_config_path)
