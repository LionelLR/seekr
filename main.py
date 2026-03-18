#!/usr/bin/env python3
"""Seekr command-line interface.

Provides three subcommands:

- ``seekr serve`` — start the FastAPI web server
- ``seekr index <dir>`` — batch-index local video files
- ``seekr add-source <url>`` — index a remote video or playlist

All subcommands accept a ``--config`` option pointing to a custom
``config.yaml`` file (defaults to ``./config.yaml``).
"""
import os
import sys
from pathlib import Path

import typer

cli = typer.Typer(
    name="seekr",
    help="Seekr: semantic search inside videos.",
    add_completion=False,
)


def _build_pipeline(config_path: str):
    from app.core.config import load_config
    from app.core.embeddings import get_embedding_provider
    from app.core.vector_store import VectorStore
    from app.indexer.pipeline import IndexingPipeline
    from app.indexer.transcriber import Transcriber

    cfg = load_config(config_path)
    embedder = get_embedding_provider(cfg)
    vs = VectorStore(cfg.data_dir)
    tr = Transcriber(cfg.whisper_model)
    pipeline = IndexingPipeline(embedder, vs, tr, cfg.chunk_duration, cfg.data_dir, cfg.cookies_from_browser)
    return pipeline


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev mode)"),
):
    """Start the Seekr web server."""
    import uvicorn

    os.environ["SEEKR_CONFIG"] = config
    typer.echo(f"Starting Seekr on http://{host}:{port}")
    uvicorn.run("app.server:app", host=host, port=port, reload=reload)


@cli.command()
def index(
    directory: str = typer.Argument(..., help="Directory containing video files"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
):
    """Index all video files in a local directory."""
    pipeline = _build_pipeline(config)
    results = pipeline.process_directory(directory, on_progress=lambda m: typer.echo(f"  {m}"))
    typer.echo("\nSummary:")
    for r in results:
        icon = "OK" if r["status"] == "success" else ("SKIP" if r["status"] == "already_indexed" else "FAIL")
        label = r.get("title") or r.get("url", "")
        extra = f" ({r.get('chunks', 0)} chunks)" if r["status"] == "success" else ""
        typer.echo(f"  [{icon}] {label}{extra}")


@cli.command(name="add-source")
def add_source(
    url: str = typer.Argument(..., help="Video URL or playlist/channel page"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    extract_only: bool = typer.Option(False, "--extract-only", help="Only list URLs, don't index"),
):
    """Add and index a video source (YouTube, direct MP4, etc.)."""
    from app.core.config import load_config
    from app.indexer.downloader import extract_page_videos

    cfg = load_config(config)
    typer.echo(f"Extracting video URLs from: {url}")
    urls = extract_page_videos(url, cfg.cookies_from_browser)
    typer.echo(f"Found {len(urls)} video(s):")
    for u in urls:
        typer.echo(f"  - {u}")

    if extract_only:
        return

    pipeline = _build_pipeline(config)  # loads config again internally — acceptable for CLI
    typer.echo()
    for u in urls:
        typer.echo(f"Processing: {u}")
        result = pipeline.process_url(u, on_progress=lambda m: typer.echo(f"  {m}"))
        icon = "OK" if result["status"] == "success" else ("SKIP" if result["status"] == "already_indexed" else "FAIL")
        chunks = f" ({result.get('chunks', 0)} chunks)" if result["status"] == "success" else ""
        typer.echo(f"  [{icon}] {result.get('title') or u}{chunks}\n")


if __name__ == "__main__":
    cli()
