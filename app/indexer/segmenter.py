"""Segment merging utilities for creating indexable text chunks.

Whisper produces fine-grained segments that can be as short as a single
word.  This module groups them into semantic chunks that respect natural
sentence and Q&A boundaries, so that a question and its answer always end
up in the same chunk and searches land on the right moment.

Splitting rules (applied in priority order):
1. **Never cut mid-sentence** — a boundary is only considered when the last
   accumulated segment ends with ``.``, ``?``, or ``!``.
2. **Q&A grouping** — when a segment ends with ``?`` (question detected),
   any pending content that has reached *min_chunk_duration* is finalised
   *before* the question, so the question opens a fresh chunk and its full
   answer is captured alongside it.
3. **Max-duration cut** — at any sentence boundary, if the accumulated
   duration has reached *chunk_duration*, the chunk is finalised.
4. **Hard cap** — if a chunk grows beyond ``2 × chunk_duration`` with no
   sentence boundary in sight (e.g. continuous speech), it is force-split
   at that point regardless.
"""

from typing import Any, Dict, List


def _ends_sentence(text: str) -> bool:
    """Return True if *text* ends with a sentence-terminating character."""
    return text.rstrip().endswith(('.', '?', '!'))


def _is_question(text: str) -> bool:
    """Return True if *text* ends with a question mark."""
    return text.rstrip().endswith('?')


def _finalise(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of Whisper segments into a single chunk dict."""
    return {
        "text": " ".join(s["text"] for s in segments).strip(),
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
    }


def merge_segments(
    whisper_segments: List[Dict[str, Any]],
    chunk_duration: float = 120.0,
    min_chunk_duration: float = 10.0,
) -> List[Dict[str, Any]]:
    """Merge Whisper segments into semantic Q&A-aware chunks.

    A question and its answer are kept in the same chunk so that clicking
    a search result starts playback at the question and covers the full
    response.

    Args:
        whisper_segments: List of segment dicts as returned by
            :meth:`~app.indexer.transcriber.Transcriber.transcribe`.
            Each dict must contain ``"text"`` (str), ``"start"`` (float),
            and ``"end"`` (float) keys.
        chunk_duration: Maximum target chunk length in seconds.  A chunk
            is finalised at the next sentence boundary once this duration
            is reached.  Defaults to 120 seconds.
        min_chunk_duration: Minimum duration (seconds) required before a
            chunk can be split at a question boundary.  Prevents spurious
            one-sentence chunks for rapid-fire questions.  Defaults to
            10 seconds.

    Returns:
        List of chunk dicts with ``"text"``, ``"start"``, and ``"end"``
        keys.  Returns an empty list when *whisper_segments* is empty.

    Example:
        >>> segs = [
        ...     {"text": "Welcome.", "start": 0.0, "end": 5.0},
        ...     {"text": "How does X work?", "start": 5.0, "end": 10.0},
        ...     {"text": "X works like this.", "start": 10.0, "end": 20.0},
        ...     {"text": "Got it.", "start": 20.0, "end": 22.0},
        ... ]
        >>> chunks = merge_segments(segs, chunk_duration=60.0, min_chunk_duration=4.0)
        >>> chunks[0]["text"]
        'Welcome.'
        >>> chunks[1]["text"]
        'How does X work? X works like this. Got it.'
        >>> chunks[1]["start"]
        5.0
    """
    if not whisper_segments:
        return []

    chunks: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []

    for seg in whisper_segments:
        text = seg["text"]

        # ── Rule 2: Q&A grouping ────────────────────────────────────────────
        # If this segment IS a question and we already have pending content
        # that has reached the minimum duration, finalise the pending chunk
        # first.  The question then opens a fresh chunk so that the answer
        # is captured alongside it.
        if _is_question(text) and pending:
            elapsed = pending[-1]["end"] - pending[0]["start"]
            if elapsed >= min_chunk_duration:
                chunks.append(_finalise(pending))
                pending = []

        pending.append(seg)
        elapsed = pending[-1]["end"] - pending[0]["start"]

        # ── Rule 4: Hard cap ────────────────────────────────────────────────
        if elapsed >= chunk_duration * 2:
            chunks.append(_finalise(pending))
            pending = []
            continue

        # ── Rule 3: Max-duration cut at sentence boundary ──────────────────
        if elapsed >= chunk_duration and _ends_sentence(text):
            chunks.append(_finalise(pending))
            pending = []

    # Emit any remaining segments as the final chunk
    if pending:
        chunks.append(_finalise(pending))

    return chunks
