"""FAISS-backed vector store with JSON metadata sidecar.

The store uses an ``IndexFlatIP`` (inner product) index over L2-normalised
vectors, which is equivalent to cosine similarity search.  Both the index
and the metadata are persisted to disk so that restarts do not require
re-indexing.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


class VectorStore:
    """Persistent vector store combining FAISS index and JSON metadata.

    Each call to :meth:`add` appends new embedding vectors to the FAISS
    index **and** the corresponding metadata dicts to an in-memory list
    that is written to ``metadata.json``.  Vectors are L2-normalised
    before insertion so that inner-product search equals cosine similarity.

    The store is loaded from disk automatically on construction.  If no
    on-disk data exist the store starts empty and a new index is created
    on the first :meth:`add` call (the dimensionality is inferred from
    the first batch of vectors).

    Args:
        data_dir: Directory where ``faiss.index`` and ``metadata.json``
            are stored.  Created if it does not exist.

    Example:
        >>> vs = VectorStore("./data")
        >>> vs.add([[0.1, 0.2, 0.3]], [{"text": "hello", "video_url": "u"}])
        >>> score, meta = vs.search([0.1, 0.2, 0.3], top_k=1)[0]
        >>> meta["text"]
        'hello'
    """

    def __init__(self, data_dir: str = "./data") -> None:
        """Initialise the vector store and load persisted data if present.

        Args:
            data_dir: Path to the directory used for index and metadata
                persistence.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.data_dir / "faiss.index"
        self._meta_path = self.data_dir / "metadata.json"
        self._lock = threading.Lock()
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load a persisted index and metadata from disk.

        If either file is missing this method does nothing (the store
        starts empty and new files are written on the first :meth:`add`).
        """
        if self._index_path.exists() and self._meta_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            with open(self._meta_path) as f:
                self.metadata = json.load(f)

    def _save(self) -> None:
        """Persist the current index and metadata to disk.

        Called automatically after every :meth:`add` and
        :meth:`clear_by_url` call.
        """
        if self.index is not None:
            faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Add embedding vectors and their associated metadata.

        Vectors are L2-normalised before being inserted so that the
        inner-product index behaves as cosine similarity search.  The
        index dimensionality is set from the first batch ever added; all
        subsequent batches must have the same dimension.

        Args:
            embeddings: List of dense float vectors.  All must have the
                same length.
            metadata: List of metadata dicts, one per vector.  Must have
                the same length as *embeddings*.  Each dict should contain
                at minimum ``text``, ``start``, ``end``, and ``video_url``.

        Raises:
            ValueError: (from FAISS) if vector dimensions are inconsistent.
        """
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        with self._lock:
            if self.index is None:
                dim = vectors.shape[1]
                self.index = faiss.IndexFlatIP(dim)
            self.index.add(vectors)
            self.metadata.extend(metadata)
            self._save()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Tuple[float, Dict]]:
        """Find the *top_k* most similar segments to *query_embedding*.

        Returns an empty list when the index contains no vectors.  The
        number of results is capped at the total number of stored vectors
        to prevent FAISS from returning out-of-range indices.

        Args:
            query_embedding: Dense query vector.  Must have the same
                dimension as the stored vectors.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(score, metadata)`` tuples sorted by descending
            cosine similarity.  Scores are in the range ``[0, 1]`` for
            unit-normalised vectors.

        Example:
            >>> results = vs.search(query_vec, top_k=3)
            >>> score, meta = results[0]
            >>> meta["video_url"]
            'https://...'
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            query = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query)
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query, k)
            return [
                (float(scores[0][i]), self.metadata[indices[0][i]])
                for i in range(k)
                if indices[0][i] >= 0 and indices[0][i] < len(self.metadata)
            ]

    def get_by_url(self, video_url: str) -> List[Tuple[int, Dict]]:
        """Return (index, metadata) pairs for all segments of video_url."""
        with self._lock:
            return [(i, m) for i, m in enumerate(self.metadata) if m.get("video_url") == video_url]

    def delete_by_url(self, video_url: str) -> int:
        """Remove all segments for video_url, rebuilding the FAISS index properly."""
        with self._lock:
            keep = [i for i, m in enumerate(self.metadata) if m.get("video_url") != video_url]
            removed = len(self.metadata) - len(keep)
            if removed == 0:
                return 0
            self._rebuild_unsafe(keep)
            return removed

    def update_segment(
        self,
        idx: int,
        text: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        new_embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update text/timestamps and optionally replace the embedding vector."""
        with self._lock:
            if not (0 <= idx < len(self.metadata)):
                return False
            if text is not None:
                self.metadata[idx]["text"] = text
            if start is not None:
                self.metadata[idx]["start"] = start
            if end is not None:
                self.metadata[idx]["end"] = end
            if new_embedding is not None and self.index is not None:
                dim = self.index.d
                vecs = np.zeros((len(self.metadata), dim), dtype=np.float32)
                for i in range(len(self.metadata)):
                    if i == idx:
                        vecs[i] = np.array(new_embedding, dtype=np.float32)
                    else:
                        self.index.reconstruct(i, vecs[i])
                faiss.normalize_L2(vecs)
                new_index = faiss.IndexFlatIP(dim)
                new_index.add(vecs)
                self.index = new_index
            self._save()
            return True

    def delete_segment(self, idx: int) -> bool:
        """Delete a single segment, rebuilding the FAISS index."""
        with self._lock:
            if not (0 <= idx < len(self.metadata)):
                return False
            self._rebuild_unsafe([i for i in range(len(self.metadata)) if i != idx])
            return True

    def _rebuild(self, keep_indices: List[int]) -> None:
        """Rebuild the FAISS index (acquires lock)."""
        with self._lock:
            self._rebuild_unsafe(keep_indices)

    def _rebuild_unsafe(self, keep_indices: List[int]) -> None:
        """Rebuild the FAISS index without acquiring the lock (caller must hold it)."""
        new_meta = [self.metadata[i] for i in keep_indices]
        if self.index is not None and keep_indices:
            dim = self.index.d
            vecs = np.zeros((len(keep_indices), dim), dtype=np.float32)
            for new_i, old_i in enumerate(keep_indices):
                self.index.reconstruct(old_i, vecs[new_i])
            new_index = faiss.IndexFlatIP(dim)
            new_index.add(vecs)
            self.index = new_index
        else:
            self.index = None
        self.metadata = new_meta
        self._save()

    def reset(self) -> None:
        """Delete all vectors and metadata, removing persisted files."""
        with self._lock:
            self.index = None
            self.metadata = []
            if self._index_path.exists():
                self._index_path.unlink()
            if self._meta_path.exists():
                self._meta_path.unlink()

    def clear_by_url(self, video_url: str) -> int:
        """Remove all metadata records belonging to *video_url*.

        Only the metadata sidecar is modified; the FAISS index vectors
        themselves are **not** removed (FAISS ``IndexFlatIP`` does not
        support selective deletion without a full rebuild).  As a result
        the removed segments may still appear in search results via
        out-of-bounds metadata accesses, which is why this method is
        primarily useful before a full re-index.

        Args:
            video_url: The ``video_url`` value to filter out.

        Returns:
            The number of metadata records that were removed.
        """
        with self._lock:
            kept_meta = [m for m in self.metadata if m.get("video_url") != video_url]
            removed = len(self.metadata) - len(kept_meta)
            if removed == 0:
                return 0
            self.metadata = kept_meta
            self._save()
            return removed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_segments(self) -> int:
        """Total number of metadata records (indexed segments)."""
        with self._lock:
            return len(self.metadata)

    @property
    def unique_videos(self) -> List[str]:
        """Deduplicated list of ``video_url`` values in insertion order."""
        with self._lock:
            seen: set = set()
            videos: List[str] = []
            for m in self.metadata:
                url = m.get("video_url", "")
                if url not in seen:
                    seen.add(url)
                    videos.append(url)
            return videos

    @property
    def video_list(self) -> List[Dict[str, Any]]:
        """Return one metadata dict per unique video with a segment_count field."""
        with self._lock:
            seen: dict = {}
            for m in self.metadata:
                url = m.get("video_url", "")
                if url not in seen:
                    seen[url] = {
                        "url": url,
                        "title": m.get("video_title", ""),
                        "thumbnail": m.get("video_thumbnail", ""),
                        "segment_count": 0,
                    }
                seen[url]["segment_count"] += 1
            return list(seen.values())
