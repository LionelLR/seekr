"""High-level search engine combining embedding and vector retrieval."""

from typing import Any, Dict, List, Optional

from .embeddings import EmbeddingProvider
from .vector_store import VectorStore


class SearchEngine:
    """Orchestrates query embedding and nearest-neighbour retrieval.

    The search engine converts a natural-language query into a dense
    vector, queries the :class:`VectorStore`, and returns the top
    matching segment metadata annotated with a cosine-similarity score.

    Args:
        embedder: Provider used to embed search queries at runtime.
        vector_store: FAISS-backed store that holds indexed segments.
        top_k: Default number of results to return when *top_k* is not
            specified in the :meth:`search` call.

    Example:
        >>> engine = SearchEngine(embedder, vector_store, top_k=3)
        >>> results = engine.search("what is gradient descent?")
        >>> results[0]["score"]  # cosine similarity in [0, 1]
        0.8732
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 3,
    ) -> None:
        """Initialise the search engine.

        Args:
            embedder: Embedding provider for encoding queries.
            vector_store: Populated :class:`VectorStore` to search.
            top_k: Fallback result count when not overridden per query.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for the most relevant video segments matching *query*.

        The query is embedded with the configured :class:`EmbeddingProvider`
        and compared against all stored segment vectors using cosine
        similarity (inner product on L2-normalised vectors).

        Args:
            query: Natural-language question or phrase to search for.
            top_k: Number of results to return.  Falls back to the
                instance default when *None*.

        Returns:
            List of segment metadata dicts, sorted by descending
            similarity.  Each dict contains at minimum:

            - ``text`` — transcript snippet
            - ``start`` — segment start time in seconds
            - ``end`` — segment end time in seconds
            - ``video_url`` — playable URL for the source video
            - ``video_title`` — human-readable video title
            - ``score`` — cosine similarity rounded to four decimal places

        Example:
            >>> results = engine.search("backpropagation", top_k=2)
            >>> results[0].keys()
            dict_keys(['text', 'start', 'end', 'video_url', 'video_title', ..., 'score'])
        """
        k = top_k if top_k is not None else self.top_k
        embedding = self.embedder.embed(query)
        raw = self.vector_store.search(embedding, k)
        return [{**meta, "score": round(score, 4)} for score, meta in raw]
