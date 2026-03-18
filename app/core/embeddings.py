"""Pluggable embedding providers.

All providers implement :class:`EmbeddingProvider`.  New providers can be
added by subclassing it and registering a branch in
:func:`get_embedding_provider`.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding backends.

    Subclasses must implement :meth:`embed`, :meth:`embed_batch`, and the
    :attr:`dimension` property.  The vector store normalises all vectors
    before storage, so embeddings do **not** need to be unit-normalised
    by the provider.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single piece of text.

        Args:
            text: The input string to embed.

        Returns:
            A list of floats representing the dense embedding vector.
            The length equals :attr:`dimension`.

        Example:
            >>> vec = provider.embed("hello world")
            >>> len(vec) == provider.dimension
            True
        """

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single request where possible.

        The order of returned vectors matches the order of *texts*.

        Args:
            texts: List of input strings.

        Returns:
            A list of embedding vectors, one per input text.

        Example:
            >>> vecs = provider.embed_batch(["foo", "bar"])
            >>> len(vecs)
            2
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API.

    Uses ``text-embedding-3-small`` (1 536 dimensions) by default.
    Requests are batched — the API accepts up to 2 048 inputs per call.

    Args:
        api_key: OpenAI API key.
        model: OpenAI embedding model identifier.

    Example:
        >>> provider = OpenAIEmbedding(api_key="sk-...")
        >>> vec = provider.embed("neural networks")
        >>> len(vec)
        1536
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        """Initialise the OpenAI client.

        Args:
            api_key: OpenAI secret key (``sk-...``).
            model: Model identifier, e.g. ``"text-embedding-3-small"`` or
                ``"text-embedding-ada-002"``.
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536

    def embed(self, text: str) -> List[float]:
        """Embed a single string using the OpenAI API.

        Args:
            text: Input text.

        Returns:
            Dense float vector of length 1 536.
        """
        resp = self.client.embeddings.create(input=[text], model=self.model)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings in one API request.

        Results are sorted by the index returned by the API to guarantee
        order consistency.

        Args:
            texts: List of input strings (max 2 048 per call).

        Returns:
            List of dense float vectors, one per input text.
        """
        resp = self.client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality (1 536)."""
        return self._dimension


class GeminiEmbedding(EmbeddingProvider):
    """Embedding provider backed by the Google Gemini Embeddings API.

    Uses ``text-embedding-004`` (768 dimensions) by default.
    Requests are sent one at a time because the Gemini SDK does not
    expose a native batching endpoint.

    Args:
        api_key: Google AI API key.
        model: Gemini embedding model identifier.

    Example:
        >>> provider = GeminiEmbedding(api_key="AIza...")
        >>> vec = provider.embed("deep learning")
        >>> len(vec)
        768
    """

    def __init__(self, api_key: str, model: str = "models/text-embedding-004") -> None:
        """Initialise the Gemini client.

        Args:
            api_key: Google AI API key starting with ``AIza``.
            model: Model resource name, e.g. ``"models/text-embedding-004"``.
        """
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model
        self._dimension = 768

    def embed(self, text: str) -> List[float]:
        """Embed a single string using the Gemini API.

        Args:
            text: Input text.

        Returns:
            Dense float vector of length 768.
        """
        result = self.genai.embed_content(model=self.model, content=text)
        return result["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings in a single API call.

        The Gemini ``embed_content`` endpoint accepts a list of strings and
        returns one embedding per input, avoiding the O(n) sequential call
        overhead.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors in input order.
        """
        if not texts:
            return []
        result = self.genai.embed_content(model=self.model, content=texts)
        # When content is a list, the SDK returns {"embedding": [[...], [...]]}
        return result["embedding"]

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality (768)."""
        return self._dimension


class LocalEmbedding(EmbeddingProvider):
    """Embedding provider backed by a local sentence-transformers model.

    Runs entirely on your machine — no API key or internet connection
    required after the model is downloaded on first use.

    The default model ``all-MiniLM-L6-v2`` is 80 MB, fast on CPU, and
    produces 384-dimensional vectors.

    Args:
        model: Any sentence-transformers model name or local path.

    Example:
        >>> provider = LocalEmbedding()
        >>> vec = provider.embed("hello world")
        >>> len(vec)
        384
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(config) -> EmbeddingProvider:
    """Instantiate the correct :class:`EmbeddingProvider` from *config*.

    Reads ``config.embedding_provider`` to choose the backend and
    validates that the required API key is present.

    Args:
        config: A :class:`~app.core.config.Config` instance.

    Returns:
        A ready-to-use :class:`EmbeddingProvider`.

    Raises:
        ValueError: If the API key for the chosen provider is missing, or
            if ``embedding_provider`` names an unknown backend.

    Example:
        >>> from app.core.config import Config
        >>> cfg = Config(embedding_provider="openai", openai_api_key="sk-x")
        >>> provider = get_embedding_provider(cfg)
        >>> isinstance(provider, OpenAIEmbedding)
        True
    """
    if config.embedding_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        return OpenAIEmbedding(api_key=config.openai_api_key)
    elif config.embedding_provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        return GeminiEmbedding(api_key=config.gemini_api_key)
    elif config.embedding_provider == "local":
        model = getattr(config, "local_embedding_model", "all-MiniLM-L6-v2")
        return LocalEmbedding(model=model)
    else:
        raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")
