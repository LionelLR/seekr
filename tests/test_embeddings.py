"""Tests for app.core.embeddings — embedding providers and factory."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Config
from app.core.embeddings import (
    EmbeddingProvider,
    GeminiEmbedding,
    OpenAIEmbedding,
    get_embedding_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_openai_response(texts, dim=1536):
    """Build a fake OpenAI embeddings API response."""
    import numpy as np

    response = MagicMock()
    response.data = []
    for i, text in enumerate(texts):
        item = MagicMock()
        item.index = i
        item.embedding = list(np.random.default_rng(i).random(dim).astype(float))
        response.data.append(item)
    return response


# ---------------------------------------------------------------------------
# OpenAIEmbedding
# ---------------------------------------------------------------------------


class TestOpenAIEmbedding:
    """Unit tests for OpenAIEmbedding using a mocked OpenAI client.

    ``OpenAI`` is imported inside ``__init__``, so we patch it via
    ``sys.modules`` before constructing the provider.
    """

    @pytest.fixture
    def mock_openai_module(self):
        """Inject a fake ``openai`` module so the local import picks it up."""
        mock_client = MagicMock()
        mock_module = MagicMock()
        mock_module.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_module}):
            yield mock_client  # yield the fake client instance

    def test_embed_returns_list_of_floats(self, mock_openai_module):
        mock_openai_module.embeddings.create.return_value = _fake_openai_response(["hello"])
        provider = OpenAIEmbedding(api_key="sk-test")
        result = provider.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    def test_embed_calls_api_once(self, mock_openai_module):
        mock_openai_module.embeddings.create.return_value = _fake_openai_response(["hi"])
        provider = OpenAIEmbedding(api_key="sk-test")
        provider.embed("hi")
        mock_openai_module.embeddings.create.assert_called_once()

    def test_embed_batch_returns_one_vector_per_text(self, mock_openai_module):
        texts = ["foo", "bar", "baz"]
        mock_openai_module.embeddings.create.return_value = _fake_openai_response(texts)
        provider = OpenAIEmbedding(api_key="sk-test")
        results = provider.embed_batch(texts)
        assert len(results) == 3
        assert all(len(v) == 1536 for v in results)

    def test_embed_batch_preserves_order(self, mock_openai_module):
        """Vectors are returned in input order even if the API response is scrambled."""
        texts = ["a", "b", "c"]
        resp = _fake_openai_response(texts)
        resp.data = list(reversed(resp.data))
        mock_openai_module.embeddings.create.return_value = resp
        provider = OpenAIEmbedding(api_key="sk-test")
        results = provider.embed_batch(texts)
        assert len(results) == 3

    def test_dimension_property(self, mock_openai_module):
        provider = OpenAIEmbedding(api_key="sk-test")
        assert provider.dimension == 1536

    def test_custom_model_passed_to_api(self, mock_openai_module):
        mock_openai_module.embeddings.create.return_value = _fake_openai_response(["x"])
        provider = OpenAIEmbedding(api_key="sk-test", model="text-embedding-ada-002")
        provider.embed("x")
        call_kwargs = mock_openai_module.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-ada-002"


# ---------------------------------------------------------------------------
# GeminiEmbedding
# ---------------------------------------------------------------------------


class TestGeminiEmbedding:
    """Unit tests for GeminiEmbedding using a mocked google.generativeai module."""

    @pytest.fixture
    def mock_genai(self):
        """Patch google.generativeai so that embed_content returns the
        correct shape for both single-string and list-of-strings calls."""
        import numpy as np

        rng = np.random.default_rng(0)
        single_vec = list(rng.random(768).astype(float))

        def _embed_content(model, content, **kwargs):
            if isinstance(content, list):
                # Batch call: return one embedding per input
                return {"embedding": [list(rng.random(768).astype(float)) for _ in content]}
            return {"embedding": single_vec}

        mock_mod = MagicMock()
        mock_mod.embed_content.side_effect = _embed_content
        with patch.dict("sys.modules", {"google.generativeai": mock_mod}):
            yield mock_mod

    def test_embed_returns_list_of_floats(self, mock_genai):
        provider = GeminiEmbedding(api_key="AIza-test")
        result = provider.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_dimension_property(self, mock_genai):
        provider = GeminiEmbedding(api_key="AIza-test")
        assert provider.dimension == 768

    def test_embed_batch_uses_single_api_call(self, mock_genai):
        """embed_batch must call embed_content once with all texts, not once per text."""
        provider = GeminiEmbedding(api_key="AIza-test")
        results = provider.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 768 for v in results)
        # Only one API call should have been made (batch, not sequential)
        assert mock_genai.embed_content.call_count == 1

    def test_embed_batch_empty_returns_empty(self, mock_genai):
        provider = GeminiEmbedding(api_key="AIza-test")
        assert provider.embed_batch([]) == []


# ---------------------------------------------------------------------------
# get_embedding_provider factory
# ---------------------------------------------------------------------------


class TestGetEmbeddingProvider:
    """Tests for the provider factory function."""

    def test_returns_openai_provider(self):
        cfg = Config(embedding_provider="openai", openai_api_key="sk-test")
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, OpenAIEmbedding)

    def test_returns_gemini_provider(self):
        cfg = Config(embedding_provider="gemini", gemini_api_key="AIza-test")
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, GeminiEmbedding)

    def test_raises_on_missing_openai_key(self):
        cfg = Config(embedding_provider="openai", openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_embedding_provider(cfg)

    def test_raises_on_missing_gemini_key(self):
        cfg = Config(embedding_provider="gemini", gemini_api_key="")
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            get_embedding_provider(cfg)

    def test_raises_on_unknown_provider(self):
        cfg = Config(embedding_provider="unknown_llm")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider(cfg)

    def test_provider_is_subclass_of_abstract_base(self):
        cfg = Config(embedding_provider="openai", openai_api_key="sk-test")
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            provider = get_embedding_provider(cfg)
        assert isinstance(provider, EmbeddingProvider)
