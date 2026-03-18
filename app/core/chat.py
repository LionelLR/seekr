"""LLM chat providers for RAG-based question answering over video transcripts.

Supports OpenAI (GPT-4o-mini, etc.) and Google Gemini.  Both providers
stream response tokens so the frontend can render a typewriter effect.

Usage::

    from app.core.chat import get_chat_provider
    provider = get_chat_provider(cfg)   # None if chat_model not set
    if provider:
        for token in provider.stream_answer(question, chunks):
            print(token, end="", flush=True)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based solely on video "
    "transcript segments provided as context.\n\n"
    "Rules:\n"
    "- Answer ONLY using the provided context. Do not use outside knowledge.\n"
    "- Cite the source of each piece of information inline using [1], [2], etc.\n"
    "- Be concise: 2–5 sentences is ideal.\n"
    "- If the answer is not clearly present in the context, say so honestly.\n"
    "- Do not repeat the question. Do not add a preamble."
)


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format transcript chunks into a numbered context string for the LLM."""
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("video_title") or c.get("source_url") or c.get("video_url", "")
        platform = c.get("platform", "")
        start = int(c.get("start", 0))
        if platform == "pdf":
            end = int(c.get("end", start))
            ts = f"page {start}" if start == end else f"pages {start}–{end}"
        else:
            m, s = divmod(start, 60)
            ts = f"{m}:{s:02d}"
        parts.append(f"[{i}] {title} @ {ts}\n{c['text']}")
    return "\n\n".join(parts)


class ChatProvider(ABC):
    """Abstract base class for streaming LLM providers."""

    @abstractmethod
    def stream_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        """Yield response tokens one at a time."""
        ...


class OpenAIChat(ChatProvider):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        import openai  # lazy import — only needed if OpenAI chat is used
        self._client = openai.OpenAI()
        self.model = model

    def stream_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        context = _build_context(context_chunks)
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            stream=True,
            max_tokens=600,
            temperature=0.2,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text


class GeminiChat(ChatProvider):
    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        import google.generativeai as genai  # lazy import
        self._model = genai.GenerativeModel(model)

    def stream_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Generator[str, None, None]:
        context = _build_context(context_chunks)
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        response = self._model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


def get_chat_provider(cfg) -> Optional[ChatProvider]:
    """Instantiate the appropriate chat provider from config.

    Returns ``None`` when ``chat_model`` is not set, which disables the
    AI answer feature and falls back to plain vector search.

    Args:
        cfg: :class:`~app.core.config.Config` instance.

    Returns:
        A :class:`ChatProvider` instance, or ``None``.
    """
    model: str = getattr(cfg, "chat_model", "")
    if not model:
        return None
    provider: str = getattr(cfg, "embedding_provider", "openai")
    try:
        if provider == "gemini":
            return GeminiChat(model)
        return OpenAIChat(model)
    except Exception:
        # Missing API key or other init error — chat disabled until key is set
        return None
