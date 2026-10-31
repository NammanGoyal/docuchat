from app.config import EMBEDDING_DIM, Settings
from app.providers.base import AnswerGenerator, Embedder, Passage, ProviderError
from app.providers.local import ExtractiveAnswerer, HashingEmbedder
from app.providers.openai_compat import OpenAIChatAnswerer, OpenAIEmbedder

__all__ = [
    "AnswerGenerator",
    "Embedder",
    "Passage",
    "ProviderError",
    "build_answerer",
    "build_embedder",
]


def build_embedder(s: Settings) -> Embedder:
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(
            s.openai_base_url, s.openai_api_key, s.openai_embedding_model, EMBEDDING_DIM, s.openai_timeout_s
        )
    return HashingEmbedder(EMBEDDING_DIM)


def build_answerer(s: Settings) -> AnswerGenerator:
    if s.llm_provider == "openai":
        return OpenAIChatAnswerer(s.openai_base_url, s.openai_api_key, s.openai_chat_model, s.openai_timeout_s)
    return ExtractiveAnswerer()
