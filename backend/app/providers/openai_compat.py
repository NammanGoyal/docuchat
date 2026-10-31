"""Providers for any OpenAI-compatible HTTP API (OpenAI, Azure-style gateways,
Ollama, vLLM, LM Studio, ...). Uses plain httpx so the wire protocol is visible."""

import json
from collections.abc import Iterator

import httpx

from app.providers.base import Passage, ProviderError

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. "
    "If the passages do not contain the answer, reply exactly: "
    '"I couldn\'t find that in your documents." Be concise.'
)


def _client(base_url: str, api_key: str | None, timeout: float, transport: httpx.BaseTransport | None) -> httpx.Client:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout, transport=transport)


class OpenAIEmbedder:
    batch_size = 64

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        dim: int,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,  # injectable for tests
    ):
        self.model = model
        self.dim = dim
        self.name = f"openai:{model}:{dim}"
        self._http = _client(base_url, api_key, timeout, transport)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                resp = self._http.post("embeddings", json={"model": self.model, "input": batch, "dimensions": self.dim})
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                raise ProviderError(f"embedding request failed: {exc}") from exc
            for item in data:
                vec = item["embedding"]
                if len(vec) != self.dim:
                    raise ProviderError(
                        f"model {self.model} returned {len(vec)}-d vectors; expected {self.dim}. "
                        "Use a model that supports the `dimensions` parameter."
                    )
                out.append(vec)
        return out


def build_messages(question: str, passages: list[Passage]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{p.marker}] (source: {p.document}{f', page {p.page}' if p.page else ''})\n{p.text}" for p in passages
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"},
    ]


class OpenAIChatAnswerer:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,  # injectable for tests
    ):
        self.model = model
        self.name = f"openai:{model}"
        self._http = _client(base_url, api_key, timeout, transport)

    def stream_answer(self, question: str, passages: list[Passage]) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": build_messages(question, passages),
            "temperature": 0,
            "stream": True,
        }
        try:
            with self._http.stream("POST", "chat/completions", json=body) as resp:
                resp.raise_for_status()
                # Server-sent events: lines of `data: {json}` ending with `data: [DONE]`.
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    choices = json.loads(payload).get("choices") or []
                    delta = choices[0].get("delta", {}).get("content") if choices else None
                    if delta:
                        yield delta
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"chat completion failed: {exc}") from exc
