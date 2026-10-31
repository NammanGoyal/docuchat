"""Provider interfaces. Swapping embedding/LLM backends only touches this package."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


class ProviderError(RuntimeError):
    """An upstream embedding/LLM call failed."""


class Embedder(Protocol):
    # Stored next to every chunk vector: vectors from different models live in
    # different spaces and must never be compared with each other.
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Passage:
    marker: int  # the [n] the answer uses to cite this passage
    document: str
    page: int | None
    text: str


class AnswerGenerator(Protocol):
    name: str

    def stream_answer(self, question: str, passages: list[Passage]) -> Iterator[str]:
        """Yield the answer incrementally. Must cite passages as [marker]."""
        ...
