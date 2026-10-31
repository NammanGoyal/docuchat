"""Offline, deterministic providers: no network, no model weights, no API key.

HashingEmbedder
    A "hashing trick" bag-of-features vector. Each feature (a stemmed word,
    an adjacent word pair, or a character trigram) is hashed to one of `dim`
    buckets with a +/-1 sign, weighted by 1 + log(tf), and the vector is
    L2-normalised so cosine similarity equals the dot product. This captures
    lexical (not semantic) similarity; character trigrams add tolerance for
    morphology and typos. The same text always produces the same vector.

ExtractiveAnswerer
    Instead of generating text, it quotes the sentences from the retrieved
    passages that share the most terms with the question, each followed by
    its [n] citation marker. It can never hallucinate: every answer sentence
    is verbatim from a cited passage.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterator

from app.providers.base import Passage
from app.text import content_terms

_EMPTY_FEATURE = "__empty__"


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


def _features(text: str) -> Counter[str]:
    terms = content_terms(text)
    feats: Counter[str] = Counter()
    for t in terms:
        feats["w:" + t] += 1
        padded = f"#{t}#"
        for i in range(len(padded) - 2):
            feats["c:" + padded[i : i + 3]] += 1
    for a, b in zip(terms, terms[1:], strict=False):
        feats[f"b:{a}_{b}"] += 1
    return feats


# Relative importance of each feature family.
_WEIGHTS = {"w": 1.0, "b": 0.7, "c": 0.25}


class HashingEmbedder:
    def __init__(self, dim: int):
        self.dim = dim
        self.name = f"local-hashing-v1-{dim}"

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        feats = _features(text) or Counter({_EMPTY_FEATURE: 1})
        for feat, tf in feats.items():
            idx, sign = _bucket(feat, self.dim)
            vec[idx] += sign * _WEIGHTS.get(feat[0], 1.0) * (1.0 + math.log(tf))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:  # every feature cancelled out; vanishingly rare
            idx, _ = _bucket(_EMPTY_FEATURE, self.dim)
            vec[idx] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=[-*#>\d])")


def split_sentences(text: str) -> list[str]:
    parts = (s.strip(" \n-*#>") for s in _SENTENCE_SPLIT.split(text))
    return [" ".join(p.split()) for p in parts if len(p.split()) >= 3]


class ExtractiveAnswerer:
    name = "local-extractive-v1"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def select(self, question: str, passages: list[Passage]) -> list[tuple[str, int]]:
        """Return up to `max_sentences` (sentence, marker) pairs, best first."""
        q_terms = set(content_terms(question))
        scored: list[tuple[float, int, int, str, int]] = []
        for rank, p in enumerate(passages):
            for pos, sentence in enumerate(split_sentences(p.text)):
                overlap = len(q_terms & set(content_terms(sentence)))
                if overlap:
                    # More shared terms first; then prefer higher-ranked passages
                    # and earlier sentences.
                    scored.append((overlap - 0.1 * rank, -rank, -pos, sentence, p.marker))
        scored.sort(reverse=True)
        picked: list[tuple[str, int]] = []
        seen: set[str] = set()
        for *_, sentence, marker in scored:
            if sentence not in seen:
                seen.add(sentence)
                picked.append((sentence, marker))
            if len(picked) == self.max_sentences:
                break
        if not picked and passages:
            first = split_sentences(passages[0].text) or [passages[0].text.strip()]
            picked.append((first[0], passages[0].marker))
        return picked

    def stream_answer(self, question: str, passages: list[Passage]) -> Iterator[str]:
        yield "From your documents:\n"
        for sentence, marker in self.select(question, passages):
            yield f"\n> {sentence} [{marker}]\n"
