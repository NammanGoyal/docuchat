"""Split extracted pages into overlapping, character-bounded chunks.

Chunks never cross page boundaries, so every chunk can be cited with a single
page number. Within a page we cut on whitespace: a chunk is the longest run of
whole words that fits in `size` characters, and the next chunk starts at the
first word that begins inside the last `overlap` characters of the previous
chunk. Overlap keeps a sentence that straddles a cut retrievable from either
side.
"""

import re
from dataclasses import dataclass

from app.ingestion.extract import Page

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    index: int  # position of the chunk within its document (0-based)
    page: int | None  # 1-based page number for PDFs, None for plain text
    text: str


def _word_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Word (start, end) offsets; words longer than `size` are hard-split."""
    spans: list[tuple[int, int]] = []
    for m in _WORD.finditer(text):
        start, end = m.span()
        while end - start > size:
            spans.append((start, start + size))
            start += size
        spans.append((start, end))
    return spans


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    spans = _word_spans(text, size)
    chunks: list[str] = []
    i = 0
    while i < len(spans):
        chunk_start = spans[i][0]
        j = i
        # Greedily take words while the chunk still fits in `size` characters.
        while j + 1 < len(spans) and spans[j + 1][1] - chunk_start <= size:
            j += 1
        chunk_end = spans[j][1]
        chunks.append(text[chunk_start:chunk_end])
        if j == len(spans) - 1:
            break
        # Next chunk begins at the first word starting within the overlap window,
        # but always advances by at least one word so the loop terminates.
        next_i = j + 1
        k = j
        while k > i and spans[k][0] >= chunk_end - overlap:
            next_i = k
            k -= 1
        i = next_i
    return chunks


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, size, overlap):
            chunks.append(Chunk(index=len(chunks), page=page.number, text=text))
    return chunks
