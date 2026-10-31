import pytest

from app.ingestion.chunking import chunk_pages, split_text
from app.ingestion.extract import Page

WORDS = [f"word{i:03d}" for i in range(300)]
TEXT = " ".join(WORDS)


def test_chunks_respect_size_limit():
    chunks = split_text(TEXT, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_cover_every_word_in_order():
    chunks = split_text(TEXT, size=100, overlap=20)
    seen: list[str] = []
    for c in chunks:
        for w in c.split():
            if not seen or WORDS.index(w) > WORDS.index(seen[-1]):
                seen.append(w)
    assert seen == WORDS


def test_consecutive_chunks_overlap():
    chunks = split_text(TEXT, size=100, overlap=30)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        shared = set(prev.split()) & set(nxt.split())
        assert shared, "expected the overlap window to repeat at least one word"
        # the overlap must come from the *end* of the previous chunk
        assert nxt.split()[0] in prev.split()[-4:]


def test_zero_overlap_partitions_words():
    chunks = split_text(TEXT, size=100, overlap=0)
    assert " ".join(chunks).split() == WORDS


def test_short_text_is_single_chunk():
    assert split_text("just a few words", size=100, overlap=10) == ["just a few words"]


def test_empty_text_has_no_chunks():
    assert split_text("   \n  ", size=100, overlap=10) == []


def test_word_longer_than_size_is_hard_split():
    chunks = split_text("x" * 250, size=100, overlap=10)
    assert [len(c) for c in chunks] == [100, 100, 50]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text(TEXT, size=size, overlap=overlap)


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [Page(1, "alpha " * 50), Page(2, "beta " * 50)]
    chunks = chunk_pages(pages, size=120, overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        words = set(c.text.split())
        assert words == ({"alpha"} if c.page == 1 else {"beta"})
    assert {c.page for c in chunks} == {1, 2}
