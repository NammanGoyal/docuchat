import pytest

from app.ingestion.extract import ExtractionError, detect_kind, extract_pages
from tests.pdfgen import make_pdf


def test_pdf_pages_are_numbered_from_one():
    data = make_pdf(["Page one talks about tomatoes.", "Page two talks about aphids."])
    pages = extract_pages("guide.PDF", data)
    assert [p.number for p in pages] == [1, 2]
    assert "tomatoes" in pages[0].text and "aphids" in pages[1].text


def test_blank_pdf_pages_are_skipped_but_numbering_is_kept():
    pages = extract_pages("x.pdf", make_pdf(["first", "", "third"]))
    assert [p.number for p in pages] == [1, 3]


def test_markdown_and_text_are_single_unnumbered_page():
    pages = extract_pages("notes.md", b"# Title\r\n\r\n\r\n\r\nBody   text\twith  spaces")
    assert len(pages) == 1 and pages[0].number is None
    assert pages[0].text == "# Title\n\nBody text with spaces"


@pytest.mark.parametrize("name", ["report.docx", "noextension", "image.png"])
def test_unsupported_types_rejected(name):
    with pytest.raises(ExtractionError, match="unsupported file type"):
        detect_kind(name)


def test_non_utf8_text_rejected():
    with pytest.raises(ExtractionError, match="UTF-8"):
        extract_pages("a.txt", "café".encode("latin-1"))


def test_corrupt_pdf_rejected():
    with pytest.raises(ExtractionError):
        extract_pages("a.pdf", b"%PDF-1.4 this is not really a pdf")


def test_whitespace_only_document_rejected():
    with pytest.raises(ExtractionError, match="no extractable text"):
        extract_pages("a.txt", b"  \n\n  ")
