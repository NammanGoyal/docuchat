"""Turn uploaded bytes into plain-text pages."""

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}


class ExtractionError(ValueError):
    """The file could not be read as a supported document."""


@dataclass(frozen=True)
class Page:
    number: int | None  # 1-based for PDFs; None for formats without pages
    text: str


def detect_kind(filename: str) -> str:
    ext = PurePath(filename).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[ext]
    except KeyError:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ExtractionError(f"unsupported file type {ext or '(none)'}; allowed: {allowed}") from None


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(filename: str, data: bytes) -> list[Page]:
    """Return non-empty pages of text. Raises ExtractionError if nothing is readable."""
    kind = detect_kind(filename)
    if kind == "pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            raw = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except (PdfReadError, ValueError, KeyError, TypeError) as exc:
            raise ExtractionError(f"could not parse PDF: {exc}") from exc
        pages = [Page(n, _normalize(t)) for n, t in raw]
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError("text files must be UTF-8 encoded") from exc
        pages = [Page(None, _normalize(text))]

    pages = [p for p in pages if p.text]
    if not pages:
        raise ExtractionError("no extractable text found (scanned PDFs are not supported)")
    return pages
