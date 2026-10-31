"""Render eval/sources/balcony-garden-guide.txt into sample_docs/balcony-garden-guide.pdf.

Lines reading `=== PAGE BREAK ===` in the source mark page breaks. Run from backend/:
    python -m eval.build_sample_pdf
"""

from pathlib import Path

from tests.pdfgen import make_pdf

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent / "sources" / "balcony-garden-guide.txt"
DEST = ROOT / "sample_docs" / "balcony-garden-guide.pdf"
PAGE_BREAK = "\n=== PAGE BREAK ===\n"


def main() -> None:
    pages = [p.strip() for p in SRC.read_text().split(PAGE_BREAK)]
    DEST.write_bytes(make_pdf(pages))
    print(f"wrote {DEST} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
