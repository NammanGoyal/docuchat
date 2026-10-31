"""Minimal PDF writer (text only, Helvetica) used to build test and sample PDFs
without extra dependencies. Each item in `pages` becomes one page."""

import textwrap


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str], width: int = 95) -> bytes:
    objects: list[bytes] = []
    n_pages = len(pages)
    # 1 catalog, 2 pages tree, 3 font, then (page, content) pairs.
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, text in enumerate(pages):
        lines: list[str] = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width) or [""])
        ops = ["BT", "/F1 10 Tf", "13 TL", "50 790 Td"]
        ops += [f"({_escape(line)}) Tj T*" for line in lines]
        ops.append("ET")
        stream = "\n".join(ops).encode("cp1252")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)
