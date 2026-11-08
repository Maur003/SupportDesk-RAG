import json

import pytest

from supportdesk.ingest.loaders import load_corpus
from supportdesk.ingest.types import CorpusError


def _write(root, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _doc(slug: str, level: str, body: str = "Body text.") -> str:
    return (
        "---\n"
        "synthetic: true\n"
        "company: Northline Telecom\n"
        f"access_level: {level}\n"
        f"doc_slug: {slug}\n"
        f"title: {slug}\n"
        "---\n\n"
        f"{body}\n"
    )


def test_loader_reads_markdown_and_text(tmp_path) -> None:
    _write(tmp_path, "public/alpha.md", _doc("alpha", "public"))
    _write(tmp_path, "agent/beta.txt", _doc("beta", "agent", "Text body."))
    documents = load_corpus(tmp_path)
    by_slug = {document.slug: document for document in documents}
    assert by_slug["alpha"].access_level == "public"
    assert by_slug["alpha"].synthetic is True
    assert by_slug["beta"].body == "Text body."
    assert by_slug["alpha"].source_path == "public/alpha.md"


def test_loader_rejects_unlabeled_file(tmp_path) -> None:
    _write(
        tmp_path,
        "public/nope.md",
        "---\nsynthetic: false\ncompany: Northline Telecom\n"
        "access_level: public\ndoc_slug: nope\ntitle: Nope\n---\n\nBody\n",
    )
    with pytest.raises(CorpusError, match="synthetic"):
        load_corpus(tmp_path)


def test_loader_rejects_access_level_that_disagrees_with_the_directory(tmp_path) -> None:
    _write(tmp_path, "public/secret.md", _doc("secret", "finance"))
    with pytest.raises(CorpusError, match="access_level"):
        load_corpus(tmp_path)


def test_loader_reads_a_pdf_sidecar(tmp_path) -> None:
    pdf = tmp_path / "public" / "note.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(_tiny_pdf("Northline schedules maintenance on Tuesday."))
    sidecar = {
        "synthetic": "true",
        "company": "Northline Telecom",
        "access_level": "public",
        "doc_slug": "note",
        "title": "Note",
    }
    (tmp_path / "public" / "note.pdf.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    documents = load_corpus(tmp_path)
    assert documents[0].slug == "note"
    assert "Tuesday" in documents[0].body


def _tiny_pdf(text: str) -> bytes:
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        + stream
        + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    header = b"%PDF-1.4\n"
    offsets = []
    cursor = len(header)
    for obj in objects:
        offsets.append(cursor)
        cursor += len(obj)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    xref += b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{cursor}\n%%EOF\n".encode("ascii")
    return header + b"".join(objects) + xref + trailer
