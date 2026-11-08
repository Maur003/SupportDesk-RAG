import json
import re
from pathlib import Path

from supportdesk.access import ACCESS_LEVELS
from supportdesk.ingest.types import CorpusError, LoadedDocument

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_COMPANY = "Northline Telecom"
_TEXT_SUFFIXES = {".md", ".txt"}


def load_corpus(root: Path) -> list[LoadedDocument]:
    if not root.is_dir():
        raise CorpusError(f"corpus directory does not exist: {root}")
    documents: list[LoadedDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            documents.append(_load_text(path, root))
        elif suffix == ".pdf":
            documents.append(_load_pdf(path, root))
    slugs = [document.slug for document in documents]
    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicates:
        raise CorpusError(f"duplicate doc_slug values: {', '.join(duplicates)}")
    return documents


def _load_text(path: Path, root: Path) -> LoadedDocument:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusError(f"{path} is not utf-8") from exc
    meta, body = _frontmatter(text, path)
    return _document_from_meta(meta, body=body, raw=raw, path=path, root=root)


def _load_pdf(path: Path, root: Path) -> LoadedDocument:
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.is_file():
        raise CorpusError(f"missing sidecar {meta_path.name}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{meta_path.name} is not valid json") from exc
    if not isinstance(meta, dict):
        raise CorpusError(f"{meta_path.name} must be a json object")
    string_meta = {str(key): str(value) for key, value in meta.items()}
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append(extracted.strip())
    body = "\n\n".join(pages)
    if not body.strip():
        raise CorpusError(f"{path.name} has no extractable text")
    return _document_from_meta(
        string_meta,
        body=body,
        raw=path.read_bytes(),
        path=path,
        root=root,
    )


def _document_from_meta(
    meta: dict[str, str],
    *,
    body: str,
    raw: bytes,
    path: Path,
    root: Path,
) -> LoadedDocument:
    synthetic = meta.get("synthetic", "").strip().lower()
    if synthetic != "true":
        raise CorpusError(f"{path.name} must set synthetic: true")
    company = meta.get("company", "").strip()
    if company != _COMPANY:
        raise CorpusError(f"{path.name} must set company: {_COMPANY}")
    access_level = meta.get("access_level", "").strip()
    if access_level not in ACCESS_LEVELS:
        raise CorpusError(f"{path.name} has access_level {access_level!r}")
    parent = path.parent.name
    if parent != access_level:
        raise CorpusError(f"{path.name} is in {parent!r} but access_level is {access_level!r}")
    slug = meta.get("doc_slug", "").strip()
    if not _SLUG.fullmatch(slug):
        raise CorpusError(f"{path.name} has an invalid doc_slug {slug!r}")
    title = meta.get("title", "").strip()
    if not title:
        raise CorpusError(f"{path.name} is missing title")
    source_path = path.relative_to(root).as_posix()
    return LoadedDocument(
        slug=slug,
        title=title,
        access_level=access_level,
        synthetic=True,
        source_path=source_path,
        body=body.strip(),
        raw_bytes=raw,
    )


def _frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        raise CorpusError(f"{path.name} is missing frontmatter")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        raise CorpusError(f"{path.name} has unterminated frontmatter")
    meta: dict[str, str] = {}
    for line in normalized[4:end].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise CorpusError(f"{path.name} has a frontmatter line without a colon")
        meta[key.strip()] = value.strip()
    body = normalized[end + len("\n---\n") :]
    return meta, body
