import re

from supportdesk.ingest.tokenize import TokenCodec, token_count
from supportdesk.ingest.types import ChunkDraft

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_SEPARATORS = ("\n\n", "\n", ". ")


def chunk_document(
    body: str,
    *,
    title: str,
    budget: int,
    overlap: int,
    codec: TokenCodec,
) -> list[ChunkDraft]:
    """Split a document into chunks of at most `budget` content tokens.

    Headings become `heading_path` metadata and are not repeated in the chunk
    text. Overlap is the decoded tail of the previous chunk inside the same
    section. Overlap does not cross a heading. A decoded tail that would push
    the next chunk over the budget is dropped.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if overlap < 0 or overlap >= budget:
        raise ValueError("overlap must be >= 0 and smaller than budget")

    drafts: list[ChunkDraft] = []
    buffer_text = ""
    buffer_headings: list[str] = []

    def emit(text: str, headings: list[str]) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        count = token_count(codec, cleaned)
        if count > budget:
            raise RuntimeError(f"chunker emitted {count} tokens, over the budget of {budget}")
        drafts.append(
            ChunkDraft(
                ordinal=len(drafts),
                text=cleaned,
                heading_path=" | ".join(headings),
                token_count=count,
            )
        )

    for heading_path, section in _sections(body, title):
        if token_count(codec, section) > budget:
            if buffer_text:
                emit(buffer_text, buffer_headings)
                buffer_text = ""
                buffer_headings = []
            for piece in _chunk_text(section, codec, budget, overlap):
                emit(piece, [heading_path])
            continue
        proposed = section if not buffer_text else f"{buffer_text}\n\n{section}"
        if buffer_text and token_count(codec, proposed) <= budget:
            buffer_text = proposed
            if heading_path not in buffer_headings:
                buffer_headings.append(heading_path)
            continue
        if buffer_text:
            tail = _tail(buffer_text, codec, overlap)
            emit(buffer_text, buffer_headings)
            with_tail = f"{tail}\n\n{section}" if tail else section
            if tail and token_count(codec, with_tail) <= budget:
                buffer_text = with_tail
            else:
                buffer_text = section
            buffer_headings = [heading_path]
            continue
        buffer_text = section
        buffer_headings = [heading_path]
    if buffer_text:
        emit(buffer_text, buffer_headings)
    return drafts


def _sections(body: str, title: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    heading_path = title.strip() or "(untitled)"

    def flush() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            sections.append((heading_path, text))

    for line in body.splitlines():
        match = _HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue
        flush()
        level = len(match.group(1))
        name = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        heading_path = " > ".join(part for _, part in stack)
    flush()
    return sections


def _chunk_text(text: str, codec: TokenCodec, budget: int, overlap: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if token_count(codec, stripped) <= budget:
        return [stripped]

    parts = _split_once(stripped)
    if parts is None:
        return _windows(stripped, codec, budget, overlap)

    packed: list[str] = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if token_count(codec, part) > budget:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_chunk_text(part, codec, budget, overlap))
            continue
        proposed = part if not buffer else f"{buffer}\n\n{part}"
        if token_count(codec, proposed) <= budget:
            buffer = proposed
            continue
        packed.append(buffer)
        tail = _tail(buffer, codec, overlap)
        with_tail = f"{tail}\n\n{part}" if tail else part
        if tail and token_count(codec, with_tail) <= budget:
            buffer = with_tail
        else:
            buffer = part
    if buffer:
        packed.append(buffer)
    return packed


def _split_once(text: str) -> list[str] | None:
    for separator in _SEPARATORS:
        if separator not in text:
            continue
        raw_parts = text.split(separator)
        kept = [part for part in raw_parts if part.strip()]
        if len(kept) <= 1:
            continue
        if separator == ". ":
            with_periods: list[str] = []
            for index, part in enumerate(raw_parts):
                piece = part + "." if index < len(raw_parts) - 1 else part
                if piece.strip():
                    with_periods.append(piece)
            return with_periods
        return kept
    return None


def _windows(text: str, codec: TokenCodec, budget: int, overlap: int) -> list[str]:
    ids = codec.encode(text)
    if not ids:
        return []
    step = budget - overlap
    pieces: list[str] = []
    start = 0
    while start < len(ids):
        piece = _fit_decode(codec, ids[start : start + budget], budget)
        if piece:
            pieces.append(piece)
        if start + budget >= len(ids):
            break
        start += step
    return pieces


def _fit_decode(codec: TokenCodec, token_ids: list[int], budget: int) -> str:
    width = len(token_ids)
    while width > 0:
        piece = codec.decode(token_ids[:width]).strip()
        if piece and token_count(codec, piece) <= budget:
            return piece
        width -= 1
    return ""


def _tail(text: str, codec: TokenCodec, overlap: int) -> str:
    if overlap <= 0:
        return ""
    ids = codec.encode(text)
    if not ids:
        return ""
    return codec.decode(ids[-overlap:]).strip()
