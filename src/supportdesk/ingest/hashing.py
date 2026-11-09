import hashlib

CHUNKER_VERSION = "chunker-v1"


def document_hash(raw_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(CHUNKER_VERSION.encode("utf-8"))
    digest.update(b"\n")
    digest.update(raw_bytes)
    return digest.hexdigest()


def chunk_hash(title: str, heading_path: str, text: str, access_level: str) -> str:
    payload = f"{title}\n{heading_path}\n{text}\n{access_level}".encode()
    return hashlib.sha256(payload).hexdigest()
