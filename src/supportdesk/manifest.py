import json
from pathlib import Path

from supportdesk.config import get_settings
from supportdesk.ingest.loaders import load_corpus
from supportdesk.ingest.prepare import prepare_documents
from supportdesk.ingest.tokenize import HuggingFaceCodec

MANIFEST_PATH = Path("data/gold/chunk_manifest.json")


def build_manifest_payload(corpus_dir: Path, model_name: str, budget: int, overlap: int) -> dict:
    documents = load_corpus(corpus_dir)
    codec = HuggingFaceCodec(model_name)
    prepared = prepare_documents(documents, codec=codec, budget=budget, overlap=overlap)
    chunks = []
    for document in prepared:
        for chunk in document.chunks:
            chunks.append(
                {
                    "id": chunk.id,
                    "doc_slug": document.slug,
                    "access_level": document.access_level,
                    "heading_path": chunk.heading_path,
                    "ordinal": chunk.ordinal,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                }
            )
    return {
        "synthetic": True,
        "company": "Northline Telecom",
        "embedding_model": model_name,
        "token_budget": budget,
        "token_overlap": overlap,
        "tokenizer": "content tokens only; special tokens are not counted",
        "documents": len(prepared),
        "chunks": chunks,
    }


def main() -> None:
    settings = get_settings()
    payload = build_manifest_payload(
        Path(settings.corpus_dir),
        settings.embedding_model,
        settings.chunk_token_budget,
        settings.chunk_overlap_tokens,
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "path": MANIFEST_PATH.as_posix(),
                "documents": payload["documents"],
                "chunks": len(payload["chunks"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
