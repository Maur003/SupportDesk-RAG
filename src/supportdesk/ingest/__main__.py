import json
from pathlib import Path

from supportdesk.config import get_settings
from supportdesk.db import make_engine, make_session_factory
from supportdesk.embedder import BgeEmbedder
from supportdesk.ingest.loaders import load_corpus
from supportdesk.ingest.pipeline import ingest_prepared
from supportdesk.ingest.prepare import prepare_documents
from supportdesk.ingest.tokenize import HuggingFaceCodec
from supportdesk.retrieve import embed_missing


def main() -> None:
    settings = get_settings()
    documents = load_corpus(Path(settings.corpus_dir))
    codec = HuggingFaceCodec(settings.embedding_model, settings.embedding_revision)
    prepared = prepare_documents(
        documents,
        codec=codec,
        budget=settings.chunk_token_budget,
        overlap=settings.chunk_overlap_tokens,
    )
    embedder = BgeEmbedder(settings.embedding_model, settings.embedding_revision)
    engine = make_engine(settings.database_url)
    factory = make_session_factory(engine)
    with factory() as session:
        report = ingest_prepared(session, prepared, delete_missing=True)
        embeddings_written = embed_missing(session, embedder)
        session.commit()
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    print(json.dumps({"embeddings_written": embeddings_written}, indent=2))


if __name__ == "__main__":
    main()
