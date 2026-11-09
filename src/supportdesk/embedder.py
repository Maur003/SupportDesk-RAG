"""Local BGE embeddings.

Passages are the title, heading path, and chunk text, with no query instruction.
Queries are prefixed with the BGE v1.5 instruction. Vectors are L2-normalized
so pgvector cosine distance ranks them by cosine similarity.
"""

from supportdesk.config import BGE_SMALL_DIMENSIONS

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbedderError(Exception):
    """The embedding model could not encode a query. Retrieval can fall back to keywords."""


class BgeEmbedder:
    def __init__(self, model_name: str, revision: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.model_id = f"{model_name}@{revision}"
        self._model = SentenceTransformer(model_name, revision=revision, device="cpu")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # prompt="" keeps encode_document from applying a saved passage prompt.
        vectors = self._model.encode_document(
            texts,
            prompt="",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        rows = [vector.tolist() for vector in vectors]
        _check_dimensions(rows)
        return rows

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = self._model.encode_query(
                QUERY_INSTRUCTION + text,
                prompt="",
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except EmbedderError:
            raise
        except Exception as exc:
            raise EmbedderError("embedder unavailable") from exc
        row = vector.tolist()
        _check_dimensions([row])
        return row


def _check_dimensions(rows: list[list[float]]) -> None:
    for row in rows:
        if len(row) != BGE_SMALL_DIMENSIONS:
            raise RuntimeError(f"expected {BGE_SMALL_DIMENSIONS} dimensions, got {len(row)}")
