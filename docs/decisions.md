# Decisions

Measured impact is recorded only when a script in this repo printed the number. Until `make eval` exists, the impact line says so.

## Phase 1

### Makefile recipes are one `python -m` line

Scoop GNU Make 4.4.1 on this machine is the Windows build. Recipes do not use shell continuations, `&&`, or bash. Docker targets are the exception: `docker compose up -d --build` and `docker compose down` are one line and have no Python entry point. Trade-off: the file is obvious in CI and in PowerShell, and it cannot express a multi-step recipe without another target. Measured impact: not a retrieval metric.

### Corpus is fictional and labeled

The carrier is Northline Telecom. Loaders reject a file unless `synthetic: true` and `company: Northline Telecom`. The directory name must match `access_level`. Trade-off: a real document cannot be dropped in without editing the loader. That is intentional for this portfolio. Measured impact: not a retrieval metric.

### Gold-set provenance

Ten of the thirty answerable questions are reserved for the author. The rest, plus the five unanswerable questions and five permission traps, are LLM-drafted. `tests/test_gold.py` spot-checks that each drafted fact is contained in the labeled chunk. Trade-off: a containment check catches mislabeled ids and does not measure answer quality. Measured impact: not a retrieval metric.

### How later tables will be reported

Comparison tables will include `n` and per-question wins, losses, and ties. A gap of one or two questions is reported as that count and is not called an improvement. The judge model is a different id from the generator, at temperature 0. Both ids are configuration, and Phase 2 verifies them with `client.models.list()` before the first call. Measured impact: not measured yet.

### Cache order

Exact cache is the first cost feature, and it lands only after Phases 1 through 8. Semantic cache and model routing wait until that exact cache exists. The access scope will be part of the exact cache key. Measured impact: not measured yet.

### Chunking

A token is a content token from `BAAI/bge-small-en-v1.5` (`encode(..., add_special_tokens=False)`). The model card allows 512 tokens including special tokens, so a 400-token chunk still fits. Overlap is 60 tokens (15 percent) and is applied inside the packer. Headings are metadata (`heading_path`). Adjacent short sections are packed up to the budget so chunks are not one heading apiece. Overlap text that would exceed the budget is dropped. Trade-off: packing can put two headings in one chunk; the path lists both, joined by ` | `. Measured impact: not a retrieval metric. Tests check the budget and the overlap on a whitespace stand-in, and the corpus test checks the real tokenizer.

### Idempotent ingest

`documents.content_hash` is the sha256 of `chunker-v1` plus the raw file bytes. Unchanged bytes and unchanged chunk hashes skip writes. A changed file upserts chunks and deletes ordinals that disappeared. `delete_missing=True` removes documents that are no longer in the corpus directory. Embeddings are null until Phase 2. Trade-off: a chunker change requires the version prefix to move, or old rows would be treated as fresh. Measured impact: the ingest report counts inserts, updates, and unchanged rows from the run, not from a guess.

### Postgres and the vector column

Image pin: `pgvector/pgvector:0.8.7-pg16`. The host port is 5433. The container still listens on 5432. Port 5432 on a developer machine is often already a local Postgres, and a password failure there looks like an app bug. CI sets `DATABASE_URL` to the GitHub service on port 5432, where nothing else is bound. The embedding column is `vector(384)`, matching bge-small-en-v1.5. There is no HNSW index. A few hundred chunks do not need an approximate index, and an approximate index would make later ablations noisy. Search will be exact `ORDER BY embedding <=> query` when retrieval arrives. Trade-off: this will not demonstrate index tuning. The decision record will say when a sequential scan is no longer acceptable. Tests use a separate database, `supportdesk_test`, created by the fixture, so pytest does not truncate the dev corpus.

### Dependencies added in this phase

| Package | Why |
|---|---|
| fastapi, uvicorn | API and the Swagger UI that ships with FastAPI |
| pydantic-settings | Model ids and chunk size stay in configuration |
| SQLAlchemy, psycopg, pgvector | Postgres 16 access and the `vector` type |
| alembic | The schema will change in later phases |
| pypdf | PDF text extraction, no OCR |
| sentence-transformers | The BGE tokenizer now, so chunk ids do not move when embeddings arrive. The same package loads the cross-encoder in Phase 4 |
| pytest, httpx, ruff | Tests, the TestClient transport, and lint |
| torch CPU wheel | sentence-transformers needs torch. The PyPI wheel is CUDA. `requirements-torch.txt` points at the CPU index |

PyYAML is not a dependency. Frontmatter is a single `key: value` block. Redis is not a dependency. Exact cache will be a Postgres table.
