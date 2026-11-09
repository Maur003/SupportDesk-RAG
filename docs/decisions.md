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

## Phase 1.5

### Corpus size and near-duplicates

The first corpus was 52 documents and 57 chunks, mostly one chunk per file, so a lexical search could not really fail. Phase 1.5 lengthens pages and adds parallel pages that differ by a figure: Fiber 300, 500, 1000, and 2000; residential versus small-business outage credit, for the household and for the agent; a customer refund request versus the finance approval matrix; a 7-business-day posting clock versus same-day ledger posting. Neighborhood banners, symptom pages, fee-versus-waiver pairs, and ledger cases use the same shape with a different code or threshold. `tools/expand_corpus.py` rewrites those files. The opening fact on a hand-written page stays in place. A test fails if chunking produces fewer than 250 chunks. The manifest command prints the document count and the chunk count. The README quotes that printout. Trade-off: many pages share a frame, so a retriever that matches the frame and misses the figure will look confident and be wrong. Measured impact: not a retrieval metric yet. The size itself is the manifest printout in the README.

### Paraphrases and roles

`a01`–`a20` stay close to the source wording so a later keyword run has something to win. `b01`–`b22` are paraphrases. `tests/test_gold.py` rejects a paraphrase that shares a content word with the labeled chunk. Agent and finance each have at least eight drafted answerable questions. `u01`–`u10` stay empty. The hints tell the author to write paraphrases. Trade-off: a containment check still does not measure whether a model would answer the paraphrase. Measured impact: not a retrieval metric.

### Canaries on write-off and executive credits

`WRITEOFF_CANARY_8HN4` is only in `finance/write-off-policy.md`. `EXECUTIVE_CANARY_2VX6` is only in `finance/executive-credits.md`. `p04` and `p05` use those strings the same way `p01` and `p03` use the older canaries. A test counts each canary in exactly one file. Trade-off: the trap is "was the hidden string retrieved," not a graded essay. Measured impact: not measured until retrieval exists.

### How results tables will be reported

Every results table includes the corpus size, meaning the chunk count in the manifest, and a random-retrieval row. That row draws `k` chunks uniformly from the chunks the role may see, with a fixed seed, and scores them with the same metric. `recall@20` is not a headline number. The cutoff that matches the context window is `recall@5`. A wider cutoff is shown only next to the corpus size, because 20 out of 57 and 20 out of a few hundred are different claims. Wins, losses, ties, and `n` still sit beside any percentage. Measured impact: not measured until `make eval` exists. No baseline number is filled in before that.

### Pinned tokenizer, caches, and the lock split

`EMBEDDING_REVISION` is the Hub commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, the snapshot already in the local tokenizer cache. Settings reject a branch name such as `main`. Chunk ids move if this commit changes, so the manifest records the revision and CI caches `~/.cache/huggingface` with a key that includes `src/supportdesk/config.py`. `actions/setup-python` caches pip from `requirements-torch.txt`, `requirements.txt`, and `requirements-dev.txt`.

`requirements.txt` is the runtime lock. The image installs that file and does not install pytest or ruff. `requirements-dev.txt` includes the runtime lock plus pytest, ruff, and `httpx==0.28.1` (the FastAPI TestClient transport). `httpx` moves to the runtime lock when the mock CRM client needs it.

`httpx2==2.16.0` and `httpcore2==2.16.0` stay in the runtime lock. `pip show` reports that `huggingface_hub` 2.4.0 requires `httpx2`, and `httpx2` requires `httpcore2`. They are a different project from `httpx`. Removing them breaks the tokenizer download. They are not an extra HTTP client we chose.

### Phase 2 embedding input

Phase 2 will embed `embedding_input(title, heading_path, text)` from `src/supportdesk/embed_text.py`: the document title, the heading path, and the chunk text, joined by newlines. Empty parts are skipped. `chunks.text` stays the body only. Citations and the gold-fact check use that column, not the embedded string. The generated tsvector already concatenates `heading_path` and `text`. Phase 2 will add the document title to that tsvector so keyword search sees the same fields. Trade-off: the vector is not the citation string, so a later check that a citation was actually retrieved must use the stored text. Measured impact: not measured until retrieval exists. The string is fixed now so Phase 2 does not invent a different concatenation.
