# SupportDesk RAG

Synthetic customer-support agent for the fictional carrier **Northline Telecom**. Every document and gold question is synthetic. Nothing here is a real carrier's policy.

This repository is at the end of Phase 1.5: a runnable API, Postgres 16 with pgvector, a larger corpus, and idempotent ingestion. Retrieval metrics are not measured yet. Later phases will paste numbers from `make eval` only. A difference of one or two questions will be reported as that count, and will not be described as an improvement. Every comparison table will show `n`, the corpus size (the chunk count in the manifest), a random-retrieval row, and the per-question wins, losses, and ties beside any percentage. `recall@20` is not a headline metric. The cutoff that matches the context window is `recall@5`. A wider cutoff is shown only next to the corpus size.

## Gold set

`data/gold/questions.json` has 62 questions:

- 10 answerable questions are reserved for the author (`author` = `user`, `u01`–`u10`). They are unfilled. The hints ask for a paraphrase with little of the page's wording.
- 20 answerable questions (`a01`–`a20`) stay close to the source wording.
- 22 answerable questions (`b01`–`b22`) are paraphrases. The spot-check rejects one that shares a content word with the labeled chunk. Agent and finance are both in this group.
- 5 unanswerable questions and 5 permission traps are LLM-drafted. `p04` and `p05` use the write-off and executive-credit canaries.

The spot-check is `tests/test_gold.py`: each labeled fact must occur in the labeled chunk. That is not an independent human review.

## Quickstart

Activate the project virtualenv before `make`. The Makefile calls `python`, and GNU Make on Windows does not activate the virtualenv for you.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
make deps-torch
make deps-dev
make install
copy .env.example .env
docker compose up -d db
make migrate
make ingest
make test
make lint
```

`make up` builds the API image and starts it with Postgres. Swagger is at `http://localhost:8000/docs`. `GET /health` returns `{"status": "ok"}`.

Postgres is published on host port **5433**. The container still listens on 5432. Port 5432 is left alone because a local Postgres often already owns it. CI sets `DATABASE_URL` itself and uses 5432 there.

`python -m supportdesk.manifest` printed:

```json
{
  "path": "data/gold/chunk_manifest.json",
  "documents": 121,
  "chunks": 280
}
```

On an empty database, `make ingest` printed:

```json
{
  "chunks_written": 280,
  "deleted_chunks": 0,
  "deleted_documents": 0,
  "documents_seen": 121,
  "inserted": 121,
  "unchanged": 0,
  "updated": 0
}
```

The next run printed `"unchanged": 121` and `"chunks_written": 0`. Edit a file and those counts move. They are script output, not a target.

`docker compose` targets are one line each. Every Python target is one `python -m ...` call, so the same Makefile works in PowerShell, Scoop's GNU Make, and CI.

Torch is installed from the CPU index in `requirements-torch.txt` before the lockfiles. The default PyPI torch wheel is the CUDA build. `make deps` installs the runtime lock only. `make deps-dev` installs that lock plus pytest, ruff, and `httpx` (the TestClient transport). The API image uses the runtime lock and does not install the dev packages.

`httpx2` and `httpcore2` are in the runtime lock because `huggingface_hub` 2.4 imports `httpx2` to download the tokenizer, and `httpx2` imports `httpcore2`. They are a different project from `httpx`. The tokenizer revision is the Hub commit in `EMBEDDING_REVISION`, not the branch `main`.

## Layout

- `data/corpus/{public,agent,finance}` — synthetic knowledge base. `synthetic: true` is required on every file. `tools/expand_corpus.py` rewrites the long near-duplicate pages.
- Phase 2 will embed the document title, the heading path, and the chunk text together. The helper is `src/supportdesk/embed_text.py`. Retrieval is not implemented yet.
- `src/supportdesk/ingest` — loaders, a 400-token chunker with 60 tokens of overlap, and hash-based re-ingest.
- Chunk ids are `{doc_slug}#{ordinal}` and stay put while the file bytes and the chunker version stay put.

Model ids live in configuration (`GENERATOR_MODEL`, `JUDGE_MODEL`). The judge id must differ from the generator, and the judge temperature must be 0. The first Gemini call (Phase 2) checks both ids with `client.models.list()` before use.

## Not in this phase

Hybrid search, reranking, RBAC enforcement, the agent loop, failure handling, observability, and cost tracking. Exact response caching comes before any semantic cache or model routing, and only after Phases 1 through 8.
