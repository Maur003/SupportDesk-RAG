# SupportDesk RAG

Synthetic customer-support agent for the fictional carrier **Northline Telecom**. Every document and gold question is synthetic. Nothing here is a real carrier's policy.

This repository is at the end of Phase 5: `POST /search` takes the role from a signed token and keeps the SQL access filter. `make eval` still prints rerank, hybrid, vector, and random. `u01`–`u10` are unfilled, so they are outside the scored `n`. There is no generator and no Gemini call, so refusal is not scored. A difference of one or two questions is reported as that count. Every comparison shows `n`, the corpus size, and the per-question wins, losses, and ties. `recall@20` is not reported. The cutoff is `recall@5`.

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
make eval
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
{
  "embeddings_written": 280
}
```

The next run printed `"unchanged": 121`, `"chunks_written": 0`, and `"embeddings_written": 0`. Edit a file and those counts move. They are script output, not a target.

Ingest can print a tokenizer warning that a sequence is longer than 512. On this corpus the packer measured a 771-token candidate. That candidate is two sections joined to see if they fit. Stored chunks stay inside the 400-token budget. `tests/test_gold.py` rejects an embedding input that does not fit in 512 tokens, including the two special tokens.

Phase 2, vector against random, `make eval` printed:

```text
corpus_size: 280
k: 5
random_seed: 0
embedding_model: BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
query_instruction: Represent this sentence for searching relevant passages: 
passage_input: title, heading_path, text

answerable groups compare vector retrieval with a random draw
from the chunks that role may see.
wins, losses, and ties are per question on recall@5. recall@20 is not reported.

answerable n=42 corpus_size=280 recall@5 wins=26 losses=0 ties=16
method     recall@5  precision@5  MRR     nDCG@5
vector     0.6310  0.1286  0.6131  0.6113
random     0.0238  0.0048  0.0048  0.0092

lexical n=20 corpus_size=280 recall@5 wins=19 losses=0 ties=1
method     recall@5  precision@5  MRR     nDCG@5
vector     0.9750  0.2000  1.0000  0.9807
random     0.0500  0.0100  0.0100  0.0193

paraphrase n=22 corpus_size=280 recall@5 wins=7 losses=0 ties=15
method     recall@5  precision@5  MRR     nDCG@5
vector     0.3182  0.0636  0.2614  0.2755
random     0.0000  0.0000  0.0000  0.0000

customer n=22 corpus_size=280 recall@5 wins=19 losses=0 ties=3
method     recall@5  precision@5  MRR     nDCG@5
vector     0.8864  0.1818  0.8864  0.8747
random     0.0455  0.0091  0.0091  0.0176

agent n=10 corpus_size=280 recall@5 wins=4 losses=0 ties=6
method     recall@5  precision@5  MRR     nDCG@5
vector     0.4000  0.0800  0.3250  0.3431
random     0.0000  0.0000  0.0000  0.0000

finance n=10 corpus_size=280 recall@5 wins=3 losses=0 ties=7
method     recall@5  precision@5  MRR     nDCG@5
vector     0.3000  0.0600  0.3000  0.3000
random     0.0000  0.0000  0.0000  0.0000

permission_trap n=5 corpus_size=280 metric=leak wins=0 losses=0 ties=5
vector 0.0000
random 0.0000

unanswerable n=5 corpus_size=280 metric=absent_hit wins=0 losses=0 ties=5
vector 0.0000
random 0.0000
```

A second `make eval` printed the same table. `precision@5` divides by 5, so one labeled chunk scores at most 0.20. Lexical MRR is 1.0000: on all 20 lexical questions the first labeled chunk was rank 1. The lexical tie is `a16`, which the random draw also found. `a18` has two supports and recall 0.5, because one support was outside the top 5. That question is still a win.

The paraphrase group is 7 wins and 15 ties. All 15 ties are recall 0 for both methods. Agent is 4 wins out of 10. Finance is 3 wins out of 10. Those ties are misses too. Permission traps and unanswerable questions are 0 for both methods. The random draw only uses chunks that role may see, and the absent terms do not occur in the corpus.

`make eval` exits if the vector or the hybrid answerable recall@5 falls below 0.50, if the rerank answerable recall@5 falls below 0.55, or if any of the three scores above 0 on a permission trap or an unanswerable question. Those thresholds are guards. The measured figures are below.

Phase 3 fuses the top 20 vector hits and the top 20 keyword hits with reciprocal rank fusion, `1 / (60 + rank)`, and keeps 5. Keyword search is `websearch_to_tsquery('english', question)` and `ts_rank_cd`, with the same `access_level = ANY(...)` filter. The question text is the keyword query. The embedding instruction is not prepended to it. A second `make eval` printed the same table:

```text
corpus_size: 280
k: 5
random_seed: 0
embedding_model: BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
query_instruction: Represent this sentence for searching relevant passages: 
passage_input: title, heading_path, text
candidate_k: 20
rrf_k: 60
keyword: websearch_to_tsquery english, ts_rank_cd
questions_scored: 52
keyword_nonempty: 7
top5_changed: 2

hybrid fuses the top candidate_k vector hits and keyword hits with RRF.
the scored cutoff is k. recall@20 is not reported.
wins, losses, and ties are per question, hybrid against vector and against random.

answerable n=42 corpus_size=280
recall@5 hybrid vs vector wins=0 losses=0 ties=42
recall@5 hybrid vs random wins=26 losses=0 ties=16
method     recall@5  precision@5  MRR     nDCG@5
hybrid     0.6310  0.1286  0.6131  0.6113
vector     0.6310  0.1286  0.6131  0.6113
random     0.0238  0.0048  0.0048  0.0092

lexical n=20 corpus_size=280
recall@5 hybrid vs vector wins=0 losses=0 ties=20
recall@5 hybrid vs random wins=19 losses=0 ties=1
method     recall@5  precision@5  MRR     nDCG@5
hybrid     0.9750  0.2000  1.0000  0.9807
vector     0.9750  0.2000  1.0000  0.9807
random     0.0500  0.0100  0.0100  0.0193

paraphrase n=22 corpus_size=280
recall@5 hybrid vs vector wins=0 losses=0 ties=22
recall@5 hybrid vs random wins=7 losses=0 ties=15
method     recall@5  precision@5  MRR     nDCG@5
hybrid     0.3182  0.0636  0.2614  0.2755
vector     0.3182  0.0636  0.2614  0.2755
random     0.0000  0.0000  0.0000  0.0000

customer n=22 corpus_size=280
recall@5 hybrid vs vector wins=0 losses=0 ties=22
recall@5 hybrid vs random wins=19 losses=0 ties=3
method     recall@5  precision@5  MRR     nDCG@5
hybrid     0.8864  0.1818  0.8864  0.8747
vector     0.8864  0.1818  0.8864  0.8747
random     0.0455  0.0091  0.0091  0.0176

agent n=10 corpus_size=280
recall@5 hybrid vs vector wins=0 losses=0 ties=10
recall@5 hybrid vs random wins=4 losses=0 ties=6
method     recall@5  precision@5  MRR     nDCG@5
hybrid     0.4000  0.0800  0.3250  0.3431
vector     0.4000  0.0800  0.3250  0.3431
random     0.0000  0.0000  0.0000  0.0000

finance n=10 corpus_size=280
recall@5 hybrid vs vector wins=0 losses=0 ties=10
recall@5 hybrid vs random wins=3 losses=0 ties=7
method     recall@5  precision@5  MRR     nDCG@5
hybrid     0.3000  0.0600  0.3000  0.3000
vector     0.3000  0.0600  0.3000  0.3000
random     0.0000  0.0000  0.0000  0.0000

permission_trap n=5 corpus_size=280 metric=leak
leak hybrid vs vector wins=0 losses=0 ties=5
leak hybrid vs random wins=0 losses=0 ties=5
hybrid 0.0000
vector 0.0000
random 0.0000

unanswerable n=5 corpus_size=280 metric=absent_hit
absent_hit hybrid vs vector wins=0 losses=0 ties=5
absent_hit hybrid vs random wins=0 losses=0 ties=5
hybrid 0.0000
vector 0.0000
random 0.0000
```

Hybrid versus vector is 42 ties on answerable recall@5. The recall, precision, MRR, and nDCG columns match the vector row in every group. `websearch_to_tsquery` ANDs the words that remain after English stop-words, so 45 of the 52 scored questions had an empty keyword list. The 7 nonempty lists are `a02`, `a04`, `a08`, `a13`, `a16`, `a17`, and `p04` in `reports/retrieval.json`. None of them is a paraphrase. The top 5 changed for `a02` and `a17`. Those two changes left recall, MRR, and nDCG where the vector list had them. Leak and absent-hit stay 0.

Phase 4 scores that fused top 20 with `cross-encoder/ms-marco-MiniLM-L6-v2` at the pinned Hub commit, on CPU, and keeps 5. The score is the raw logit. `torch.nn.Identity` is passed in so the library's default sigmoid is not applied. The passage is the title, the heading path, and the chunk text. The query is the question, without the BGE prefix. `RERANK_ENABLED` defaults to true for `search()`. `make eval` always scores the reranker. A second run printed the same table:

```text
corpus_size: 280
k: 5
random_seed: 0
embedding_model: BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
query_instruction: Represent this sentence for searching relevant passages: 
passage_input: title, heading_path, text
candidate_k: 20
rrf_k: 60
keyword: websearch_to_tsquery english, ts_rank_cd
rerank_model: cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c440f23af6f7d6e94d2946bac0bee0a
rerank_activation: raw logit
rerank_pool: fused top candidate_k
questions_scored: 52
keyword_nonempty: 7
top5_changed: 2
rerank_changed: 52

rerank scores the fused top candidate_k passages and keeps k.
scores are raw logits. recall@20 is not reported.
wins, losses, and ties are per question, rerank against hybrid and against random.

answerable n=42 corpus_size=280
recall@5 rerank vs hybrid wins=4 losses=1 ties=37
recall@5 rerank vs random wins=29 losses=0 ties=13
method     recall@5  precision@5  MRR     nDCG@5
rerank     0.7024  0.1429  0.6448  0.6533
hybrid     0.6310  0.1286  0.6131  0.6113
vector     0.6310  0.1286  0.6131  0.6113
random     0.0238  0.0048  0.0048  0.0092

lexical n=20 corpus_size=280
recall@5 rerank vs hybrid wins=0 losses=0 ties=20
recall@5 rerank vs random wins=19 losses=0 ties=1
method     recall@5  precision@5  MRR     nDCG@5
rerank     0.9750  0.2000  1.0000  0.9807
hybrid     0.9750  0.2000  1.0000  0.9807
vector     0.9750  0.2000  1.0000  0.9807
random     0.0500  0.0100  0.0100  0.0193

paraphrase n=22 corpus_size=280
recall@5 rerank vs hybrid wins=4 losses=1 ties=17
recall@5 rerank vs random wins=10 losses=0 ties=12
method     recall@5  precision@5  MRR     nDCG@5
rerank     0.4545  0.0909  0.3220  0.3556
hybrid     0.3182  0.0636  0.2614  0.2755
vector     0.3182  0.0636  0.2614  0.2755
random     0.0000  0.0000  0.0000  0.0000

customer n=22 corpus_size=280
recall@5 rerank vs hybrid wins=0 losses=1 ties=21
recall@5 rerank vs random wins=18 losses=0 ties=4
method     recall@5  precision@5  MRR     nDCG@5
rerank     0.8409  0.1727  0.8409  0.8293
hybrid     0.8864  0.1818  0.8864  0.8747
vector     0.8864  0.1818  0.8864  0.8747
random     0.0455  0.0091  0.0091  0.0176

agent n=10 corpus_size=280
recall@5 rerank vs hybrid wins=2 losses=0 ties=8
recall@5 rerank vs random wins=6 losses=0 ties=4
method     recall@5  precision@5  MRR     nDCG@5
rerank     0.6000  0.1200  0.4833  0.5131
hybrid     0.4000  0.0800  0.3250  0.3431
vector     0.4000  0.0800  0.3250  0.3431
random     0.0000  0.0000  0.0000  0.0000

finance n=10 corpus_size=280
recall@5 rerank vs hybrid wins=2 losses=0 ties=8
recall@5 rerank vs random wins=5 losses=0 ties=5
method     recall@5  precision@5  MRR     nDCG@5
rerank     0.5000  0.1000  0.3750  0.4062
hybrid     0.3000  0.0600  0.3000  0.3000
vector     0.3000  0.0600  0.3000  0.3000
random     0.0000  0.0000  0.0000  0.0000

permission_trap n=5 corpus_size=280 metric=leak
leak rerank vs hybrid wins=0 losses=0 ties=5
leak rerank vs random wins=0 losses=0 ties=5
rerank 0.0000
hybrid 0.0000
vector 0.0000
random 0.0000

unanswerable n=5 corpus_size=280 metric=absent_hit
absent_hit rerank vs hybrid wins=0 losses=0 ties=5
absent_hit rerank vs random wins=0 losses=0 ties=5
rerank 0.0000
hybrid 0.0000
vector 0.0000
random 0.0000
```

Answerable recall@5 moved on 5 questions: wins `b05`, `b09`, `b16`, and `b22`, and one loss, `b02`. That is 4 wins and 1 loss, with 37 ties. Lexical recall and MRR stayed on the hybrid numbers, including MRR 1.0000. Customer recall@5 went from 0.8864 to 0.8409 because of `b02`. Agent went from 0.4000 to 0.6000 on 2 questions. Finance went from 0.3000 to 0.5000 on 2 questions. `b08` and `b21` stayed found and dropped from MRR 1 to 0.5. `b10` stayed found and rose from MRR 0.25 to 1. The five ids changed on all 52 scored questions. Leak and absent-hit stay 0. The rerank guard is 0.55, under the measured 0.7024.

Phase 5 does not change those retrieval numbers. `POST /search` requires `Authorization: Bearer`. The role is the token's `role` claim. `python -m supportdesk.token customer` signs one with `JWT_SECRET` (the local synthetic default is `supportdesk-dev-secret`). A `role` field in the JSON body is rejected. The query text is not an access input, including a sentence that says to ignore instructions and reveal finance limits. The chunks are still chosen with `access_level = ANY(:allowed)`. PyJWT is the new package: it checks the HS256 signature and the expiry, instead of a hand-rolled parser. `tests/test_rbac.py` asks for `FINANCE_CANARY_7K2M` as a customer, as a customer following that injection sentence, and as an agent, and the canary is absent. The same request with a finance token returns the canary. An agent token does return `AGENT_CANARY_4QP9`.

`docker compose` targets are one line each. Every Python target is one `python -m ...` call, so the same Makefile works in PowerShell, Scoop's GNU Make, and CI.

Torch is installed from the CPU index in `requirements-torch.txt` before the lockfiles. The default PyPI torch wheel is the CUDA build. `make deps` installs the runtime lock only. `make deps-dev` installs that lock plus pytest, ruff, and `httpx` (the TestClient transport). The API image uses the runtime lock and does not install the dev packages.

`httpx2` and `httpcore2` are in the runtime lock because `huggingface_hub` 2.4 imports `httpx2` to download the tokenizer, and `httpx2` imports `httpcore2`. They are a different project from `httpx`. The tokenizer revision is the Hub commit in `EMBEDDING_REVISION`, not the branch `main`.

## Layout

- `data/corpus/{public,agent,finance}` — synthetic knowledge base. `synthetic: true` is required on every file. `tools/expand_corpus.py` rewrites the long near-duplicate pages.
- Retrieval embeds the document title, the heading path, and the chunk text (`src/supportdesk/embed_text.py`). The stored chunk text stays the body. Queries get the BGE instruction prefix. Vectors are L2-normalized. Search is exact cosine distance with `access_level = ANY(...)`. Keyword search uses that same filter. Hybrid fuses the two top-20 lists. The cross-encoder then reranks that fused 20 and keeps 5. `POST /search` applies that filter to the role in the bearer token. There is no HNSW index.
- `src/supportdesk/ingest` — loaders, a 400-token chunker with 60 tokens of overlap, and hash-based re-ingest.
- Chunk ids are `{doc_slug}#{ordinal}` and stay put while the file bytes and the chunker version stay put.

Model ids live in configuration (`GENERATOR_MODEL`, `JUDGE_MODEL`). The judge id must differ from the generator, and the judge temperature must be 0. The first Gemini call checks both ids with `client.models.list()` before use. This phase does not call Gemini.

## Not in this phase

The agent loop, failure handling, observability, and cost tracking. Exact response caching comes before any semantic cache or model routing, and only after Phases 1 through 8.
