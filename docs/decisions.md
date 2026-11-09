# Decisions

Measured impact is recorded only when a script in this repo printed the number. Until that command exists, the impact line says so.

## Phase 1

### Makefile recipes are one `python -m` line

Scoop GNU Make 4.4.1 on this machine is the Windows build. Recipes do not use shell continuations, `&&`, or bash. Docker targets are the exception: `docker compose up -d --build` and `docker compose down` are one line and have no Python entry point. Trade-off: the file is obvious in CI and in PowerShell, and it cannot express a multi-step recipe without another target. Measured impact: not a retrieval metric.

### Corpus is fictional and labeled

The carrier is Northline Telecom. Loaders reject a file unless `synthetic: true` and `company: Northline Telecom`. The directory name must match `access_level`. Trade-off: a real document cannot be dropped in without editing the loader. That is intentional for this portfolio. Measured impact: not a retrieval metric.

### Gold-set provenance

Ten of the thirty answerable questions are reserved for the author. The rest, plus the five unanswerable questions and five permission traps, are LLM-drafted. `tests/test_gold.py` spot-checks that each drafted fact is contained in the labeled chunk. Trade-off: a containment check catches mislabeled ids and does not measure answer quality. Measured impact: not a retrieval metric.

### How later tables will be reported

Comparison tables will include `n` and per-question wins, losses, and ties. A gap of one or two questions is reported as that count and is not called an improvement. The judge model is a different id from the generator, at temperature 0. Both ids are configuration. The first Gemini call verifies both ids with `client.models.list()` before use. Phase 2 does not call Gemini. Measured impact: not measured yet.

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

`WRITEOFF_CANARY_8HN4` is only in `finance/write-off-policy.md`. `EXECUTIVE_CANARY_2VX6` is only in `finance/executive-credits.md`. `p04` and `p05` use those strings the same way `p01` and `p03` use the older canaries. A test counts each canary in exactly one file. Trade-off: the trap is "was the hidden string retrieved," not a graded essay. Measured impact: `permission_trap` n=5, vector leak 0.0000, random leak 0.0000, 5 ties. The random pool is already limited to chunks that role may see, so the tie means the SQL filter also stayed off the forbidden chunk.

### How results tables will be reported

Every results table includes the corpus size, meaning the chunk count in the manifest, and a random-retrieval row. That row draws `k` chunks uniformly from the chunks the role may see, with a fixed seed, and scores them with the same metric. `recall@20` is not a headline number. The cutoff that matches the context window is `recall@5`. A wider cutoff is shown only next to the corpus size, because 20 out of 57 and 20 out of a few hundred are different claims. Wins, losses, ties, and `n` still sit beside any percentage. Measured impact: the first `make eval` printout is in the README. The random row sorts chunk ids before the draw, so seed 0 does not follow a Postgres scan.

### Pinned tokenizer, caches, and the lock split

`EMBEDDING_REVISION` is the Hub commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, the snapshot already in the local tokenizer cache. Settings reject a branch name such as `main`. Chunk ids move if this commit changes, so the manifest records the revision and CI caches `~/.cache/huggingface` with a key that includes `src/supportdesk/config.py`. `actions/setup-python` caches pip from `requirements-torch.txt`, `requirements.txt`, and `requirements-dev.txt`.

`requirements.txt` is the runtime lock. The image installs that file and does not install pytest or ruff. At this phase `requirements-dev.txt` included the runtime lock plus pytest, ruff, and `httpx==0.28.1` (the FastAPI TestClient transport). Phase 6 moves that `httpx` pin into the runtime lock.

`httpx2==2.16.0` and `httpcore2==2.16.0` stay in the runtime lock. `pip show` reports that `huggingface_hub` 2.4.0 requires `httpx2`, and `httpx2` requires `httpcore2`. They are a different project from `httpx`. Removing them breaks the tokenizer download. They are not an extra HTTP client we chose.

### Phase 2 baseline retrieval

Passages are `embedding_input(title, heading_path, text)`: the document title, the heading path, and the chunk text, joined by newlines. Empty parts are skipped. `chunks.text` stays the body. Citations and the gold-fact check use that column. The query is prefixed with `Represent this sentence for searching relevant passages: `. Both encode calls pass `prompt=""` so a saved sentence-transformers prompt cannot add a second prefix. Vectors are L2-normalized. Distance is pgvector cosine (`<=>`). The access filter is SQL `access_level = ANY(:allowed)`: customer sees public, agent sees public and agent, finance sees all three. There is no HNSW index. Search is an exact order by distance over 280 embedded chunks.

The document title is a real column on `chunks`. A generated tsvector cannot join to `documents`, and keyword search in Phase 3 needs the same fields as the vector. The column is in the chunk hash and in `to_tsvector('english', title || heading_path || text)`. Putting the title into the hash made one re-ingest of the already loaded corpus update all 121 documents and write 280 embeddings. An empty database then printed `inserted` 121, `chunks_written` 280, and `embeddings_written` 280. The next run printed `unchanged` 121 and `embeddings_written` 0.

The random baseline draws k=5 ids with seed 0 from the chunks that role may see, after sorting those ids. Without the sort, truncating and reloading the same chunks changed the random row, because `random.sample` follows list order and the scan order changed. Two `make eval` runs after the sort printed the same table.

`u01`–`u10` are excluded because they are unfilled. Unanswerable questions are not in the recall. There is no generator, so refusal is not scored. A permission leak is 1 when a retrieved id is a labeled support or a labeled fact appears in a retrieved chunk. An absent-hit is 1 when an absent term appears in a retrieved chunk. `precision@5` divides by 5, so one labeled chunk scores at most 0.20.

`make eval` exits if answerable recall@5 is below 0.50, or if the permission or unanswerable vector rate is above 0. That 0.50 is a guard under the measured 0.6310. The README shows the measured table, not the guard.

No new package. sentence-transformers was already installed for the tokenizer. google-genai is not installed.

Trade-off: the vector is not the citation string, and a sequential scan will not show index tuning. The paraphrase and role splits are the reason to keep this baseline before hybrid search. Measured impact, from `python -m supportdesk.eval`, corpus 280, k=5, seed 0, confirmed by a second run: answerable n=42 recall@5 0.6310 versus random 0.0238, wins 26, losses 0, ties 16. Lexical n=20 recall@5 0.9750, MRR 1.0000, wins 19, losses 0, ties 1. The tie is `a16`, which the random draw also found. `a18` recall is 0.5 because one of its two supports was outside the top 5, and that question is a win. Paraphrase n=22 recall@5 0.3182, wins 7, losses 0, ties 15, and all 15 ties are recall 0 for both methods. Customer n=22 recall@5 0.8864, wins 19, losses 0, ties 3. Agent n=10 recall@5 0.4000, wins 4, losses 0, ties 6. Finance n=10 recall@5 0.3000, wins 3, losses 0, ties 7. Those agent and finance ties are misses. Permission leak and unanswerable absent-hit are 0.0000 for both methods, 5 ties each. Seven of 22 paraphrases, and 3 of 10 finance questions, are the counts. They are not a solved retrieval problem.

## Phase 3

### Hybrid search is reciprocal rank fusion of two top-20 lists

Keyword search runs `websearch_to_tsquery('english', question)` and ranks with `ts_rank_cd`. The question text is the keyword query. The BGE instruction prefix stays on the embedding query only. The access predicate is the same SQL `access_level = ANY(:allowed)` as the vector query, and both lists skip chunks whose embedding is null. Each list keeps 20. Reciprocal rank fusion scores a chunk as the sum of `1 / (60 + rank)` over the lists it appears in. Rank 1 is the best row. Equal scores break on chunk id. The scored cutoff is the first 5. Phase 4 can rerank a longer fused list; this phase does not.

`websearch_to_tsquery` ANDs the words left after English stop-words. A paraphrase that shares no content word with its chunk cannot match. A long lexical question also misses when any remaining word is absent from the chunk. Trade-off: the operator is the one Postgres provides for a typed question, and it leaves most of this gold set with an empty keyword list. Switching to OR after seeing that result would be a different experiment. No new package.

Measured impact, from `python -m supportdesk.eval`, confirmed by a second run: corpus 280, k=5, candidate_k=20, rrf_k=60, 52 scored questions. `keyword_nonempty` is 7. `top5_changed` is 2. Answerable hybrid versus vector is wins 0, losses 0, ties 42. Hybrid recall@5, precision@5, MRR, and nDCG@5 match the vector row in every group, including answerable recall@5 0.6310 versus random 0.0238 (hybrid versus random wins 26, losses 0, ties 16). The per-question report names the seven keyword hits as `a02`, `a04`, `a08`, `a13`, `a16`, `a17`, and `p04`. The two changed top-5 lists are `a02` and `a17`. Their recall, MRR, and nDCG stayed with the vector list. Permission leak and unanswerable absent-hit stay 0.0000 for hybrid, vector, and random, 5 ties each. The guard is still 0.50 under the measured 0.6310, now checked on both the vector row and the hybrid row. Zero wins against the vector list is the result.

## Phase 4

### Rerank the fused top 20 with raw cross-encoder logits

`cross-encoder/ms-marco-MiniLM-L6-v2` is the Hub id. `model_info` returns commit `233902d25c440f23af6f7d6e94d2946bac0bee0a`. The hyphenated `L-6` name resolves to this same repo. The model runs on CPU. `CrossEncoder` is already in the installed `sentence-transformers` package, so this phase adds no dependency. Its default activation for one label is a sigmoid. Construction and `predict` both pass `torch.nn.Identity`, and startup checks that the sigmoid of the identity score matches a sigmoid call. The stored score is the logit.

The rerank pool is the fused top 20, which is already limited by `access_level = ANY(:allowed)`. The reranker does not see other chunks. The passage is `embedding_input(title, heading_path, text)`. The query is the question, without the BGE prefix. Higher logits rank first. Equal logits break on chunk id. The cutoff stays 5. `RERANK_ENABLED` defaults to true and switches `search()` between that cutoff and the RRF cutoff. `make eval` always scores the reranker, so the table does not depend on the flag.

Trade-off: the cross-encoder can promote a near-duplicate that the bi-encoder had ranked just outside 5, and it can also demote a hit the bi-encoder had at rank 1. The lexical questions were already at MRR 1.0000, so reordering them cannot raise recall. No new package.

Measured impact, from `python -m supportdesk.eval`, confirmed by a second run: corpus 280, k=5, candidate_k=20, 52 scored questions, `rerank_changed` 52. Answerable rerank versus hybrid is wins 4, losses 1, ties 37. Recall@5 is 0.7024 versus hybrid 0.6310 and random 0.0238 (versus random: wins 29, losses 0, ties 13). The four wins are `b05`, `b09`, `b16`, and `b22`. The loss is `b02`. Lexical n=20 is wins 0, losses 0, ties 20, and lexical MRR stays 1.0000. Customer n=22 is wins 0, losses 1, ties 21, recall@5 0.8409 versus hybrid 0.8864. Agent n=10 is wins 2, losses 0, ties 8, recall@5 0.6000 versus 0.4000. Finance n=10 is wins 2, losses 0, ties 8, recall@5 0.5000 versus 0.3000. `b08` and `b21` stayed inside the top 5 and fell from MRR 1 to 0.5. `b10` stayed inside and rose from MRR 0.25 to 1. Permission leak and unanswerable absent-hit stay 0.0000. The rerank guard is 0.55, under the measured 0.7024. Four wins and one loss are the count.

## Phase 5

### The role comes from the token, and the filter stays in SQL

`POST /search` reads `Authorization: Bearer`. `read_role` accepts only HS256, checks the expiry, and accepts only `customer`, `agent`, and `finance`. PyJWT does that check. It is the one new package: a hand-rolled header parser would be the place an `alg: none` token gets through. There is no login route. `python -m supportdesk.token <role>` signs a token with `JWT_SECRET`. The default secret is a local synthetic string, `supportdesk-dev-secret`.

The JSON body is the query and nothing else. A `role` field is a 422. The query text is passed to retrieval and is not consulted for access, so an instruction to ignore previous rules and reveal finance limits stays a customer query when the token says customer. `allowed_levels` still maps customer to public, agent to public and agent, and finance to all three. `search()` already applies `access_level = ANY(:allowed)` before fusion and before the reranker. This phase does not change the ranker, so the Phase 4 table is not re-measured.

Trade-off: anyone who has the signing secret can mint any role. That is the stand-in for an identity provider, not a password check on each request. Measured impact: `tests/test_rbac.py` posts `FINANCE_CANARY_7K2M` with a customer token, with a customer token and the injection sentence, and with an agent token. The canary is absent from those responses, and every returned `access_level` is inside that role's set. The same query with a finance token returns the canary. An agent token returns `AGENT_CANARY_4QP9`. A missing token, a bad token, an expired token, a token whose claim is `root`, and an `alg: none` token are rejected. Retrieval recall was not run again.

## Phase 6

### The model can request a ticket. Only the user confirmation writes it

The loop is manual. `run_agent` calls `generate`, executes one tool, appends the tool result, and stops after five model calls. It does not use the SDK's automatic function-calling loop. `GeminiLLM` passes `AutomaticFunctionCallingConfig(disable=True)`. The real client calls `models.list()` once before the first `generate_content` and raises if `GENERATOR_MODEL` or `JUDGE_MODEL` is absent. Tests pass a fake client. This phase did not call Gemini.

`search_knowledge_base` calls the existing `search()` with `allowed_levels(role)`. The role still comes from the bearer token. `get_ticket_status` calls the mock CRM over HTTP. The support process does not import the CRM's ticket dict. `calculate` walks a restricted AST: numbers, parentheses, unary plus and minus, and `+ - * /`. Names, calls, and other operators raise. `eval` is not called.

`create_ticket` stores a pending confirmation and returns `confirmation_required`. The arguments may include `confirmed: true`. That field is not read. `POST /confirmations/{id}` is the write, and only when the bearer role matches the role that requested the ticket. A missing confirmation is 404. The other role is 403. If the CRM write fails, the confirmation stays so the user can retry. There is no tool the model can call to confirm itself.

The CRM is a second FastAPI app in Compose, the same image, command `python -m uvicorn supportdesk.crm_server:app --host 0.0.0.0 --port 8001`. It requires `X-API-Key`. `TCK-1001` is seeded in that process. The client retries only 429 and 5xx, with delay `backoff * 2^attempt * uniform(0.5, 1.0)`. A timeout and any other 4xx are not retried. Tickets and pending confirmations are in memory. A CRM restart drops created tickets and brings the seed back. A support-API restart drops pending confirmations.

`httpx==0.28.1` moved into `requirements.txt` because the CRM client imports it. `google-genai==3.0.0` is the new package: it is the agreed generator client. The lock also pins what that install pulled in, including `google-auth`, `requests`, and `cryptography`. `httpx2` stays for `huggingface_hub`.

Trade-off: the confirmation store does not survive a restart, and two confirms racing the same id can both pass the lookup before either delete. The write is still not something the model can do inside the loop. Retrieval was not re-run. The Phase 4 table is unchanged.

Measured impact, from the tests, not from a live Gemini call: a customer `search_knowledge_base` for `FINANCE_CANARY_7K2M` leaves that string out of the tool result; the finance role's tool result contains it. `create_ticket` with `confirmed: true` leaves the CRM at the seed ticket `TCK-1001`. The matching role's `POST /confirmations/{id}` creates `TCK-1002`. The other role gets 403. A second confirm gets 404. A tool named `confirm_ticket` returns `unknown tool` and does not write. Five `calculate` calls in a row return `step_limit` and do not make a sixth model call. `2 + 3 * 4` returns 14. `__import__('os')` returns `invalid expression`. `get_ticket_status` for `TCK-1001` reads `status: open` over HTTP. A scripted 500, then 429, then 200 sleeps twice, `0.1` then `0.2` when the backoff is `0.2` and the jitter draw is `0`. A 400 and a timeout do not sleep. A wrong API key is 401. Three CRM 500s on confirm return 502 and leave the confirmation in place for another attempt. The fake Gemini client records one `list()` call across two generates, and `disable` is true. A list that omits `gemini-2.5-flash` raises before `generate_content`. `POST /ask` without `GEMINI_API_KEY` is 503. A `role` field on that body is 422.
