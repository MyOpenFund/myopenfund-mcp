# myopenfund-mcp

One [MCP](https://modelcontextprotocol.io) server over the MyOpenFund bricks: official documents and
pipeline health, cited passage search over the embedded corpus, and company data.

Any MCP client (a desktop assistant, a coding agent, your own agent, a local model) gets outcome-oriented
tools over MyOpenFund's data: find official primary-source documents, read their facts with provenance
(`doc_id`, dates, source URLs), search the text of the documents by meaning and get passages to cite, count
the corpus, check the ingestion pipeline's health. Toolsets are
enabled per deployment: a toolset whose environment variables are not set is not configured, so it is not
mounted, and `server_info` says so. A toolset whose variables are set but whose extra is not installed
stops the server at start with a clear configuration error (it never runs silently without it).

> **MyOpenFund manages nobody's money and gives no investment advice.** This server returns documents and
> facts from official sources, with their provenance. What you do with them is your decision.

- Two transports, same code: **stdio** for local use, **Streamable HTTP** (bearer tokens) in a Docker image.
- Host-agnostic: the same image runs on a laptop, on a host on a private network or on a cloud VM;
  everything deployment-specific comes from `MOF_*` environment variables.
- Deterministic: no tool needs a hosted LLM (see [Local models](#local-models)).

## Contents

- [Quick start (stdio)](#quick-start-stdio)
- [HTTP deployment](#http-deployment)
- [Deploying the rag toolset](#deploying-the-rag-toolset)
- [Tools](#tools)
- [Errors](#errors)
- [Tiers and tokens](#tiers-and-tokens)
- [Configuration and limits](#configuration-and-limits)
- [Local models](#local-models)
- [Roadmap](#roadmap)
- [Development](#development)

## Quick start (stdio)

Prerequisites:

- Python 3.12+ and git (the `[vault]` extra installs a pinned dependency from git);
- read access to the vault database as the read-only role `vault_readonly` (never a writing role).

```bash
git clone https://github.com/MyOpenFund/myopenfund-mcp.git
cd myopenfund-mcp
python -m venv .venv
.venv/bin/pip install ".[vault]"
```

Register it with your MCP client. With Claude Code:

```bash
claude mcp add myopenfund \
  -e MOF_VAULT_DATABASE_URL='postgresql://vault_readonly:<password>@<host>:5432/documents?sslmode=verify-full' \
  -- /path/to/myopenfund-mcp/.venv/bin/myopenfund-mcp serve
```

Any other client: run the command `myopenfund-mcp serve` (stdio is the default transport) with the same
environment. Over stdio there is no token: the local user owns the process and gets every tier
(`MOF_MCP_TOKENS`, if set, is ignored with a warning on stderr). Then ask the client to call `server_info`.

> The snippet above stores the database URL, **password included**, in the MCP client's configuration
> file. Use the read-only role `vault_readonly` only, and keep that file private.

## HTTP deployment

`MOF_MCP_TRANSPORT=http` serves Streamable HTTP at `MOF_MCP_PATH` (default `/mcp`). It **requires**
`MOF_MCP_TOKENS`: the HTTP server never runs unauthenticated, and refuses to start without tokens.

The server speaks plain HTTP. **Network exposure and TLS are the deployment's job**, never the public
internet by default. The repository ships a generic [`Dockerfile`](Dockerfile),
[`compose.example.yaml`](compose.example.yaml) and [`.env.example`](.env.example); nothing host-specific
lives in it.

```bash
cp compose.example.yaml compose.yaml
cp .env.example .env                                   # never committed
.venv/bin/myopenfund-mcp hash-token alice              # once per client, see Tiers and tokens
.venv/bin/myopenfund-mcp hash-token agent-1 --tier read
# edit .env: MOF_MCP_TOKENS, MOF_VAULT_DATABASE_URL, MOF_MCP_BIND_ADDR ...
docker compose up -d --build
```

No venv on the deployment host? Run `hash-token` in the container instead (after `cp .env.example .env`,
which compose needs, and with the image built):

```bash
docker compose build
docker compose run --rm mcp myopenfund-mcp hash-token alice
```

`hash-token` prints the **token on stderr** (copy it to the client) and **only the `MOF_MCP_TOKENS`
entry on stdout**, so redirecting stdout (`>> entries.txt`) never writes the clear token into a file.

The image runs as a non-root user (uid 10001), listens on `0.0.0.0:8000` **inside the container**, and
the compose file publishes it on `127.0.0.1` of the host unless `MOF_MCP_BIND_ADDR` says otherwise. In
`.env`, leave `MOF_MCP_TRANSPORT` and `MOF_MCP_HOST` unset: the image sets them. The compose file also runs it with a
read-only root filesystem (writable `/tmp` only), no Linux capabilities and `no-new-privileges`.

A client connects with its token, e.g. with Claude Code:

```bash
claude mcp add --transport http myopenfund https://<your-host>/mcp \
  --header "Authorization: Bearer <token>"
```

### Connect from any MyOpenFund repo

Each MyOpenFund code repo ships a `.mcp.json` holding variables only, never an address or a token. Clone one
repo or all of them: each works on its own, including in new worktrees, and projects outside MyOpenFund do not
see the server. Set these two client-side variables once, in your shell profile or a private file it loads:

```bash
export MOF_MCP_URL=https://<your-host>/mcp
export MOF_MCP_TOKEN=<token>   # your client token, the one from hash-token
```

`MOF_MCP_TOKEN` is the client's clear token; `MOF_MCP_TOKENS` is the server's list of digests. Claude Code asks
you to approve the server once per project. Start it from a shell that has the variables. Without them, it
warns about the missing variables and everything else keeps working.

### Pattern 1: a host on a private network

A home server, a NAS or an office machine reachable only over a VPN or a private interface.

- Set `MOF_MCP_BIND_ADDR` to the host's **private** address (the VPN or LAN interface), never `0.0.0.0`
  on a host that also has a public interface.
- If the vault Postgres runs as a compose stack on the same Docker host, do not publish its port: join its
  Docker network instead (uncomment the `networks` blocks in `compose.yaml`, set
  `MOF_VAULT_DOCKER_NETWORK`) and use the database service name as `<host>` in `MOF_VAULT_DATABASE_URL`.
  On that private Docker network, `sslmode=disable` is acceptable.
- TLS: the VPN already encrypts the traffic; for HTTPS on top, put a reverse proxy holding a certificate
  for the private name in front, or use your VPN's own HTTPS feature if it has one.

### Pattern 2: a cloud VM (or container service)

- Keep the default `MOF_MCP_BIND_ADDR=127.0.0.1` and terminate **TLS at a reverse proxy** on the same VM
  (Caddy, nginx, Traefik) proxying to `127.0.0.1:8000`, or run the container behind the provider's **load
  balancer** with a TLS listener and a security group that only admits the balancer.
- Health check for the load balancer or orchestrator: `GET /health` answers `200` with the body `ok`,
  without a token (it reveals no version and no data); `/mcp` keeps requiring one. The image's Docker
  `HEALTHCHECK` uses the same endpoint.
- Reach the vault database over the provider's private network with `sslmode=verify-full` and the
  provider's CA certificate (`sslrootcert=...`). `sslmode=require` is a fallback: it encrypts but does not
  authenticate the server.
- Restrict who can reach the HTTPS endpoint (IP allow-list, private load balancer) wherever you can: the
  bearer token is the only authentication in v1.

## Deploying the rag toolset

The `rag_search` tool needs a Qdrant collection already filled by the ingestion (rag-engine through the
data-orchestrator) and the embedding model `intfloat/multilingual-e5-base` as an ONNX file. The server only
reads: it never creates a collection, an index or a point.

1. **A read-only Qdrant API key.** On the Qdrant stack, set a read-only key next to the main one
   (`QDRANT__SERVICE__READ_ONLY_API_KEY`) and give the MCP that key only, as `MOF_RAG_QDRANT_API_KEY`.
2. **Payload indexes, once, from the data side.** The filters and the list of valid filter values use
   Qdrant's payload indexes. Whoever owns the collection creates them with a key that can write, once per
   collection (running it again does nothing):

   ```bash
   QDRANT_HOST=<qdrant-host> QDRANT_API_KEY=<key-that-can-write> python -c "from rag_engine.store import QdrantStore; print(QdrantStore().ensure_payload_indexes('<collection>'))"
   ```

   The MCP never does this (its key cannot). Without the indexes `rag_search` still answers, more slowly:
   filters read every point, and the valid values are read by scrolling the collection (up to 100,000
   points; beyond that, filter values are no longer checked).
3. **The image with the extra.** Build with `MOF_EXTRAS=vault,rag` (see `compose.example.yaml`; from a
   checkout without Docker, `pip install ".[vault,rag]"`). No PyTorch is installed: queries are embedded on
   the ONNX runtime, on CPU.
4. **Fill the model cache, once.** The model is about 1.1 GB. It is pinned to one revision and every file is
   hash-checked at each start, so a changed or damaged file is refused. Download it into the volume with
   this one-off command, on a host that can reach `huggingface.co`:

   ```bash
   docker run --rm -v myopenfund-mcp-rag-models:/models myopenfund-mcp python -c "from rag_engine.config import Settings; from rag_engine.embedding import make_embedder; make_embedder(Settings(backend='onnx', model_cache='/models')).release()"
   ```

   Without Docker, run the same `python -c "..."` in the server's venv with your own directory in place of
   `/models`. A filled directory can also be copied as it is from another host. Run again on a filled
   cache, the command downloads nothing and only verifies the files.
5. **Set the variables** in `.env`: `MOF_RAG_QDRANT_HOST`, `MOF_RAG_QDRANT_PORT`, `MOF_RAG_QDRANT_API_KEY`,
   `MOF_RAG_COLLECTION`, `MOF_RAG_MODEL_CACHE=/models` (and `MOF_RAG_QDRANT_HTTPS=true` when Qdrant is
   behind TLS). In `compose.yaml`, uncomment the rag blocks: the volume **read-only**, and at least 2 GB of
   memory for the container (the model takes about 1.9 GB while it loads, about 1.5 GB afterwards). With a
   CPU limit, set `MOF_RAG_ONNX_THREADS` to it.
6. **Restart**, then call `server_info`: `toolsets.rag` is `enabled` and `rag.status` is `ready` (the model
   loads at start, in about 5 to 7 seconds on 4 cores with a warm disk; a first read of the file from a
   cold disk takes longer). Then call `rag_search` with a short query.

Once the cache is filled the server makes no outbound connection for the model. If `rag.status` is
`unavailable`, `rag.reason` and the server log say why (see [Passage search](#passage-search-rag)).

## Tools

Tool names carry their toolset prefix. All v1 tools are read-only (`readOnlyHint`). Only
`vault_search_documents` is paged: it returns `next_cursor` to pass back unchanged with the same filters.
`vault_corpus_stats` and `vault_runs_status` only bound their lists with `limit` (no cursor). Every
`limit` defaults to `MOF_MCP_DEFAULT_ROWS`; a value outside `[1, MOF_MCP_MAX_ROWS]` is refused, not
clamped. String arguments `doc_id`, `text`
and `cursor` are at most 200 characters.

| Tool | Purpose | Key arguments |
|---|---|---|
| `server_info` | What this deployment offers: enabled toolsets, your tier, version, data freshness (last vault ingest). Call it first. Open to every authenticated caller. | none |
| `vault_search_documents` | Find official documents (metadata, not text). Start here: filter, then open one with `vault_get_document`. Rows are ordered by `sort_by` in `sort_dir`; equal sort values are always ordered by `doc_id` ascending, whatever `sort_dir`. `total` counts the whole filtered set. Cursors are offsets: if the vault is ingesting between two pages, a row can shift (seen twice or skipped). | `corpus` (e.g. `central-bank`), `source_code` (publisher, e.g. `us` for the Fed), `doc_type`, `language`, `year`, `mime_type`, `date_from` / `date_to` (YYYY-MM-DD, inclusive), `text` (case-insensitive substring of the title, matched literally: `%` and `_` are ordinary characters), `sort_by` (`date` default, `year`, `corpus`, `source_code`, `doc_type`, `title`, `created_at`), `sort_dir` (`desc` default), `limit`, `cursor`, `include_deleted` (also return withdrawn documents, with `deleted_at` set) |
| `vault_get_document` | Every stored fact of one document: dates, source and PDF URLs, publisher, type, `extra`. | `doc_id` |
| `vault_corpus_stats` | Live-document counts grouped by one field, largest first (NULL is a real group). | `group_by` (`source_code` default, `corpus`, `doc_type`, `language`, `year`, `provenance`, `mime_type`), `corpus`, `limit` |
| `vault_runs_status` | Is the pipeline healthy? Four lists, newest first: `runs` (one row per run, with outcome and totals; a run that crashed before reaching any source appears only here), `source_runs` (one row per run and source: counts, outcome, bounded error samples), `open_discovery_errors` (unresolved discovery failures; while `seen_at_is_ingestion_time` is true, `last_seen_at` is approximate), `source_health` (expected-vs-observed publication cadence per series). | `corpus`, `source_code`, `since` (ISO 8601, UTC if no offset, an explicit offset is kept: whole minutes, at most ±15:59; keeps runs *finished* at or after it, leaves out runs still in flight; applies to `runs` and `source_runs` only), `limit` (applies to each of `runs`, `source_runs`, `open_discovery_errors`) |

Resource: `vault://documents/{doc_id}`, one document's metadata as JSON (same content as
`vault_get_document`).

The `vault_*` tools and the resource exist only when `MOF_VAULT_DATABASE_URL` is set; the `[vault]` extra
must then be installed, or the server refuses to start (`MOF_VAULT_DATABASE_URL is set but the [vault]
extra is not installed`). Timestamps in results are UTC: the server's database session runs with
`timezone=UTC`, whatever the database or role default.

If the vault database is down, `server_info` still answers (version, tier, toolsets) with `freshness`
null and `"freshness_error": "backend_unavailable"`; the server log names the exception class only.

### Passage search (rag)

`rag_search` exists only when `MOF_RAG_QDRANT_HOST`, `MOF_RAG_COLLECTION` and `MOF_RAG_MODEL_CACHE` are set;
the `[rag]` extra must then be installed, or the server refuses to start (`MOF_RAG_QDRANT_HOST is set but
the [rag] extra is not installed`). It searches the text of the embedded central-bank documents by meaning
(one dense search over the chunks, with rag-engine) and returns passages, best first. It does not write an
answer: the client's model reads the passages and cites them. Read tier.

| Argument | Meaning |
|---|---|
| `query` (required) | What to look for, in plain language, in any language; at most 2000 characters (the model reads the first 512 tokens). |
| `source_codes` | List of central-bank codes, exact: `ecb`, `us` (Federal Reserve), `gb`, `au`, `ca`, `jp`. |
| `doc_types` | List of document-type codes, exact: `A1` rate-decision press release, `A2` policy statement, `A3` minutes or accounts, `B1` press conference, `E1` monetary policy report, `F1` projections. |
| `languages` | List of language codes as stored, e.g. `en`. |
| `date_from`, `date_to` | Publication date bounds, `YYYY-MM-DD`, both inclusive. As soon as one is set, a document without a full date is left out. |
| `limit` | Number of passages; default `MOF_RAG_DEFAULT_LIMIT` (8), at most `MOF_RAG_MAX_LIMIT` (20). |

The values of one list are alternatives (OR), the arguments combine (AND). An omitted argument does not
restrict; an empty list is refused. The codes above are those of the corpus today; the server reads the
valid values from the collection itself.

Example call and result (text shortened):

```json
{"query": "Why did the ECB raise interest rates in July 2022?", "source_codes": ["ecb"], "date_from": "2022-07-01", "limit": 1}
```

```json
{
  "query": "Why did the ECB raise interest rates in July 2022?",
  "filters": {"source_codes": ["ecb"], "doc_types": null, "languages": null, "date_from": "2022-07-01", "date_to": null},
  "count": 1,
  "hits": [
    {
      "rank": 1,
      "score": 0.8189,
      "text": "The Governing Council decided to raise the three key ECB interest rates by 50 basis points. ...",
      "doc_id": "ecb/A1/2022-07-21",
      "title": "Monetary policy decisions",
      "date": "2022-07-21",
      "source_code": "ecb",
      "doc_type": "A1",
      "language": "en",
      "page": 1,
      "chunk_number": 0,
      "citation": "Monetary policy decisions (ecb A1, 2022-07-21), p. 1"
    }
  ]
}
```

- **Hits.** `score` is the cosine between the query and the passage (4 decimals; higher is closer). Several
  hits may come from one document; nothing is grouped. `filters` echoes the filters as applied. The point id
  and the rest of the stored payload are not returned.
- **Pages.** `page` is the page the passage starts on **as a reader counts it: the first page is 1** (the
  collection stores pages from 0; the server adds one). It is `null` for a document without pages, and the
  `citation` then has no page part. `citation` is `<title> (<source_code> <doc_type>, <date>), p. <page>`,
  each missing part left out (the `doc_id` stands for a missing title).
- **Unknown filter values are refused, never corrected.** The server reads the distinct `source_code`,
  `doc_type` and `language` values from the collection (Qdrant's facet API on the keyword indexes; a full
  scroll when the collection has no such index) and keeps them `MOF_RAG_CODES_TTL_S` seconds. A value that
  is not among them is `[invalid_argument]` listing the valid values, with a hint for a near miss (`ECB`:
  did you mean `ecb`?), instead of an empty result. If the values cannot be read, they are not checked (one
  warning in the log) and an unknown value simply finds nothing.
- **Limits and size.** A `limit` above `MOF_RAG_MAX_LIMIT` is `[limit_exceeded]`, never lowered. A result
  larger than `MOF_MCP_MAX_RESULT_CHARS` is refused with `[result_too_large]`, never cut; the message names
  the largest `limit` that fits. Passages are 200 to 400 words: the default 8 are a few thousand tokens.
- **Start-up.** The embedding model is loaded and the collection is checked when the server starts, not at
  the first call, and a model is never loaded or downloaded by a tool call. If Qdrant cannot be reached,
  refuses the key, or does not hold the collection, **the server still starts** (as it does when the vault
  database is down): `rag_search` answers `[rag_unavailable]` with the reason, the collection is checked
  again at the next call, and the tool works as soon as the check passes. If the model cannot be loaded
  (files missing with no way to download them, or a file that is not the pinned one), the server starts
  too, logs the cause as an error, and `rag_search` answers `[rag_unavailable]` until the cache is fixed
  and the server restarted.
- **`server_info`** adds a `rag` object: `status` (`ready`, or `unavailable` with a `reason`), `collection`,
  `model`, `engine_version`, `default_limit`, `max_limit`. It reports the state recorded at start and by
  the last calls, without a request to Qdrant.
- **Privacy.** As for every tool, the audit line holds the tool name, the client, the outcome and the
  duration, never the arguments: the text of a query is not logged, at any level.
- **Concurrency.** One embedder and one Qdrant client are shared by the process; inference runs one query
  at a time (about 20 ms for a short query on 4 cores) off the event loop, so other tools keep answering.

### Companies and datasets

The `companies_*` and `datasets_*` tools (read-only, over the published my-open-data store) exist only when
`MOF_COMPANIES_CATALOG` and `MOF_COMPANIES_DATA_ROOT` are set; the `[companies]` extra must then be installed
(from a checkout, `pip install ".[vault,companies]"`, or from the git URL,
`pip install "myopenfund-mcp[companies] @ git+https://github.com/MyOpenFund/myopenfund-mcp"`; the extra pins two
git dependencies, so git must be available). The
store is read-only (as the `lake_reader` role); nothing is ever written. In a hardened container, DuckDB
downloads and loads its `ducklake` and `postgres` extensions under `$HOME/.duckdb` at each start: `HOME` must be
writable, a tmpfs holding it must allow exec (`/tmp:exec`), and the container needs outbound HTTPS to
`extensions.duckdb.org`. The process must also be able to read the store's `lake/` files (my-open-data creates
them `0700` for its own uid). Configuration:
`MOF_COMPANIES_CATALOG`, `MOF_COMPANIES_DATA_ROOT`, `MOF_COMPANIES_QUERY_TIMEOUT_S`,
`MOF_COMPANIES_MAX_COMPANIES`, `MOF_COMPANIES_MAX_CONCURRENCY`, `MOF_COMPANIES_Q4_RULE` (see [Configuration and limits](#configuration-and-limits)).

| Tool | What it does |
|---|---|
| `companies_resolve` | Find companies by name (former names too), ticker, CIK or entity id; returns candidates. |
| `companies_facts` | Raw reported facts of up to `MOF_COMPANIES_MAX_COMPANIES` companies, as filed, with filing reference and provenance. |
| `companies_fundamentals` | Standardised concepts and ratios per company and period, with their provenance. |
| `datasets_list` | The datasets the store publishes: name, version, kind, licence, row estimate. |
| `datasets_describe` | Columns (type, allowed values, doc), natural key and licence of one dataset. |
| `datasets_read_asof` | Rows of a dataset as known at `as_of`, with structured filters (no free SQL), paged with `next_cursor`. |
| `datasets_export_csv` | The same rows as CSV text to save or paste; refused, never cut, above `MOF_MCP_MAX_RESULT_CHARS`. |

- **Point in time.** Company data is bitemporal: `as_of` (a timestamp, default now; a value without an
  offset is read as UTC) returns what was publicly known at that moment, so a restatement filed later
  does not leak into an earlier view. Results echo the `as_of` used. Datetimes are UTC.
- **Concepts and ratios are company-corpus's.** The server computes none: it calls company-corpus for the
  concept catalogue and the ratios. The one quarterly derivation, TTM, is company-corpus's too: by default
  (`MOF_COMPANIES_Q4_RULE=ytd9`) a fourth quarter the company did not report on its own is FY minus the
  9-month year-to-date, the rule of company-corpus's published outputs, computed from the store's own values,
  not by this server. The store holds the 9-month values (`fundamentals` 3.1.0, frequency `ytd9`); a TTM
  window needing a 9-month value the company did not file is absent (no row), never computed another way.
  `sum_of_quarters` instead takes FY minus Q1, Q2 and Q3 whenever those three quarters are in the store.
- **`frequency`.** In `companies_fundamentals`, `annual`, `quarterly` and `semiannual` also return the
  balance-sheet values (frequency `instant`) dated at the ends of the included periods, as the ratios use
  them; `instant` alone returns balance-sheet values only. In `companies_facts`, `frequency` filters the
  filed column as it is (raw), including `ytd9` (the 9-month year-to-date, used only to derive an unreported
  fourth quarter; `companies_fundamentals` has no 9-month periods).
- **Sector unknown.** There is no SIC classification yet: `sector_relevant` is null on the
  sector-sensitive ratios and true on the others.
- **CSV.** `datasets_export_csv` takes the same arguments as `datasets_read_asof`; decimals are written
  exactly (a column's decimals share the widest scale, so `1.5` can appear as `1.500000`). With no rows, the
  header alone if `columns` is given, else an empty string.

## Errors

Tool errors are MCP tool errors whose message starts with a stable code, `[code] message`, so agents can
branch on the code and show the message.

| Code | Meaning | What to do |
|---|---|---|
| `invalid_argument` | A bad argument: `limit` outside `[1, MOF_MCP_MAX_ROWS]`, a cursor this server did not issue, `date_from` after `date_to`, a value the vault database refuses (a `since` offset beyond ±15:59, a NUL character in a text argument); for companies and datasets, `period_from` after `period_to`, a filter value of the wrong type for its column, `'all'` combined with other metrics; for `rag_search`, a blank query, a date not written `YYYY-MM-DD`, an empty list, a filter value the collection does not hold (the message lists the valid ones). | Fix the argument; pass `next_cursor` unchanged. |
| `not_found` | No document with that `doc_id`. | Find ids with `vault_search_documents`. |
| `timeout` | The vault query or the company store read exceeded its time budget (`MOF_MCP_QUERY_TIMEOUT_S` / `MOF_COMPANIES_QUERY_TIMEOUT_S`). | Narrow the filters. |
| `backend_unavailable` | The vault database or the company data store is unreachable (or refused the query). | Retry later; check the deployment. |
| `rag_unavailable` | `rag_search` cannot answer. The message starts with a reason: `qdrant_unreachable` (no answer within `MOF_RAG_QDRANT_TIMEOUT`), `qdrant_auth` (API key refused), `qdrant_error` (another error from Qdrant), `collection_missing`, `collection_mismatch` (vector size or distance), `model_unavailable`, `model_integrity`. It never holds the host, the key or Qdrant's response. | Retry later for `qdrant_unreachable` and `qdrant_error`; the others need the deployment fixed. |
| `limit_exceeded` | `rag_search` was asked for more passages than `MOF_RAG_MAX_LIMIT`. | Ask for at most the maximum the message names. |
| `result_too_large` | The result exceeds `MOF_MCP_MAX_RESULT_CHARS`. Results are refused, never silently cut. | Ask for a smaller `limit` (`rag_search` names the largest that fits). |
| `ambiguous_company` | A company reference matches several companies. | Pick a candidate from the message and pass its CIK. |
| `unknown_company` | No company matches the reference. | Use `companies_resolve`. |
| `unknown_concept` | A concept key is not in the catalogue. | The message lists valid keys. |
| `unknown_metric` | A ratio key is not in the catalogue. | The message lists valid keys. |
| `unknown_dataset` | No dataset with that name. | Names come from `datasets_list`. |
| `unknown_column` | A filter or column is not in the dataset. | Columns come from `datasets_describe`. |
| `too_many_companies` | More companies than `MOF_COMPANIES_MAX_COMPANIES`. | Split the request. |

Arguments that do not match a tool's input schema (wrong type, string too long, e.g. an over-long company
id) are rejected by the MCP framework before a tool runs, without a `[code]`. An unexpected internal failure reaches the client as a generic error
without details (they may hold connection strings); the server log has the rest. Over HTTP, a missing or
unknown token is rejected with HTTP `401` before any tool runs.

## Tiers and tokens

- Two tiers: **`read`** (read-only tools) and **`act`** (every tool; implies `read`). An entry without a
  tier is `act`. v1 ships no act-only tool yet; the tiers are enforced all the same. A tool above the
  caller's tier is not listed to that caller and cannot be called.
- One token per client, each with its tier, in `MOF_MCP_TOKENS` as `client:sha256hex[:tier]`,
  comma-separated. Client ids are 1-32 characters of `A-Z a-z 0-9 _ . -`. The client id appears in the
  audit log line of every tool call (`tool=... client=... outcome=... ms=...`, on stderr; arguments are
  never logged).
- Make a token with `myopenfund-mcp hash-token <client> [--tier read|act]` (in a container:
  `docker compose run --rm mcp myopenfund-mcp hash-token <client>`). It **generates** a random 256-bit
  token, prints it on stderr, and prints the `MOF_MCP_TOKENS` entry alone on stdout. Give the token to that client only; the
  server stores only its SHA-256 digest. It never accepts a token you supply.
- **Entropy:** digests are unsalted SHA-256, which is safe only for high-entropy random tokens such as the
  generated ones. Never use a passphrase or a hand-picked token.
- Revoke a client: remove its entry and restart. Configuration errors name the faulty entry by position,
  never by content, so a misplaced token is not echoed into logs.

## Configuration and limits

Environment variables only (a `.env` file in a deployment); see [`.env.example`](.env.example). An unset
or empty variable takes its default; a bad value stops the server at start, naming the variable.

| Variable | Default | Meaning |
|---|---|---|
| `MOF_MCP_TRANSPORT` | `stdio` | `stdio` or `http`. The Docker image sets `http`. |
| `MOF_MCP_HOST` | `127.0.0.1` | HTTP listen address. The Docker image sets `0.0.0.0` (inside the container). |
| `MOF_MCP_PORT` | `8000` | HTTP port (1-65535). |
| `MOF_MCP_PATH` | `/mcp` | HTTP endpoint path; must start with `/`. |
| `MOF_MCP_TOKENS` | unset | `client:sha256hex[:tier],...`; required when the transport is `http`. |
| `MOF_VAULT_DATABASE_URL` | unset | Vault Postgres URL, read-only role; unset = no vault toolset. Set `sslmode` explicitly (also avoids a wasted SSL attempt per call): `verify-full` with the provider's CA preferred; `require` as a fallback (encrypts, does not authenticate the server); `disable` only on a trusted private network. |
| `MOF_MCP_QUERY_TIMEOUT_S` | `20` | Postgres `statement_timeout` per call, in seconds (1-2147483). |
| `MOF_MCP_CONNECT_TIMEOUT_S` | `10` | Postgres connection timeout, in seconds (2-300). |
| `MOF_MCP_DEFAULT_ROWS` | `50` | Rows per page when the caller gives no `limit`; at most `MOF_MCP_MAX_ROWS`. |
| `MOF_MCP_MAX_ROWS` | `200` | Largest `limit` a caller may ask for. |
| `MOF_MCP_MAX_RESULT_CHARS` | `200000` | Largest tool result in characters (at least 1000); larger results are refused. The cap measures the result's text block (compact JSON); the protocol also carries the same data as structured content, so a response on the wire is roughly twice that size. |
| `MOF_MCP_MAX_ERROR_SAMPLES` | `5` | Error samples kept per row of `vault_runs_status`. |
| `MOF_MCP_MAX_TEXT_CHARS` | `2000` | Each error text in `vault_runs_status` is cut to this length. |
| `MOF_COMPANIES_CATALOG` | unset | my-open-data catalogue connection string as the read-only `lake_reader` role (`postgres:dbname=<db> host=<host> port=5432 user=lake_reader password=<password>`); unset = no companies or datasets toolsets. Needs `MOF_COMPANIES_DATA_ROOT` too. |
| `MOF_COMPANIES_DATA_ROOT` | unset | Folder holding `lake/` as this host sees it; read-only is enough. Needs `MOF_COMPANIES_CATALOG` too. |
| `MOF_COMPANIES_QUERY_TIMEOUT_S` | `20` | Seconds before a store call stops waiting (1-3600). |
| `MOF_COMPANIES_MAX_COMPANIES` | `10` | Most companies a single call may name (1-100). |
| `MOF_COMPANIES_MAX_CONCURRENCY` | `2` | Store reads running at once (1-16); a read abandoned at its time budget keeps its slot until it ends, so the next callers may also get `[timeout]` for a while. |
| `MOF_COMPANIES_Q4_RULE` | `ytd9` | How TTM derives a fourth quarter the company did not report on its own: `ytd9` = FY minus the 9-month year-to-date (the rule of company-corpus's published outputs; a TTM needing a 9-month value the store lacks is absent), `sum_of_quarters` = FY minus Q1, Q2 and Q3 (present whenever those three quarters are, some differ from company-corpus's). |
| `MOF_RAG_QDRANT_HOST` | unset | Qdrant host name or address, alone: a scheme, a path or a port (`qdrant:6333`) is refused at start, the port being `MOF_RAG_QDRANT_PORT`. An IPv6 address is given bare or in brackets. Unset = no rag toolset. Needs `MOF_RAG_COLLECTION` and `MOF_RAG_MODEL_CACHE` too. |
| `MOF_RAG_QDRANT_PORT` | `6333` | Qdrant HTTP port (1-65535). |
| `MOF_RAG_QDRANT_API_KEY` | unset | Qdrant API key; a **read-only** key is enough and recommended. Never logged, shown or put in an error. |
| `MOF_RAG_QDRANT_HTTPS` | `false` | Connect to Qdrant over TLS (`true` / `false`). Without TLS the key travels in clear: private network only. |
| `MOF_RAG_QDRANT_TIMEOUT` | `10` | Seconds one request to Qdrant may take (1-300, at most `MOF_RAG_CALL_TIMEOUT_S`). A request that takes longer fails, and the call that made it answers `[rag_unavailable] qdrant_unreachable`. |
| `MOF_RAG_CALL_TIMEOUT_S` | `15` | Time budget of one `rag_search` call in seconds, its wait for a free slot included (1-3600, at least `MOF_RAG_QDRANT_TIMEOUT`). At its end the call answers `[rag_unavailable]` with `timeout` or `busy`. |
| `MOF_RAG_MAX_CONCURRENCY` | `4` | Searches running at once (1-64), in threads of the toolset's own. The calls in excess wait without holding a thread. |
| `MOF_RAG_CHECK_RETRY_S` | `5` | Seconds a failed collection check or search is remembered (1-3600): calls made meanwhile answer `[rag_unavailable]` at once. Also the longest the toolset takes to notice that Qdrant is back. |
| `MOF_RAG_COLLECTION` | unset | Collection searched (768-dimension cosine vectors of `intfloat/multilingual-e5-base`). Needs `MOF_RAG_QDRANT_HOST` too. |
| `MOF_RAG_MODEL_CACHE` | unset | Directory holding the embedding model files (about 1.1 GB), as an absolute path. Required with the rag toolset, for every transport: the model is never kept in a temporary directory. Fill it once, then mount it read-only. The process needs about 2 GB of memory while the model loads: give a container at least 3 GB. |
| `MOF_RAG_ONNX_THREADS` | unset | Threads of the ONNX runtime (1-256). Unset = one per core of the host; in a container with a CPU limit, set it to the limit. |
| `MOF_RAG_DEFAULT_LIMIT` | `8` | Passages `rag_search` returns when the caller gives no `limit`; at most `MOF_RAG_MAX_LIMIT`. |
| `MOF_RAG_MAX_LIMIT` | `20` | Largest `limit` a caller may ask `rag_search` for (1-1000); a larger one is refused with `[limit_exceeded]`. |
| `MOF_RAG_CODES_TTL_S` | `300` | Seconds the collection's filter values (source codes, doc types, languages) are kept before being read again (1-86400). |
| `MOF_RAG_CODES_RETRY_S` | `5` | Least seconds between two reads of the filter values that the TTL did not ask for: the read made before a value is refused, and the retry after a failed read (1-3600). |
| `MOF_RAG_DESCRIPTION_NOTES` | unset | One line of text appended to the description of `rag_search`: what the collection holds, what its codes mean (at most 1500 characters). See [Passage search](#passage-search-rag) for a suggested value. |

Read by `compose.example.yaml` only, not by the server: `MOF_MCP_BIND_ADDR` (host address the port is
published on, default `127.0.0.1`), `MOF_MCP_BIND_PORT` (default `8000`), `MOF_VAULT_DOCKER_NETWORK` (the
vault stack's Docker network to join).

## Local models

The server works with any MCP client, including clients driving local models. No tool needs a hosted
LLM: every v1 tool is deterministic and the client's model does the reasoning. Tools that will use an LLM
later (such as an answer tool over RAG) are optional, always support a local backend, and are not mounted
when no LLM is configured. Tool descriptions are directive and arguments are validated against the input
schema, to help small models call them correctly.

## Roadmap

- **v1** (this release): `server_info` and the `vault_*` toolset over the central-bank corpus, read-only;
  tiers, tokens, Docker image.
- **v1.1**: `rag_search` (cited passage search over the embedded corpus, delivered); embedding as an `act`
  job and `jobs_*` to follow long actions are still to come.
- **v2**: `companies_*` (identity resolution, filings fetched on demand, reported fundamentals) and
  `datasets_*` over the MyOpenFund data library.
- **Later**: `graph_*` and more sources.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[vault,companies,rag,dev]"
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

The default test run needs no database, network or model. Opt-in suites are excluded by default and run
with an explicit marker:

- `pytest -m integration`: against a real vault database, with `MOF_MCP_IT_DATABASE_URL` set.
- `pytest -m integration tests/integration/test_real_store.py`: against a real company store, with `MOF_MCP_IT_COMPANIES_CATALOG` and `MOF_MCP_IT_COMPANIES_DATA_ROOT` set (optional: `MOF_MCP_IT_COMPANIES_TIMEOUT_S`, default 120; `MOF_MCP_IT_ASSERT_LATENCY=1` to enforce the 5 s criterion, on the NAS only; add `-s` or `-rP` to see the printed fundamentals latency).
- `pytest -m integration tests/integration/test_real_qdrant.py`: `rag_search` against a **throwaway** Qdrant, with `MOF_MCP_IT_QDRANT_URL` set (e.g. `http://127.0.0.1:6333`). The test creates, seeds (through rag-engine, with its model-free `FakeEmbedder`) and deletes collections named `mof-mcp-it-*`. Optional, for a Qdrant with keys: `MOF_MCP_IT_QDRANT_API_KEY` (can write) and `MOF_MCP_IT_QDRANT_READ_ONLY_API_KEY` (given to the server).
- `pytest -m llm`: a scripted session with a local model, with `MOF_MCP_IT_OLLAMA_URL` set.

Build the image with `docker build -t myopenfund-mcp .`; `docker run --rm myopenfund-mcp myopenfund-mcp
--help` lists the commands.

## License

MIT, see [LICENSE](LICENSE).
