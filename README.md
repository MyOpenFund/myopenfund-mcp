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

### Pinned versions

Every rebuild of the image and every start of the `install` service installs the same Python packages,
the ones the CI tested:

- **Base image**: `PYTHON_IMAGE` in the [`Dockerfile`](Dockerfile), a tag **and** a digest
  (`python:3.13.<patch>-slim-<debian>@sha256:...`); the variant without a built image in
  `compose.example.yaml` uses the same value. The image's Python and pip come with it.
- **PyPI runtime dependencies**: [`constraints.txt`](constraints.txt), the exact version of every
  dependency of `[vault,companies,rag]` on that image (fastmcp and everything below it), passed with
  `-c`.
- **Build backends**: this package and its git dependencies are built from source at every install, in
  isolated build environments that `-c` does not reach; [`build-constraints.txt`](build-constraints.txt)
  pins their backends (hatchling, setuptools and their dependencies, plus `editables`, which hatchling
  needs for the CI's editable install), passed with `--build-constraint`. The file records the git pins it
  was generated from: bumping one without regenerating fails the tests. That test compares the recorded
  list only; the `locked` job also runs `scripts/check_constraints.py --build`, which reads every build
  requirement (this package's, and each git dependency's at its pinned commit) and fails when one is not
  pinned in the file at a version its specifier allows.
- **Git dependencies** (`vault-query`, `open-data-store`, `company-corpus`, `myopenfund-rag-engine`): pinned by full
  commit in `pyproject.toml`, as before.

The `install` service passes both files of the same commit as URLs (the same `<commit>` three times):

```
pip install -c https://raw.githubusercontent.com/MyOpenFund/myopenfund-mcp/<commit>/constraints.txt \
  --build-constraint https://raw.githubusercontent.com/MyOpenFund/myopenfund-mcp/<commit>/build-constraints.txt \
  'myopenfund-mcp[...] @ git+https://github.com/MyOpenFund/myopenfund-mcp@<commit>'
```

so the host also needs outbound HTTPS to `raw.githubusercontent.com` at each start, besides `github.com`
and PyPI.

Not covered: the `git` package the build stage installs with apt (a build tool, absent at runtime), file
hashes (pins are `==` versions, not `--require-hashes`, which pip refuses next to git requirements), and,
outside Python, the DuckDB extensions the companies toolset downloads at start and the embedding model
(pinned by its own revision, see [Deploying the rag toolset](#deploying-the-rag-toolset)).

The CI runs the tests twice: floating within the ranges of `pyproject.toml` (Python 3.12 and 3.13, so a
coming break shows early) and locked (`locked` job: the image's Python, both constraint files, then
`scripts/check_constraints.py`, which fails when a runtime dependency of the server is missing from
`constraints.txt` or installed at another version, e.g. a new dependency brought by a release, and when an
extra it is asked to check does not exist, then `scripts/check_constraints.py --build`, which reads the git
dependencies' build requirements from `raw.githubusercontent.com`). A test checks that the job's Python is
the patch of `PYTHON_IMAGE`.

**Upgrading** is a deliberate change in a pull request: for the base image, look up the digest of the new
tag (`docker buildx imagetools inspect python:<tag>`) and replace `PYTHON_IMAGE` in the Dockerfile and the
image lines of `compose.example.yaml`; for the packages, run `python3.13 scripts/update_constraints.py` (on
linux x86_64, the image's platform; it reads the git dependencies' build requirements from GitHub at their
pinned commits) and commit both regenerated files. The `locked` job then tests the new set.

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
reads: it never creates a collection, an index or a point. Angle brackets are placeholders: no real host
name, address or key belongs in this repository or in a compose file under version control.

The steps below work as they are for the two ways `compose.example.yaml` runs the server: the image built
from this repository, or its variant without a built image, where an `install` service puts the package
into a volume at each start of the stack.

1. **A read-only API key on the Qdrant side.** In the Qdrant stack, set a read-only key next to the main
   one (`QDRANT__SERVICE__READ_ONLY_API_KEY`) and restart Qdrant. The MCP gets that key only. Check it
   from any host that reaches Qdrant. The key is typed at a prompt that shows nothing (`read -s`, in bash
   and in zsh) and kept in a variable of that shell: it is written neither in a command nor in the
   shell history.

   ```bash
   # Run this first line alone and type the key: pasted with the lines below, some shells
   # (bash before 5.1, the bash of macOS) take the next pasted line as the key.
   printf 'read-only key: '; read -rs MOF_KEY_CHECK; echo
   curl -s -o /dev/null -w '%{http_code}\n' -H "api-key: $MOF_KEY_CHECK" http://<qdrant-host>:<qdrant-port>/collections/<collection>
   curl -s -o /dev/null -w '%{http_code}\n' -H "api-key: $MOF_KEY_CHECK" -H 'Content-Type: application/json' -X PUT -d '{"vectors":{"size":4,"distance":"Cosine"}}' http://<qdrant-host>:<qdrant-port>/collections/mof-key-check
   ```

   - The first request is a read: it must print `200` (`401` or `403`: Qdrant does not accept the key;
     `404`: no collection of that name).
   - The second is a write: it must print `403`. The key cannot write, and nothing was created.
   - If the second prints `200`, **the key can write**: it is not a read-only key, and the request has
     just created an empty collection named `mof-key-check`. Delete that collection, then give the MCP a
     real read-only key and run the two checks again:

     ```bash
     curl -s -o /dev/null -w '%{http_code}\n' -H "api-key: $MOF_KEY_CHECK" -X DELETE http://<qdrant-host>:<qdrant-port>/collections/mof-key-check
     ```

   - At the end: `unset MOF_KEY_CHECK`.
   - Qdrant behind TLS: write `https://` in these URLs, with the port TLS is served on; add
     `QDRANT_HTTPS=true` to the command of step 2 and set `MOF_RAG_QDRANT_HTTPS=true` in step 6. The key
     then no longer travels in clear.

2. **Payload indexes, once, from the data side.** The filters and the list of valid filter values use
   Qdrant's payload indexes. Whoever owns the collection creates them, once per collection, with a key
   that can write, from an environment where `rag-engine` is installed (the ingestion's own, or any venv
   with `pip install "myopenfund-rag-engine @ git+https://github.com/MyOpenFund/rag-engine@ce5a745a9e0b4a6a96c6b21d79aa0b7b1e7cc0e7"`,
   the commit this server is built on). Give the host AND the port: the engine's default port is 6333.
   The key is typed at a prompt here too:

   ```bash
   # Alone first, as in step 1.
   printf 'key that can write: '; read -rs QDRANT_API_KEY; echo; export QDRANT_API_KEY
   QDRANT_HOST=<qdrant-host> QDRANT_PORT=<qdrant-port> python -c "from rag_engine.store import QdrantStore; print(QdrantStore().ensure_payload_indexes('<collection>'))"
   unset QDRANT_API_KEY
   ```

   It prints the indexes it created (`[]` when they were all there: running it again does nothing). The MCP
   never does this (its key cannot). Without the indexes `rag_search` still answers, more slowly: a
   filtered search reads every point (about 0.15 s on 34,000 points), and the valid filter values are
   read by one scroll of the collection, in the background (up to 100,000 points; beyond that, filter
   values are no longer checked).

   At this commit the engine also creates the datetime indexes on `date_start` and `date_end` (payload
   schema version 2). Once they exist, `auto` mode applies date bounds to the intervals (within
   `MOF_RAG_DATE_FILTER_REFRESH_S` seconds, without a restart of the MCP). That is safe before the payloads
   carry their intervals (a point without one is still found by its `date`), but a month or a year
   document is found by its whole period only once the orchestrator's `sync-metadata` has written the
   intervals; `sync-metadata` creates the two indexes itself (see [Dates](#dates-in-rag_search)).
3. **The `[rag]` extra in the install line.** With the built image: `MOF_EXTRAS=vault,rag` as a build
   argument (`compose.example.yaml`). Without a built image: `'myopenfund-mcp[vault,rag] @ git+https://github.com/MyOpenFund/myopenfund-mcp@<commit>'`
   in the `install` service. From a checkout: `pip install ".[vault,rag]"`. No PyTorch is installed: queries
   are embedded on the ONNX runtime, on CPU. Without the extra the server refuses to start as soon as
   `MOF_RAG_QDRANT_HOST` is set.
4. **Fill the model cache, once.** The model is about 1.1 GB. It is pinned to one revision and every file
   is hash-checked at each start, so a changed or damaged file is refused. The fill is a one-shot
   container with the cache volume mounted **read-write**; it needs outbound HTTPS to `huggingface.co` and
   to the hosts it redirects downloads to (under `hf.co`), and a writable `HOME`. The exact line it runs:

   ```bash
   python -c "from rag_engine.config import Settings; from rag_engine.embedding import make_embedder; make_embedder(Settings(backend='onnx', model_cache='/models')).release(); print('model cache ready')"
   ```

   It succeeded when its **last line is `model cache ready`** and its **exit code is 0**. Anything else
   (a traceback, another exit code) is a failed fill: the service would start without the model.

   - Built image: `docker run --rm -v myopenfund-mcp-rag-models:/models myopenfund-mcp python -c "..."`
     with the line above (the image owns `/models` as its service user and sets `HOME`).
   - No built image: the `rag-model-fill` service of `compose.example.yaml` runs it for the first start
     of the stack (the `install` service gives the volume to the service user). Check both signs with
     `docker compose logs rag-model-fill` (last line `model cache ready`) and `docker compose ps -a
     rag-model-fill` (`Exited (0)`), then delete that service.
   - Without Docker: the same line in the server's venv, with your own absolute directory in place of
     `/models`.

   Run again on a filled cache, the line downloads nothing, verifies the files and prints the same line. A filled
   directory can be copied from another host instead, **keeping its symbolic links** (`snapshots/` links
   into `blobs/`): `cp -a`, `rsync -a` or `tar`, never a copy that follows or drops links.
5. **The service** (`compose.yaml`):
   - the cache volume mounted **read-only** at `/models`;
   - `mem_limit: 3g` or more: the process peaks at about 2 GB at every start, while the model loads, and
     stays around 1.5 GB afterwards; with a CPU limit, also set `MOF_RAG_ONNX_THREADS` to it;
   - `ORT_DISABLE_TELEMETRY=1` and `HF_HUB_OFFLINE=1` in its environment (see
     [Outbound connections](#outbound-connections-of-the-rag-toolset));
   - a healthcheck start period of 60 s: the port opens once the model is loaded, 5 to 7 s from a warm
     disk on 4 cores, more than 30 s when the 1.1 GB file is read from a cold disk (the image's
     `HEALTHCHECK` has it);
   - a route to Qdrant. On the same Docker host, join the Qdrant stack's network and use its **service
     name** and its **internal port** (`MOF_RAG_QDRANT_HOST=<qdrant-service>`, `MOF_RAG_QDRANT_PORT=6333`
     unless that stack changed it), without publishing anything. Otherwise use a host port Qdrant
     publishes on a private interface (`MOF_RAG_QDRANT_HOST=<private-address>`,
     `MOF_RAG_QDRANT_PORT=<published-port>`). Without TLS the key travels in clear: private network only.
6. **The variables**, in `.env`: `MOF_RAG_QDRANT_HOST` (the host alone: `qdrant:6333` is refused),
   `MOF_RAG_QDRANT_PORT`, `MOF_RAG_QDRANT_API_KEY` (the read-only key of step 1), `MOF_RAG_COLLECTION`,
   `MOF_RAG_MODEL_CACHE=/models`; `MOF_RAG_QDRANT_HTTPS=true` when Qdrant is behind TLS; optionally
   `MOF_RAG_DESCRIPTION_NOTES` (see [Passage search](#passage-search-rag)). The others have defaults
   ([Configuration and limits](#configuration-and-limits)); the date settings
   (`MOF_RAG_DATE_FILTER_MODE=auto`, `MOF_RAG_DATE_FILTER_REFRESH_S=300`, `MOF_RAG_DATE_FILTER_RETRY_S=30`) need
   no value of their own.
7. **Restart and check.**
   - `GET /health` answers `200` within the start period.
   - The log holds, in this order, `rag: embedding model intfloat/multilingual-e5-base ready in ... s`,
     `rag: collection <collection> checked, searches are served` and `rag: filter values read from the
     collection (source_codes: ..., doc_types: ..., languages: ...)`, then (logger `rag_engine.store`)
     `Qdrant collection '<collection>' has no datetime index on both date_start and date_end: date filters
     in auto mode use the date field (legacy)` (before `sync-metadata`) or `... has datetime indexes on
     date_start and date_end: date filters in auto mode use the intervals` (after). It holds **no** line with the word
     `Downloading`: the running service never downloads. An error line naming `MOF_RAG_MODEL_CACHE` means
     the cache is incomplete, unreadable or not the pinned model.
   - `server_info`: `toolsets.rag` is `enabled`, `rag.status` is `ready`, and `rag.source_codes` /
     `rag.doc_types` list the collection's codes. If `rag.status` is `unavailable`, `rag.reason` and the log
     say why (see [Passage search](#passage-search-rag)).
   - `rag_search` with a short query and one valid code, e.g. `{"query": "inflation outlook",
     "source_codes": ["<a code of rag.source_codes>"]}`: passages come back.
   - `rag_search` with the same code in the wrong case: the answer is `[invalid_argument] source_codes:
     unknown value "..." (did you mean "..."? values are matched exactly); valid values: ...`.
   - `rag_search` with a date bound, e.g. `{"query": "inflation outlook", "date_from": "2016-07-14",
     "date_to": "2016-07-14", "include_undated": false}`: passages come back (or none, for a day nothing is
     dated to), `filters.include_undated` is `false`, and each hit has `date_precision` and `date_kind`
     (`null` until `sync-metadata` has run).
8. **Never run this repository's integration tests against a production Qdrant.**
   `tests/integration/test_real_qdrant.py` creates and deletes collections: it is for a throwaway Qdrant
   only (see [Development](#development)).

**Redeploying a new version** (such as the one that raised the rag-engine pin to `6be21ac` and added the date
settings): the same steps, from step 3. Rebuild the image, or point the `install` service at the new
`<commit>`; restart; check as in step 7. No new volume, no new variable to set (the date settings have
defaults), and the model cache is kept: the pinned model revision did not change. Step 2 is not needed
again for the server: the indexes the new version uses are created by the orchestrator's `sync-metadata`,
and until they exist date bounds behave as before.

### Outbound connections of the rag toolset

Once the model cache is filled, the server makes **no outbound connection** besides its requests to
Qdrant (checked behind a recording proxy that refuses everything: no request in the first minute):

- `ORT_DISABLE_TELEMETRY`: the ONNX runtime installed by the `[rag]` extra sends usage events to its
  publisher (`mobile.events.data.microsoft.com`) and keeps a device id under
  `$HOME/.cache/Microsoft/DeveloperTools` unless this variable is `1`. The server sets it to `1` in its own
  process before the runtime can be imported, and so do the image and `compose.example.yaml`. A value you
  set yourself is kept (`0` allows the telemetry).
- `HF_HUB_OFFLINE=1` is recommended for the running service. With a complete cache the server asks
  `huggingface.co` nothing; with an incomplete one it would try a download at every start, fail on a
  read-only volume, and leave `rag_search` answering `[rag_unavailable] model_unavailable`. The variable
  turns that attempt into an immediate, local error. Leave it unset for the fill of step 4.

## Tools

Tool names carry their toolset prefix. All v1 tools are read-only (`readOnlyHint`). Only
`vault_search_documents` is paged: it returns `next_cursor` to pass back unchanged with the same filters.
`vault_corpus_stats`, `vault_runs_status` and `vault_vocabulary` only bound their lists with `limit` (no cursor). Every
`limit` defaults to `MOF_MCP_DEFAULT_ROWS`; a value outside `[1, MOF_MCP_MAX_ROWS]` is refused, not
clamped. String arguments `doc_id`, `text`, `cursor` and the `corpus` of `vault_get_document` and
`vault_vocabulary` are at most 200 characters.

| Tool | Purpose | Key arguments |
|---|---|---|
| `server_info` | What this deployment offers: enabled toolsets, your tier, version, data freshness (last vault ingest). Call it first. Open to every authenticated caller. | none |
| `vault_search_documents` | Find official documents (metadata, not text). Start here: filter, then open one with `vault_get_document`. Rows are ordered by `sort_by` in `sort_dir`; equal sort values are ordered by `doc_id`, then `corpus`, ascending, whatever `sort_dir` (a total order: on an unchanged vault, pages never repeat or skip a row). `total` counts the whole filtered set. Cursors are offsets: if the vault is ingesting between two pages, a row can shift (seen twice or skipped). Rows carry, besides the identity, title, dates, type and URLs, the documents-contract columns `date_precision` (`day`, `month`, `year`, or null = not stated: a month or year date is the period's first day), `date_kind` (`publication`, `meeting`, `period`) and `entity_scheme` / `entity_value` (the issuing entity, e.g. `cik` / `0000320193`). | `corpus` (e.g. `central-bank`), `source_code` (publisher, e.g. `us` for the Fed), `doc_type` (a code within its corpus: **needs `corpus`**, see below), `language`, `year`, `mime_type`, `date_from` / `date_to` (YYYY-MM-DD, inclusive), `text` (case-insensitive substring of the title, matched literally: `%` and `_` are ordinary characters), `sort_by` (`date` default, `year`, `corpus`, `source_code`, `doc_type`, `title`, `created_at`), `sort_dir` (`desc` default), `limit`, `cursor`, `include_deleted` (also return withdrawn documents, with `deleted_at` set) |
| `vault_get_document` | Every stored fact of a document: dates (with `date_precision`, `date_kind`), the document's URL (`document_url`, PDF preferred, HTML accepted) and the page it was found on (`source_url`), publisher, type, issuing entity, `extra`. A doc_id is unique within its corpus only, so it returns a **list**: with `corpus`, at most one document (the one of that corpus); without, one per corpus holding that doc_id, ordered by corpus, each with its `corpus`. None: `[not_found]` (with `corpus`, the message says to omit it to look in every corpus). The text block is the JSON list; `structuredContent` is `{"result": [...]}` (MCP structured output must be an object). | `doc_id`, `corpus` (pass the `corpus` of the search row) |
| `vault_corpus_stats` | Live-document counts grouped by one field, largest first (NULL is a real group). Grouping by `doc_type` needs `corpus`. | `group_by` (`source_code` default, `corpus`, `doc_type`, `language`, `year`, `provenance`, `mime_type`), `corpus`, `limit` |
| `vault_vocabulary` | What the codes mean: each corpus's declared codes with their labels, rows `{corpus, kind, code, label, family}` ordered by corpus, kind, code, with `returned` and `truncated` (true when `limit` cut the list: filter by corpus and kind). For turning the user's words ("10-K", "rate decision") into a corpus's `doc_type`. A corpus that has not declared its codes yet has no rows: `vault_corpus_stats(group_by="doc_type", corpus=...)` shows the codes its documents use. | `corpus`, `kind` (`doc_type`, `source_code`, `provenance`, `entity_scheme`), `limit` |
| `vault_runs_status` | Is the pipeline healthy? Four lists, newest first unless stated: `runs` (one row per run, with outcome and totals; a run that crashed before reaching any source appears only here), `source_runs` (one row per run and source: counts, outcome, bounded error samples; `bad_content` is the part of `docs_failed` where the server sent the wrong bytes, e.g. an HTML page for a PDF, NULL = not reported, never 0), `open_discovery_errors` (unresolved discovery failures; while `seen_at_is_ingestion_time` is true, `last_seen_at` is approximate), `source_health` (expected-vs-observed publication cadence per series, the least healthy first: `cadence_status` overdue, then soon, then the rest, each by most `days_late` first, unknown last; each row also carries `source_bad_content_7d`, the 7-day sum, NULL when no run of the window reported it, and `last_run_bad_content`), and `source_health_total` (every series matching `corpus` / `source_code`, so a shorter list means `limit` left some out). A result over `MOF_MCP_MAX_RESULT_CHARS` is refused with `[result_too_large]`: lower `limit` or filter by `source_code`. | `corpus`, `source_code`, `since` (ISO 8601, UTC if no offset, an explicit offset is kept: whole minutes, at most ±15:59; keeps runs *finished* at or after it, leaves out runs still in flight; applies to `runs` and `source_runs` only), `until` (same form; keeps runs *finished* at or before it, so old runs are reachable under `limit`), `limit` (applies to each of `runs`, `source_runs`, `open_discovery_errors`, `source_health`) |

**`doc_type` needs `corpus`.** A doc_type code means something only within its corpus (central-bank `A1`
and company `A1` are different document types). `vault_search_documents` with `doc_type` and no `corpus`,
and `vault_corpus_stats` grouped by `doc_type` without `corpus`, answer `[invalid_argument] doc_type needs
corpus: ...` with vault-query's message, which lists each corpus's codes and labels. The tool descriptions
tell the model to ask the user which corpus when the conversation does not make it clear, or infer it.

Resource: `vault://documents/{doc_id}`, the metadata of the documents with that doc_id as a JSON list, one
per corpus holding it (same content as `vault_get_document` without `corpus`).

**Changes for clients (vault-query 0.5.0, vault `12ffd11`).** `vault_get_document` and the resource return
a list (was one object); `vault_get_document` takes `corpus`; `doc_type` without `corpus` is refused (see
above); search and document rows carry `date_precision`, `date_kind`, `entity_scheme`, `entity_value`; new
tool `vault_vocabulary`.

The `vault_*` tools and the resource exist only when `MOF_VAULT_DATABASE_URL` is set; the `[vault]` extra
must then be installed, or the server refuses to start (`MOF_VAULT_DATABASE_URL is set but the [vault]
extra is not installed`). Timestamps in results are UTC: the server's database session runs with
`timezone=UTC`, whatever the database or role default.

**Minimum vault schema.** This server needs a vault at or after `12ffd11` (vault-query 0.5.0, pinned at 0.6.0 `d3c4497`, which takes `until` on run reads and breaks search sort ties by `doc_id, corpus`: documents
keyed `(corpus, doc_id)`, columns `date_precision` / `date_kind` / `entity_scheme` / `entity_value`, table
`vocabulary`; `runs_sources.bad_content`). Deploy the vault (and run its ingestion, which applies the
schema) before this server. Against an older schema, a tool that reads a missing column or table answers
`[schema_outdated] the vault lacks a column or table this server reads (see README, Minimum vault schema)`,
not `backend_unavailable`; the server log names the missing column or table, and `server_info` reports
`"freshness_error": "schema_outdated"`.

If the vault database is down, `server_info` still answers (version, tier, toolsets) with `freshness`
null and `freshness_error` set to the error's code (`backend_unavailable`, `timeout`, `schema_outdated`);
the server log names the exception class only.

### Passage search (rag)

`rag_search` exists only when `MOF_RAG_QDRANT_HOST`, `MOF_RAG_COLLECTION` and `MOF_RAG_MODEL_CACHE` are set;
the `[rag]` extra must then be installed, or the server refuses to start (`MOF_RAG_QDRANT_HOST is set but
the [rag] extra is not installed`). It searches the text of the embedded documents by meaning (one dense
search over the chunks, with rag-engine) and returns passages, best first. It does not write an answer:
the client's model reads the passages and cites them. Read tier.

| Argument | Meaning |
|---|---|
| `query` (required) | What to look for, in plain language, in any language; at most 2000 characters (the model reads the first 512 tokens). |
| `source_codes` | JSON array of source codes, exact (case matters). The valid ones: `server_info`, `rag.source_codes`. |
| `doc_types` | JSON array of document-type codes, exact. The valid ones: `server_info`, `rag.doc_types`. |
| `languages` | JSON array of language codes as stored, e.g. `["en"]`. The valid ones: `server_info`, `rag.languages`. |
| `date_from`, `date_to` | Date bounds, `YYYY-MM-DD`, both inclusive. A document known only to the month or the year matches any range overlapping its period (see [Dates](#dates-in-rag_search) for when). As soon as one bound is set, a document without a date is left out, unless `include_undated` is true. |
| `include_undated` | JSON boolean, default `false`. `true`: with a date bound, also keep the documents without a date (the dated ones outside the range stay out). Without a date bound it changes nothing and is accepted. A string (`"true"`) or a number is refused. |
| `limit` | Number of passages; default `MOF_RAG_DEFAULT_LIMIT` (8), at most `MOF_RAG_MAX_LIMIT` (20). |

The values of one array are alternatives (OR), the arguments combine (AND). An omitted argument does not
restrict; an empty array is refused.

**What the client's model is told.** The tool description states how to call the tool, the default and
the maximum `limit` of this deployment, what a date bound does (with a bound, a document known only to the
month or the year matches any range overlapping its period, and documents without a date are left out
unless `include_undated` is true; with `MOF_RAG_DATE_FILTER_MODE=legacy` it says instead that such a
document is found only at the first day of its period), and the hit fields, `date_precision` and
`date_kind` included. It stays under 1800 characters before the notes. It is **corpus-neutral**: it names no source
and gives no meaning to any code, because such facts go stale when the collection changes. The codes
themselves are read from the collection (`server_info`, and every refusal lists them). What they mean is
the deployment's to say, in `MOF_RAG_DESCRIPTION_NOTES`: one line of text (at most 1500 characters),
appended to the description after `About this collection:`. Suggested value for the central-bank
collection:

```text
MOF_RAG_DESCRIPTION_NOTES=Central-bank monetary-policy documents. source_codes: "ecb" European Central Bank, "us" Federal Reserve, "gb" Bank of England, "au" Reserve Bank of Australia, "ca" Bank of Canada, "jp" Bank of Japan. doc_types: "A1" rate-decision press release; "A2" policy statement (for the ECB: the press-conference page, with its questions and answers); "A3" minutes or accounts of the meeting; "B1" press-conference material (Federal Reserve: transcript; ECB: the combined decision and statement, without the Q&A); "E1" monetary policy report or staff projections; "F1" economic projections.
```

Example call and result (text shortened):

```json
{"query": "Why did the ECB raise interest rates in July 2022?", "source_codes": ["ecb"], "date_from": "2022-07-01", "limit": 1}
```

```json
{
  "query": "Why did the ECB raise interest rates in July 2022?",
  "filters": {"source_codes": ["ecb"], "doc_types": null, "languages": null, "date_from": "2022-07-01", "date_to": null, "include_undated": false},
  "count": 1,
  "hits": [
    {
      "rank": 1,
      "score": 0.8189,
      "text": "The Governing Council decided to raise the three key ECB interest rates by 50 basis points. ...",
      "doc_id": "ecb/A1/2022-07-21",
      "title": "Monetary policy decisions",
      "date": "2022-07-21",
      "date_precision": "day",
      "date_kind": "meeting",
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
  hits may come from one document; nothing is grouped. `filters` echoes the filters as applied,
  `include_undated` included. The point id and the rest of the stored payload are not returned.
- **Date fields.** `date` is the stored date, `YYYY-MM-DD`. `date_precision` says how much of it is known:
  `day`, `month` or `year` (a month or a year is stored as the first day of its period: `2016-07-01` with
  `month` is July 2016), or `null` when unknown. `date_kind` says what the date marks, as the producer
  labels it (`meeting`, `publication`, `period`), or `null`. Both are read from the stored payload as they
  are: **`null` is normal** for a collection whose payloads do not carry them yet (the production
  collection until the orchestrator's `sync-metadata` has run), and an unknown precision is read as a day.
- **Pages.** `page` is the page the passage starts on **as a reader counts it: the first page is 1** (the
  collection stores pages from 0; the server adds one). It is `null` for a document without pages, and the
  `citation` then has no page part. `citation` is `<title> (<source_code> <doc_type>, <date>), p. <page>`,
  each missing part left out (the `doc_id` stands for a missing title). The date is written at its
  precision: `July 2016` for `month`, `2016` for `year` (English month names), the stored date for `day`,
  for an unknown precision and for a date not written `YYYY-MM-DD`.
- **Argument errors are written for the client's model**, in terms of the JSON to send: a string where an
  array is expected, the array itself written as a string included (`source_codes must be a JSON array of
  strings, e.g. ["ecb"]; omit it for no restriction`, the example being the caller's own value), an empty
  array (`... omit the argument for no restriction`), a date in another form (`date_from must be
  "YYYY-MM-DD", e.g. "2016-03-10"`), a boolean written otherwise (`include_undated must be a JSON boolean,
  true or false without quotes, not "true"; omit it for false`). Nothing is corrected silently.
- **Unknown filter values are refused, never corrected.** The server reads the distinct `source_code`,
  `doc_type` and `language` values from the collection and keeps them `MOF_RAG_CODES_TTL_S` seconds. A
  value that is not among them is `[invalid_argument]` listing the valid values (as JSON strings), with a
  hint for a near miss (`"ECB"`: did you mean `"ecb"`?), instead of an empty result.
  - With the keyword indexes, the values come from Qdrant's facet API (a few milliseconds), at start and
    then when a filtered call finds them older than the TTL.
  - Without them, one scroll of the collection reads the three fields. It always runs in the background
    (it starts when the server does): no search waits for it, the searches made meanwhile are checked
    against the values already known, and not checked while none is known yet.
  - Before a value is refused, the values are read once more (not more often than every
    `MOF_RAG_CODES_RETRY_S` seconds): a code ingested since the last read is accepted at once, not after
    the TTL.
  - The values are dropped when the collection comes back after an outage (it may have been recreated).
    A field that holds no value at all (an empty collection) refuses nothing. If the values cannot be
    read, the last ones read stay in use, or nothing is checked when there is none (one warning in the
    log), and the read is tried again `MOF_RAG_CODES_RETRY_S` seconds later.
- **Limits and size.** A `limit` above `MOF_RAG_MAX_LIMIT` is `[limit_exceeded]`, never lowered. A result
  larger than `MOF_MCP_MAX_RESULT_CHARS` is refused with `[result_too_large]`, never cut; the message names
  the largest `limit` that fits. Passages are 200 to 400 words: the default 8 are a few thousand tokens.
- **Start-up.** The embedding model is loaded and the collection is checked when the server starts, not at
  the first call, and a model is never loaded or downloaded by a tool call. If Qdrant cannot be reached,
  refuses the key, or does not hold the collection, **the server still starts** (as it does when the vault
  database is down): `rag_search` answers `[rag_unavailable]` with the reason, and works again by itself
  once a collection check passes (see the next point). If the model cannot be loaded (files missing with
  no way to download them, or a file that is not the pinned one), the server starts too, logs the cause as
  an error naming `MOF_RAG_MODEL_CACHE`, and `rag_search` answers `[rag_unavailable]` until the cache is
  fixed and the server restarted.
- **Time budget, and what happens when Qdrant hangs.** One call never takes longer than
  `MOF_RAG_CALL_TIMEOUT_S` (15 s), its wait included. `MOF_RAG_QDRANT_TIMEOUT` (10 s) is the limit of
  **one request** to Qdrant, and a call makes several: the search, before it the read of the filter
  values when it is due, the collection check after a failure, and, for a call with a date bound in
  `auto` mode, the read of the collection's payload schema when it is due (see
  [Dates](#dates-in-rag_search)).
  - A request that fails (no answer in time, connection refused, key refused, collection gone, server
    error) ends its call with `[rag_unavailable]` and the reason: a call pays for one request timeout at
    most, never for two in a row (a failed read of the filter values is not followed by a search).
  - **A failed search refuses nobody else.** It only makes the next call check the collection before
    its own search. With a healthy Qdrant (one slow request, one caller's timeout) that check takes a
    few milliseconds and the call is served; a search that succeeds meanwhile cancels the check.
    When the check passes and the search that follows fails again before any search is served (Qdrant
    answers the check and no search), that failure is remembered like a failed check: one call per
    retry period pays for a request timeout, the others are answered at once (the calls that arrive
    while that one search is still waiting pay for it too: under steady traffic, more than one). A
    search that is served clears that memory at once.
  - **A failed collection check** (at start, or the one that follows a failed search) is remembered
    `MOF_RAG_CHECK_RETRY_S` seconds (5): the calls made meanwhile get its reason **at once**, without a
    request, and so do the calls that arrive while a check is running. Then one call checks again. So
    with Qdrant frozen, a burst of 50 calls is answered within one request timeout (only the call that
    runs the check after a failed search waits longer, for that check or the end of its own budget),
    later calls within milliseconds, and the tool works again at most `MOF_RAG_CHECK_RETRY_S` seconds
    after Qdrant does.
  - **A search that Qdrant refuses as a request** (HTTP 4xx, other than a refused key, a missing
    collection, 408 and 429) is the error of that call alone: `[rag_unavailable] search_refused`,
    nobody else is refused and `server_info` stays `ready`. The next call checks the collection before
    its search (a few milliseconds): when it is the collection that changed (recreated with another
    vector size, every search is refused), that check says so with `collection_mismatch`. The usual cause is a
    collection in strict mode asked to filter on a field that has no payload index (a date bound without
    the datetime index on `date`): the filter cannot be served by the collection as it is indexed. Step 2
    of [Deploying the rag toolset](#deploying-the-rag-toolset) creates the indexes.
  - At most `MOF_RAG_MAX_CONCURRENCY` searches (4) run at once, in threads of the toolset's own: the
    calls in excess wait without holding a thread, so the other toolsets and `server_info` keep
    answering whatever Qdrant does. A call whose budget ends answers `[rag_unavailable]`: `busy` if it
    never got a slot, `timeout` if its search was still running. Such a search keeps its slot until its
    request to Qdrant ends: abandoned searches cannot pile up.
- **`server_info`** adds a `rag` object: `status` (`ready`, or `unavailable` with a `reason`), `collection`,
  `model`, `engine_version`, `default_limit`, `max_limit`, and `source_codes`, `doc_types`, `languages`:
  the valid values of each filter as last read from the collection (at most 60 each; `null` while not
  known). It reports the outcome of the last collection check (at start, then by the calls that had to
  run one), without a request to Qdrant: a search that failed does not change it, the check that follows
  does.
- **Privacy.** As for every tool, the audit line holds the tool name, the client, the outcome and the
  duration, never the arguments, and the tool's own log lines never contain the query. At the levels the
  CLI logs (INFO and above), no line written for a search holds the Qdrant host, the API key or a
  response body, the framework's and rag-engine's included:
  - every exception of a search is caught by the toolset and answered with a coded reason, so FastMCP
    never prints one. An error the toolset maps (an outage, a refusal, an answer that is not JSON, such as
    a proxy's HTML page sent with status 200: `qdrant_error`) is logged by its class; any other is
    answered `internal_error` and logged at ERROR by its class only, its text and traceback at DEBUG
    only, a level the CLI never turns on (a developer who sets it accepts them);
  - the server routes every record of the `rag_engine` loggers (present and future modules) through a
    filter. A record that carries an error (an exception among its arguments or attached to it) or whose text holds the configured host
  (a whole word, as configured or lowercased, alone or with a port: host `qdrant` is found in
  `http://qdrant:6333`, not in a collection `qdrant-docs`; the configured collection's name, quoted as the
  engine writes it, is not taken for the host unless it is the host) or the key (anywhere, even inside a
  longer word) is written with its logger, its level and a fixed message naming the error's class only; the
  engine's warning on a failed schema read also keeps its collection, the filter used meanwhile and the
  delay before the next try. Other engine lines (the collection and the filter in use) are written as
  they are. The error's text is dropped because it can hold the host, a TLS host name, a proxy's error
  page naming an upstream host or an internal address, or the engine's hint about its own key variable.
  **What this changes in the process's logging** (for an operator who embeds the server or configures
  logging): the `rag_engine` logger stops propagating and gets one handler that filters each record,
  then hands it to the root logger's handlers, or to `logging.lastResort` (stderr, WARNING and above)
  when there is none, as the standard library does. The change is process-wide and lasts as long as the
  process. A later `logging.config.dictConfig` that names `rag_engine` replaces that handler, and the
  engine's records are then written as the engine writes them: leave `rag_engine` out of such a
  configuration, or build the rag toolset after it. The CLI configures logging before it builds the
  toolsets.
  The HTTP libraries under the Qdrant client are apart: `httpx` logs one INFO line per request with
  the server's address, which is why the CLI raises its logger to WARNING, and `httpcore` logs the host
  and port of each connection at DEBUG only.
- **Concurrency.** One embedder and one Qdrant client are shared by the process; inference runs one query
  at a time (about 20 ms for a short query on 4 cores) off the event loop.

#### Dates in `rag_search`

**What a bound means.** A document is dated by `date` and `date_precision`; rag-engine stores the period
it covers as `date_start` and `date_end` (the same day for `day`, the whole month for `month`, the whole
year for `year`). With intervals in use, a bound keeps a document when its period has at least one day in
common with the range: the minutes of a July 2016 meeting known only to the month are found by
`2016-07-14`..`2016-07-14` and by any range touching July 2016; a document of the year 2015 by any range
touching 2015. A document dated to the day is found on that day, as before. A document without a date is
left out of every bounded search unless `include_undated` is true. Without intervals (`legacy`) a month or
a year document is found only at the first day of its period.

**When results change.** Intervals are used only once the collection's payloads carry `date_start` and
`date_end` **and** both fields have a datetime index. The orchestrator's `sync-metadata` writes the
intervals (with the corrected dates, precisions and kinds) and creates the two indexes; until it has run,
`auto` mode finds no index and every bound is applied to `date`, exactly as before this version, and
`date_precision` / `date_kind` are `null` in the hits. Within `MOF_RAG_DATE_FILTER_REFRESH_S` seconds after the
indexes appear, the running server uses them, without a restart. A collection whose indexes exist before
all its payloads have intervals loses nothing: a point without its interval yet is still found by its
`date`.

How a date bound is applied is rag-engine's `date_filter_mode`, set here by `MOF_RAG_DATE_FILTER_MODE`
and passed to the engine with every search, with `MOF_RAG_DATE_FILTER_REFRESH_S` and
`MOF_RAG_DATE_FILTER_RETRY_S` (the engine's own `RAG_ENGINE_DATE_FILTER_*` variables are never read).

- **Mode `auto` (the default) follows the collection.** A call with a date bound asks the collection
  whether `date_start` and `date_end` both have a datetime index (one read request, which the read-only
  key may send), and the answer is kept `MOF_RAG_DATE_FILTER_REFRESH_S` seconds (300). Indexed: the bound
  is applied to the intervals; not indexed: to the `date` field, as before. A call without a date bound
  never asks. The read is also made once at start, after the collection check and the read of the filter
  values (not when either failed: then the first date-bounded call makes it), so the log says which
  filter is in use (a line of the `rag_engine.store` logger: `... date filters in auto mode use the
  intervals`, or `... use the date field (legacy)`), and it is made again by the first date-bounded call
  after the refresh delay: a collection upgraded while the server runs is followed without a restart.
- **The read is part of the call.** It runs in the call's own slot (one of `MOF_RAG_MAX_CONCURRENCY`) and
  within its budget (`MOF_RAG_CALL_TIMEOUT_S`); when it ends after the budget, the search is not sent and
  the call has already answered `[rag_unavailable] timeout`.
- **A failed read refuses no call and is not an outage.** When the read fails or times out, the engine
  answers with the last answer it read, or with the `date` field when it never read one, logs one
  warning (logger `rag_engine.store`, naming the error's class only: see Privacy above), and reads
  again `MOF_RAG_DATE_FILTER_RETRY_S` seconds later (30) at the earliest; the calls made meanwhile ask
  nothing. The toolset does not remember that failure: no `[rag_unavailable]` follows from it,
  `server_info` stays `ready`, and only the search itself says whether Qdrant answers. So with Qdrant
  frozen, a date-bounded call may spend one request timeout on that read before its search, once per
  retry delay: its budget still bounds it. While no answer has ever been read (a cold start, or a start
  that skipped the read), the date-bounded calls that arrive during that first read **wait for it**,
  each in its slot: a burst of them can hold every one of the `MOF_RAG_MAX_CONCURRENCY` slots for one
  request timeout, once, and the calls behind them, with or without a date bound, wait as long (measured
  with a 3 s request timeout and 4 slots: 8 date-bounded calls answered in about 3 s, an undated call sent
  meanwhile in 2.8 s). Once an answer is known, or the read has failed, the other calls never wait for a
  read in progress.
- **Modes `interval` and `legacy`** never read the collection: always the intervals, or always the `date`
  field. `interval` on a collection in strict mode without the two indexes is refused by Qdrant
  (`[rag_unavailable] search_refused`, the error of that call alone).

### Companies and datasets

The `companies_*` and `datasets_*` tools (read-only, over the published open-data-store) exist only when
`MOF_COMPANIES_CATALOG` and `MOF_COMPANIES_DATA_ROOT` are set; the `[companies]` extra must then be installed
(from a checkout, `pip install ".[vault,companies]"`, or from the git URL,
`pip install "myopenfund-mcp[companies] @ git+https://github.com/MyOpenFund/myopenfund-mcp"`; the extra pins two
git dependencies, so git must be available). The
store is read-only (as the `lake_reader` role); nothing is ever written. In a hardened container, DuckDB
downloads and loads its `ducklake` and `postgres` extensions under `$HOME/.duckdb` at each start: `HOME` must be
writable, a tmpfs holding it must allow exec (`/tmp:exec`), and the container needs outbound HTTPS to
`extensions.duckdb.org`. The process must also be able to read the store's `lake/` files (open-data-store creates
them `0700` for its own uid). Configuration:
`MOF_COMPANIES_CATALOG`, `MOF_COMPANIES_DATA_ROOT`, `MOF_COMPANIES_QUERY_TIMEOUT_S`,
`MOF_COMPANIES_MAX_COMPANIES`, `MOF_COMPANIES_MAX_CONCURRENCY`, `MOF_COMPANIES_Q4_RULE` (see [Configuration and limits](#configuration-and-limits)).

| Tool | What it does |
|---|---|
| `companies_resolve` | Find companies by name (former names too), ticker, CIK or entity id; returns candidates. |
| `companies_facts` | Raw reported facts of up to `MOF_COMPANIES_MAX_COMPANIES` companies, as filed, with filing reference and provenance. |
| `companies_fundamentals` | Standardised concepts and ratios per company and period, with their provenance. |
| `companies_filings` | A company's filings (`forms`, report-period window) with `accepted_utc`, the time each became public on EDGAR: the event date for share-price studies. From the store's `filings` dataset, else SEC. |
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
- **Time budget, and a store read over SMB.** One call, all its store reads together, gets
  `MOF_COMPANIES_QUERY_TIMEOUT_S` seconds (default 20) and then answers `[timeout]`. The default is sized
  for the NAS, where the server runs next to the store and reads `lake/` from local disk. Read over SMB
  instead (measured from a Mac, 2026-11), `companies_fundamentals` took 42 to 51 s cold and about 17 s
  warm, so the default fails cold calls. A deployment that reads the store over SMB (or another network
  share) should raise `MOF_COMPANIES_QUERY_TIMEOUT_S`, e.g. to 90; the client's own tool-call timeout must
  then be longer than that too.

## Errors

Tool errors are MCP tool errors whose message starts with a stable code, `[code] message`, so agents can
branch on the code and show the message.

| Code | Meaning | What to do |
|---|---|---|
| `invalid_argument` | A bad argument: `limit` outside `[1, MOF_MCP_MAX_ROWS]`, a cursor this server did not issue, `date_from` after `date_to`, a `doc_type` filter or grouping without `corpus` (the message lists each corpus's codes: ask the user which corpus, or infer it), a value the vault database refuses (a `since` offset beyond ±15:59, a NUL character in a text argument); for companies and datasets, `period_from` after `period_to`, a filter value of the wrong type for its column, `'all'` combined with other metrics; for `rag_search`, a blank query, a string where a JSON array is expected, an empty array, a date not written `YYYY-MM-DD`, an `include_undated` that is not a JSON `true` or `false`, a filter value the collection does not hold (the message lists the valid ones). | Fix the argument; pass `next_cursor` unchanged. |
| `not_found` | No document with that `doc_id` (in that `corpus`, when one is given: omit `corpus` to look in every corpus). | Find ids with `vault_search_documents`. |
| `timeout` | The vault query or the company store read exceeded its time budget (`MOF_MCP_QUERY_TIMEOUT_S` / `MOF_COMPANIES_QUERY_TIMEOUT_S`). | Narrow the filters. |
| `schema_outdated` | The vault database lacks a column or table the pinned vault-query reads: the vault is older than this server (see Minimum vault schema under [Tools](#tools)). | Deploy the vault (and run its ingestion) first; check the deployment. |
| `backend_unavailable` | The vault database or the company data store is unreachable (or refused the query). | Retry later; check the deployment. |
| `rag_unavailable` | `rag_search` cannot answer. The message starts with a reason: `qdrant_unreachable` (connection refused, or no answer within `MOF_RAG_QDRANT_TIMEOUT`; also the answer given at once while a failed collection check is remembered), `qdrant_auth` (API key refused), `qdrant_error` (another error from Qdrant), `collection_missing`, `collection_mismatch` (vector size or distance), `model_unavailable`, `model_integrity`, `search_refused` (Qdrant refused this one search, most often a filter on a field without a payload index: the other searches are served), `timeout` (the call's own budget, `MOF_RAG_CALL_TIMEOUT_S`, ended while its search was running), `busy` (it ended before one of the `MOF_RAG_MAX_CONCURRENCY` slots was free), `internal_error` (the search failed with an error the server does not expect; the log names its class). It never holds the host, the key, Qdrant's response or the engine's own error text. | Retry later for `qdrant_unreachable`, `qdrant_error`, `timeout` and `busy`; for `internal_error`, retry and report it if it lasts; for `search_refused`, retry with fewer filters (without the date bounds first); the others need the deployment fixed. |
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
without details (they may hold connection strings); the server log has the rest. `rag_search` is the
exception: every error of its search is answered with a coded `[rag_unavailable]` reason, so nothing of
an exception reaches the framework (see Privacy in [Passage search](#passage-search-rag)). Over HTTP, a missing or
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
| `MOF_COMPANIES_CATALOG` | unset | open-data-store catalogue connection string as the read-only `lake_reader` role (`postgres:dbname=<db> host=<host> port=5432 user=lake_reader password=<password>`); unset = no companies or datasets toolsets. Needs `MOF_COMPANIES_DATA_ROOT` too. |
| `MOF_COMPANIES_DATA_ROOT` | unset | Folder holding `lake/` as this host sees it; read-only is enough. Needs `MOF_COMPANIES_CATALOG` too. |
| `MOF_COMPANIES_QUERY_TIMEOUT_S` | `20` | Seconds one companies tool call may spend on all its store reads before it answers `[timeout]` (1-3600). |
| `MOF_COMPANIES_MAX_COMPANIES` | `10` | Most companies a single call may name (1-100). |
| `MOF_COMPANIES_MAX_CONCURRENCY` | `2` | Store reads running at once (1-16); a read abandoned at its time budget keeps its slot until it ends, so the next callers may also get `[timeout]` for a while. |
| `MOF_COMPANIES_Q4_RULE` | `ytd9` | How TTM derives a fourth quarter the company did not report on its own: `ytd9` = FY minus the 9-month year-to-date (the rule of company-corpus's published outputs; a TTM needing a 9-month value the store lacks is absent), `sum_of_quarters` = FY minus Q1, Q2 and Q3 (present whenever those three quarters are, some differ from company-corpus's). |
| `MOF_RAG_QDRANT_HOST` | unset | Qdrant host name or address, alone: a scheme, a path or a port (`qdrant:6333`) is refused at start, the port being `MOF_RAG_QDRANT_PORT`. An IPv6 address is given bare or in brackets. Unset = no rag toolset. Needs `MOF_RAG_COLLECTION` and `MOF_RAG_MODEL_CACHE` too. |
| `MOF_RAG_QDRANT_PORT` | `6333` | Qdrant HTTP port (1-65535). |
| `MOF_RAG_QDRANT_API_KEY` | unset | Qdrant API key; a **read-only** key is enough and recommended. Never logged, shown or put in an error. |
| `MOF_RAG_QDRANT_HTTPS` | `false` | Connect to Qdrant over TLS (`true` / `false`). Without TLS the key travels in clear: private network only. |
| `MOF_RAG_QDRANT_TIMEOUT` | `10` | Seconds **one request** to Qdrant may take (1-300, at most `MOF_RAG_CALL_TIMEOUT_S`); a call makes several (collection check after a failure, read of the filter values when due, read of the payload schema for a date bound in `auto` mode when due, search). A request that takes longer fails, and the call that made it answers `[rag_unavailable] qdrant_unreachable`. |
| `MOF_RAG_CALL_TIMEOUT_S` | `15` | Time budget of one `rag_search` call in seconds, its wait for a free slot included (1-3600, at least `MOF_RAG_QDRANT_TIMEOUT`). At its end the call answers `[rag_unavailable]` with `timeout` or `busy`. |
| `MOF_RAG_MAX_CONCURRENCY` | `4` | Searches running at once (1-64), in threads of the toolset's own. The calls in excess wait without holding a thread. |
| `MOF_RAG_CHECK_RETRY_S` | `5` | Seconds a failed collection check is remembered (1-3600): calls made meanwhile answer `[rag_unavailable]` at once. Also the longest the toolset takes to notice that Qdrant is back. A failed search is not remembered: it makes the next call run the check. |
| `MOF_RAG_COLLECTION` | unset | Collection searched (768-dimension cosine vectors of `intfloat/multilingual-e5-base`). Needs `MOF_RAG_QDRANT_HOST` too. |
| `MOF_RAG_MODEL_CACHE` | unset | Directory holding the embedding model files (about 1.1 GB), as an absolute path. Required with the rag toolset, for every transport: the model is never kept in a temporary directory. Fill it once, then mount it read-only. The process needs about 2 GB of memory while the model loads: give a container at least 3 GB. |
| `MOF_RAG_ONNX_THREADS` | unset | Threads of the ONNX runtime (1-256). Unset = one per core of the host; in a container with a CPU limit, set it to the limit. |
| `MOF_RAG_DEFAULT_LIMIT` | `8` | Passages `rag_search` returns when the caller gives no `limit`; at most `MOF_RAG_MAX_LIMIT`. |
| `MOF_RAG_MAX_LIMIT` | `20` | Largest `limit` a caller may ask `rag_search` for (1-1000); a larger one is refused with `[limit_exceeded]`. |
| `MOF_RAG_CODES_TTL_S` | `300` | Seconds the collection's filter values (source codes, doc types, languages) are kept before being read again (1-86400). |
| `MOF_RAG_CODES_RETRY_S` | `5` | Least seconds between two reads of the filter values that the TTL did not ask for: the read made before a value is refused, and the retry after a failed read (1-3600). |
| `MOF_RAG_DESCRIPTION_NOTES` | unset | One line of text appended to the description of `rag_search`: what the collection holds, what its codes mean (at most 1500 characters). See [Passage search](#passage-search-rag) for a suggested value. |
| `MOF_RAG_DATE_FILTER_MODE` | `auto` | How a date bound of `rag_search` is applied (rag-engine's `date_filter_mode`): `auto` (the intervals on a collection where `date_start` and `date_end` both have a datetime index, the `date` field elsewhere), `interval` (always the intervals; a collection in strict mode without the two indexes refuses it: `search_refused`) or `legacy` (always the `date` field). Lower case, exactly. See [Dates](#dates-in-rag_search). |
| `MOF_RAG_DATE_FILTER_REFRESH_S` | `300` | `auto` mode: seconds the collection's answer (are the two indexes there?) is kept before a date-bounded call reads it again (0-86400; `0` reads it before every date-bounded search). The longest the server takes to use the intervals once the indexes exist, without a restart. |
| `MOF_RAG_DATE_FILTER_RETRY_S` | `30` | `auto` mode: seconds before a collection whose answer could not be read is read again (1-3600). Meanwhile date-bounded calls use the last answer read, or `legacy` when none was. Such a failure refuses no call and is not remembered as an outage. |
| `MOF_COMPANIES_SOURCE` | `auto` | Where `companies_*` read each company: `auto` (per company, for the call's `as_of`: the store when it holds the company, SEC's latest report filing accepted by then and the statement positions of SEC's latest 10-K/10-Q/20-F/40-F accepted by then; else SEC; each row says `source` and, when relevant, `reason`; when SEC cannot be asked (no contact, or SEC down) store companies are served with `reason: freshness_unchecked`; without a store, SEC only and no `datasets_*` tools), `store` (needs `MOF_COMPANIES_CATALOG`) or `sec` (needs `MOF_SEC_CONTACT`). The companies toolset is mounted when a store or a contact is configured. Needs the `[companies]` extra, which now also installs `markitdown` and `magika` (document conversion for live filings). |
| `MOF_SEC_CONTACT` | unset | E-mail or name sent in the User-Agent of every request to SEC, as their fair-access rules ask. Printable ASCII, at most 200 characters. Never logged, shown or put in an error. |
| `MOF_SEC_MAX_RPS` | `5.0` | Requests per second to SEC (above 0, at most 10, SEC's own limit). |
| `MOF_SEC_TIMEOUT_S` | `30` | Seconds one request to SEC may take (1-600). |
| `MOF_SEC_RETRIES` | `4` | Retries of a failed request to SEC (0-20). |
| `MOF_SEC_CALL_BUDGET_S` | `50` | Time budget of one tool call that fetches from SEC, in seconds (1-3600). Past it the call answers that the company is still being fetched; the fetch goes on and a later call gets the result. |
| `MOF_SEC_POSITIONS_LOOKBACK_Y` | `5` | Years before now (before `period_to` when given) whose facts get their filings' statement positions fetched (three requests per filing) when a call names no period start (1-50); a call with `period_from` starts there instead. Either start is then moved back by `MOF_SEC_TTM_LOOKBACK_M` months. |
| `MOF_SEC_POSITIONS_DONOR_DEPTH` | `4` | Earlier report filings (10-K, 10-Q, 20-F, 40-F) looked through, newest first, for the statement positions a filing without its own borrows, as the store's concept selection does (0-40; three requests each). A filing whose donor lies deeper does not read the store's donor: its location reads `unknown` where the store reads `inferred`, or it borrows from an older filing the call reads anyway. |
| `MOF_SEC_TTM_LOOKBACK_M` | `20` | Months before the period's start (`period_from`, or the start `MOF_SEC_POSITIONS_LOOKBACK_Y` gives a call without one) whose facts' filings also get their statement positions, so the TTM, derived fourth quarters and prior-year averages of the period's ratios are the store's (0-120). A quarterly TTM spans four quarter ends up to 300 days back; its earliest quarter, when derived, reads a 9-month YTD up to 100 days earlier (`ytd9`: under 400 days, 14 months) or three quarters up to 300 days earlier (`sum_of_quarters`: under 600 days, 20 months); the prior-year averages read up to 385 days back. The default covers both Q4 rules; below 14 (or 20 with `sum_of_quarters`) a ratio may differ from the store's. |
| `MOF_SEC_MEMO_COMPANIES` | `50` | Companies whose live tables (computed from SEC) are kept in memory for `MOF_CACHE_TTL_COMPANYFACTS_H`; past it the oldest built is dropped (1-10000). |
| `MOF_SEC_FAILURE_TTL_S` | `60` | Seconds a failed fetch of a company from SEC (SEC unreachable, refused, unreadable answer) is remembered: calls meanwhile get its error at once, even when it came after their time budget, instead of fetching again (1-3600). |
| `MOF_SEC_FRESHNESS_BUDGET_S` | `5` | `MOF_COMPANIES_SOURCE=auto` with a store: seconds a call gives the check of the store against SEC's filing list (one cached request per company, at most `MOF_COMPANIES_MAX_CONCURRENCY` at once). A company not checked by then is answered by the store with `reason: freshness_unchecked`, as are all of them for `MOF_SEC_FAILURE_TTL_S` after SEC failed during a check (1-600). |
| `MOF_CACHE` | `on` | Keep fetched data between calls: `on` or `off` (`off`: entries live for the process only). |
| `MOF_CACHE_DIR` | platform cache folder + `/myopenfund-mcp` | Absolute path (`~` is not expanded). Default: `$XDG_CACHE_HOME` or `~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows. |
| `MOF_CACHE_MAX_GB` | `20` | Size of the cached HTTP bodies above which the oldest-written ones are removed (1-100000). The model files and the local index sit under the same folder, on top of this figure, and are never removed. |
| `MOF_CACHE_EVICT_TO` | `0.9` | Fraction of `MOF_CACHE_MAX_GB` the cache is shrunk to when it is over the limit (above 0, at most 1). |
| `MOF_CACHE_TMP_GRACE_S` | `3600` | Seconds a temporary file of an interrupted write is kept before eviction removes it (60-86400). |
| `MOF_CACHE_TTL_SUBMISSIONS_H` | `24` | Hours a cached filing list is reused (1-8760). |
| `MOF_CACHE_TTL_COMPANYFACTS_H` | `24` | Hours a cached company-facts file is reused (1-8760). |
| `MOF_FILINGS_MAX_DOCS` | `40` | Filings downloaded and indexed by one call (1-1000). |
| `MOF_INDEX_WORKERS` | `1` | Filings indexed in parallel (1-32). |
| `MOF_RAG_ONNX_PROVIDER` | `cpu` | ONNX execution provider of the local embedding: `cpu` or `coreml`. |
| `MOF_SEARCH_RRF_K` | `60` | Constant k of the reciprocal-rank fusion of a filings search (1-1000). |
| `MOF_SEARCH_CANDIDATES` | `50` | Passages each retriever hands to the fusion (1-1000). |
| `MOF_SEARCH_EIGENMIND` | `false` | Add the graph (eigenmind) retriever to a filings search. Off until an evaluation decides. Needs the `[filings]` extra. |
| `MOF_FILINGS_DEFAULT_FORMS` | `10-K` | Comma-separated form types fetched when the caller names none. |
| `MOF_CHUNK_MAX_TOKENS` | `480` | Largest passage of the local filings index, in tokens (16-8192). |
| `MOF_CHUNK_OVERLAP_TOKENS` | `32` | Tokens shared by neighbouring passages (0-2048, below `MOF_CHUNK_MAX_TOKENS`). |

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
.venv/bin/pip install -e ".[vault,companies,rag,filings,dev]"
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

The default test run needs no database, network or model. Opt-in suites are excluded by default and run
with an explicit marker:

- `pytest -m integration`: against a real vault database, with `MOF_MCP_IT_DATABASE_URL` set.
- `pytest -m integration tests/integration/test_real_store.py`: against a real company store, with `MOF_MCP_IT_COMPANIES_CATALOG` and `MOF_MCP_IT_COMPANIES_DATA_ROOT` set (optional: `MOF_MCP_IT_COMPANIES_TIMEOUT_S`, default 120; `MOF_MCP_IT_ASSERT_LATENCY=1` to enforce the 5 s criterion, on the NAS only; add `-s` or `-rP` to see the printed fundamentals latency).
- `pytest -m integration tests/integration/test_real_qdrant.py`: `rag_search` against a **throwaway** Qdrant (never a production one: the test creates and deletes collections), with `MOF_MCP_IT_QDRANT_URL` set (e.g. `http://127.0.0.1:6333`). The test creates, seeds (through rag-engine, with its model-free `FakeEmbedder`) and deletes collections named `mof-mcp-it-*`. Optional, for a Qdrant with keys: `MOF_MCP_IT_QDRANT_API_KEY` (can write) and `MOF_MCP_IT_QDRANT_READ_ONLY_API_KEY` (given to the server).
- `pytest -m llm`: a scripted session with a local model, with `MOF_MCP_IT_OLLAMA_URL` set.

Build the image with `docker build -t myopenfund-mcp .`; `docker run --rm myopenfund-mcp myopenfund-mcp
--help` lists the commands.

## License

MIT, see [LICENSE](LICENSE).
