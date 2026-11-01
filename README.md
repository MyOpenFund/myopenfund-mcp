# myopenfund-mcp

One [MCP](https://modelcontextprotocol.io) server over the MyOpenFund bricks: official documents and
pipeline health today, RAG and company data next.

Any MCP client (a desktop assistant, a coding agent, your own agent, a local model) gets outcome-oriented
tools over MyOpenFund's data: find official primary-source documents, read their facts with provenance
(`doc_id`, dates, source URLs), count the corpus, check the ingestion pipeline's health. Toolsets are
enabled per installed stack: a toolset whose extra is not installed or whose environment variables are
not set is simply not mounted, and `server_info` says so.

> **MyOpenFund manages nobody's money and gives no investment advice.** This server returns documents and
> facts from official sources, with their provenance. What you do with them is your decision.

- Two transports, same code: **stdio** for local use, **Streamable HTTP** (bearer tokens) in a Docker image.
- Host-agnostic: the same image runs on a laptop, on a host on a private network or on a cloud VM;
  everything deployment-specific comes from `MOF_*` environment variables.
- Deterministic: no tool needs a hosted LLM (see [Local models](#local-models)).

## Contents

- [Quick start (stdio)](#quick-start-stdio)
- [HTTP deployment](#http-deployment)
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
| `vault_search_documents` | Find official documents (metadata, not text). Start here: filter, then open one with `vault_get_document`. Rows are ordered by `sort_by` in `sort_dir`; equal sort values are always ordered by `doc_id` ascending, whatever `sort_dir`. `total` counts the whole filtered set. | `corpus` (e.g. `central-bank`), `source_code` (publisher, e.g. `us` for the Fed), `doc_type`, `language`, `year`, `mime_type`, `date_from` / `date_to` (YYYY-MM-DD, inclusive), `text` (case-insensitive match on the title), `sort_by` (`date` default, `year`, `corpus`, `source_code`, `doc_type`, `title`, `created_at`), `sort_dir` (`desc` default), `limit`, `cursor`, `include_deleted` (also return withdrawn documents, with `deleted_at` set) |
| `vault_get_document` | Every stored fact of one document: dates, source and PDF URLs, publisher, type, `extra`. | `doc_id` |
| `vault_corpus_stats` | Live-document counts grouped by one field, largest first (NULL is a real group). | `group_by` (`source_code` default, `corpus`, `doc_type`, `language`, `year`, `provenance`, `mime_type`), `corpus`, `limit` |
| `vault_runs_status` | Is the pipeline healthy? Four lists, newest first: `runs` (one row per run, with outcome and totals; a run that crashed before reaching any source appears only here), `source_runs` (one row per run and source: counts, outcome, bounded error samples), `open_discovery_errors` (unresolved discovery failures; while `seen_at_is_ingestion_time` is true, `last_seen_at` is approximate), `source_health` (expected-vs-observed publication cadence per series). | `corpus`, `source_code`, `since` (ISO 8601, UTC if no offset; keeps runs *finished* at or after it, leaves out runs still in flight; applies to `runs` and `source_runs` only), `limit` (applies to each of `runs`, `source_runs`, `open_discovery_errors`) |

Resource: `vault://documents/{doc_id}`, one document's metadata as JSON (same content as
`vault_get_document`).

The `vault_*` tools and the resource exist only when `MOF_VAULT_DATABASE_URL` is set and the `[vault]`
extra is installed.

## Errors

Tool errors are MCP tool errors whose message starts with a stable code, `[code] message`, so agents can
branch on the code and show the message.

| Code | Meaning | What to do |
|---|---|---|
| `invalid_argument` | A bad argument: `limit` outside `[1, MOF_MCP_MAX_ROWS]`, a cursor this server did not issue, `date_from` after `date_to`, a value the vault database refuses. | Fix the argument; pass `next_cursor` unchanged. |
| `not_found` | No document with that `doc_id`. | Find ids with `vault_search_documents`. |
| `timeout` | The vault query exceeded `MOF_MCP_QUERY_TIMEOUT_S` (cancelled by Postgres). | Narrow the filters. |
| `backend_unavailable` | The vault database is unreachable or refused the query. | Retry later; check the deployment. |
| `result_too_large` | The result exceeds `MOF_MCP_MAX_RESULT_CHARS`. Results are refused, never silently cut. | Ask for a smaller `limit`. |

Arguments that do not match a tool's input schema (wrong type, string too long) are refused by schema
validation before the tool runs. An unexpected internal failure reaches the client as a generic error
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
| `MOF_MCP_MAX_RESULT_CHARS` | `200000` | Largest tool result in characters (at least 1000); larger results are refused. |
| `MOF_MCP_MAX_ERROR_SAMPLES` | `5` | Error samples kept per row of `vault_runs_status`. |
| `MOF_MCP_MAX_TEXT_CHARS` | `2000` | Each error text in `vault_runs_status` is cut to this length. |

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
- **v1.1**: `rag_*` (cited passage search over the embedded corpus, embedding as an `act` job) and
  `jobs_*` to follow long actions.
- **v2**: `companies_*` (identity resolution, filings fetched on demand, reported fundamentals) and
  `datasets_*` over the MyOpenFund data library.
- **Later**: `graph_*` and more sources.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[vault,dev]"
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
```

The default test run needs no database, network or model. Opt-in suites are excluded by default and run
with an explicit marker:

- `pytest -m integration`: against a real vault database, with `MOF_MCP_IT_DATABASE_URL` set.
- `pytest -m llm`: a scripted session with a local model, with `MOF_MCP_IT_OLLAMA_URL` set.

Build the image with `docker build -t myopenfund-mcp .`; `docker run --rm myopenfund-mcp myopenfund-mcp
--help` lists the commands.

## License

MIT, see [LICENSE](LICENSE).
