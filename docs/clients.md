# SOMA client libraries

SOMA's REST API (`uvicorn soma.serve:app`) is consumable from any
HTTP client. We additionally ship an official TypeScript client
generated from the server's own `/openapi.json` so downstream
TypeScript / JavaScript agents get full autocomplete on every
request body, response shape, and error branch.

## `@soma-ai/client` (TypeScript)

**Scope:** `@soma-ai/client` on npm. Works in Node 18+, browsers,
Deno, Bun, and Cloudflare Workers (`openapi-fetch` uses the platform
`fetch`).

### Install

```bash
npm install @soma-ai/client
```

### Usage

```ts
import { createClient } from "@soma-ai/client";

const soma = createClient({
  baseUrl: "http://localhost:8420",
  token: process.env.SOMA_TOKEN, // minted by `soma auth issue`
});

// Store
await soma.POST("/store", {
  body: { text: "Paris is the capital of France." },
});

// Retrieve
const { data, error } = await soma.POST("/retrieve", {
  body: { query: "capital?", k: 3 },
});
if (error) {
  console.error(error.detail); // typed as string
} else {
  for (const hit of data?.hits ?? []) {
    console.log(hit.score, hit.text);
  }
}
```

### Auth modes

| Option   | Header                        | Server mode             |
| -------- | ----------------------------- | ----------------------- |
| `token`  | `Authorization: Bearer <JWT>` | JWT (`SOMA_JWT_SECRET`) |
| `apiKey` | `Authorization: Bearer <key>` | Legacy `SOMA_API_KEY`   |
| *none*   | *(omitted)*                   | Open-mode dev           |

`token` wins when both are supplied. The legacy-key path triggers an
`X-SOMA-Deprecated: use JWT` response header — see
[`auth.md`](auth.md) for the JWT claim shape, rotation workflow, and
the deprecation timeline for `SOMA_API_KEY`.

### Multi-tenant bundles

```ts
await soma.POST("/bundles/{name}/store", {
  params: { path: { name: "alex-brain" } },
  body: { text: "user is vegetarian" },
});

const { data } = await soma.POST("/bundles/{name}/retrieve", {
  params: { path: { name: "alex-brain" } },
  body: { query: "diet?", k: 5 },
});
```

### Retry / logging middleware

Inject a custom `fetch` to layer in retries, structured logging, or a
platform-specific transport:

```ts
import createRetry from "fetch-retry";
const retrying = createRetry(fetch, { retries: 3, retryDelay: 200 });

const soma = createClient({
  baseUrl: "http://localhost:8420",
  token: process.env.SOMA_TOKEN,
  fetch: retrying,
});
```

### Browser notes

CORS is allow-listed to `http://localhost:*` by default. Override for
production origins with `SOMA_CORS_ORIGINS=https://app.example.com`
(comma-separated, `*` wildcards allowed).

### How the client is built

`@soma-ai/client` is a thin wrapper over
[`openapi-fetch`](https://openapi-ts.dev/openapi-fetch/). Types are
generated from `clients/typescript/openapi.json` — a snapshot of the
live SOMA server's `/openapi.json` taken with `SOMA_EMBED_MODEL=stub`
(no sbert download). The CI workflow
(`.github/workflows/client-ts.yml`) re-snapshots on every PR to
`src/soma/serve.py` and fails if the committed copy has drifted.

### Publishing

CI publishes on `v*` tag push. One-time operator setup:

1. Claim the npm scope: `npm org create soma-ai` (requires
   `npm login` with the `soma-ai` GitHub org owner's account).
2. If the scope is already taken, fall back to `@soma-memory` or
   `@soma-ml` — update `clients/typescript/package.json#name` + this
   doc + the README snippet together; no other call sites change.
3. Add `NPM_TOKEN` to the repo's GitHub Actions secrets (scope-level
   publish token from npm).
4. Tag and push: `git tag v0.1.0 && git push --tags`.

The first tag push triggers `.github/workflows/client-ts.yml` which
regenerates types, runs the full test suite, and publishes with
`--provenance` (sigstore-signed). No local publish is expected; skip
placeholder releases — jump straight to `v0.1.0` on the first real
server-API cut.

### Development

```bash
cd clients/typescript/
npm ci
npm run generate    # schema.d.ts from openapi.json
npm run typecheck
npm test            # vitest (types + msw smoke)
npm run build       # tsc -> dist/
```

The `openapi.json` snapshot is a committed file. To refresh it against
a local server change:

```bash
# from repo root
SOMA_EMBED_MODEL=stub python -m uvicorn soma.serve:app --port 8420 &
curl -sf http://127.0.0.1:8420/openapi.json \
  | python -m json.tool \
  > clients/typescript/openapi.json
kill %1
cd clients/typescript && npm run generate
```

## Python

Use `MemoryLayer` directly — `pip install soma` — or call the REST
API with `httpx` / `requests`. A generated Python client isn't
shipped today; the native `MemoryLayer` API is lower-overhead and
already covers the full surface.

## Other languages

The server ships `/openapi.json` at runtime — point any generator at
that URL (see [openapi.tools](https://openapi.tools/) for options) to
produce a Go, Rust, or Java client. Contributions welcome.
