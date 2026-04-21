# soma-memory (TypeScript client)

TypeScript client for [SOMA](https://github.com/danthi123/soma) — a
local-first agent-memory layer with hybrid retrieval.

Thin wrapper over [`openapi-fetch`](https://openapi-ts.dev/openapi-fetch/)
with types generated from the live SOMA `/openapi.json`. Works in
**Node 18+**, **browsers**, **Deno**, **Bun**, and **Cloudflare Workers**.

## Install

```bash
npm install soma-memory
```

## Quick start

```ts
import { createClient } from "soma-memory";

const soma = createClient({
  baseUrl: "http://localhost:8420",
  token: process.env.SOMA_TOKEN, // minted by `soma auth issue`
});

await soma.POST("/store", {
  body: { text: "Paris is the capital of France." },
});

const { data, error } = await soma.POST("/retrieve", {
  body: { query: "capital?", k: 3 },
});

if (error) {
  console.error(error.detail);
} else {
  console.log(data?.hits);
}
```

Every path, request body, and response type is inferred from
`openapi.json` — your editor autocompletes the full SOMA API.

## Auth

SOMA supports three auth modes; the client papers over all three with
one option:

| Option    | Header sent                    | Server mode             |
| --------- | ------------------------------ | ----------------------- |
| `token`   | `Authorization: Bearer <JWT>`  | JWT (`SOMA_JWT_SECRET`) |
| `apiKey`  | `Authorization: Bearer <key>`  | Legacy `SOMA_API_KEY`   |
| *none*    | *(omitted)*                    | Open-mode dev           |

`token` wins when both are set. See
[`docs/auth.md`](../../docs/auth.md) for JWT claim shape, rotation,
and the legacy deprecation timeline.

## Multi-tenant bundles

```ts
await soma.POST("/bundles/{name}/store", {
  params: { path: { name: "alex-brain" } },
  body: { text: "user is vegetarian" },
});
```

## Browser / edge runtimes

`openapi-fetch` uses the platform `fetch`. Browsers and Workers work
out of the box; Node 18+ ships `fetch` natively. For older Node or
custom retry/logging, pass a `fetch` override:

```ts
import createRetry from "fetch-retry";
const retrying = createRetry(fetch, { retries: 3 });

const soma = createClient({
  baseUrl: "http://localhost:8420",
  token: process.env.SOMA_TOKEN,
  fetch: retrying,
});
```

## Development

```bash
npm ci
npm run generate    # regenerate src/schema.d.ts from openapi.json
npm run typecheck
npm test
npm run build
```

The `openapi.json` snapshot in this directory is produced by CI from
a live `uvicorn soma.serve:app` process with `SOMA_EMBED_MODEL=stub`
(no sbert download). PRs that change `src/soma/serve.py` re-snapshot
the spec and fail if the committed copy has drifted.

## Publishing

CI publishes `soma-memory` on `v*` tag push via
`.github/workflows/client-ts.yml`. See `docs/clients.md` for the
release checklist. The npm package name matches the PyPI distribution
name (`soma-memory`) for cross-language consistency; no scope claim is
required since the unscoped `soma-memory` name is available on npm.

## License

MIT
