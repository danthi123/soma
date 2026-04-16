# Phase 5 — TypeScript client via OpenAPI

> **For Claude:** Phase 5a (schema polish) already landed in commit `7f03875`. This plan covers Phase 5b: generator choice, package shape, CI workflow, and first publish. Execute after Phase 4 (JWT) lands so the client gets the final security scheme metadata.

**Goal:** `@soma-ai/client` on npm — a 6 KB TypeScript client generated from `GET /openapi.json`. Works in Node 18+, browsers, Deno, Bun, Cloudflare Workers. Full type inference on request/response shapes. CI publishes on tag push.

**Architecture:**
- **`openapi-typescript`** (types-only `schema.d.ts`) + **`openapi-fetch`** (6 KB runtime client). Zero codegen runtime, no classes, fetch-native.
- **Scoped npm package** `@soma-ai/client`. Unscoped alias `soma-memory-client` points at it via `deprecate`.
- **CI workflow** spins up uvicorn with `SOMA_EMBED_MODEL=stub`, snapshots `/openapi.json` into `clients/typescript/openapi.json`, regenerates `schema.d.ts`, typechecks, tests with `msw`, publishes on tag.
- **Auth injection** via `createClient` header option or `openapi-fetch` middleware — no fork of generated code.

**Tech stack:** Node 20, `openapi-typescript>=7.13`, `openapi-fetch`, `vitest`, `msw`.

---

### Task 1: `clients/typescript/` package skeleton

**Files:**
- Create: `clients/typescript/package.json`
- Create: `clients/typescript/tsconfig.json`
- Create: `clients/typescript/src/index.ts`
- Create: `clients/typescript/src/schema.d.ts` (placeholder for generator output)
- Create: `clients/typescript/.gitignore` (node_modules, dist)
- Create: `clients/typescript/README.md`

**Step 1:** package.json shape:
```json
{
  "name": "@soma-ai/client",
  "version": "0.1.0",
  "description": "TypeScript client for SOMA — local-first agent memory layer",
  "type": "module",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js"
    }
  },
  "files": ["dist", "README.md"],
  "scripts": {
    "generate": "openapi-typescript openapi.json -o src/schema.d.ts",
    "build": "tsc",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "prepublishOnly": "npm run generate && npm run build"
  },
  "dependencies": {
    "openapi-fetch": "^0.13.0"
  },
  "devDependencies": {
    "openapi-typescript": "^7.13.0",
    "typescript": "^5.4.0",
    "vitest": "^2.0.0",
    "msw": "^2.0.0"
  }
}
```

**Step 2:** `src/index.ts`:
```typescript
import createOpenApiClient from "openapi-fetch";
import type { paths } from "./schema";

export type SomaClientOptions = {
  baseUrl: string;
  apiKey?: string;      // legacy SOMA_API_KEY mode
  token?: string;       // JWT mode (preferred)
  fetch?: typeof fetch; // custom transport (retries, logging)
};

export function createClient(options: SomaClientOptions) {
  const auth = options.token ?? options.apiKey;
  return createOpenApiClient<paths>({
    baseUrl: options.baseUrl,
    headers: auth ? { Authorization: `Bearer ${auth}` } : {},
    fetch: options.fetch,
  });
}

export type { paths } from "./schema";
```

**Step 3:** commit `feat(client-ts): package skeleton for @soma-ai/client`.

### Task 2: Generator run + typecheck against the live SOMA server

**Files:**
- Create: `clients/typescript/openapi.json` (snapshot; CI-committed)
- Generate: `clients/typescript/src/schema.d.ts`
- Test: `clients/typescript/tests/types.test.ts`

**Step 1:** `uvicorn soma.serve:app --port 8420` with `SOMA_EMBED_MODEL=stub`. `curl http://127.0.0.1:8420/openapi.json | jq . > clients/typescript/openapi.json`.

**Step 2:** `npm run generate` — produces `schema.d.ts`. `npm run typecheck` — zero errors.

**Step 3:** `tests/types.test.ts` (compile-time, not runtime):
```typescript
import { expectTypeOf } from "vitest";
import type { paths } from "../src/schema";

// Validate a subset of paths resolve with correct types.
type StoreReq = paths["/store"]["post"]["requestBody"]["content"]["application/json"];
expectTypeOf<StoreReq>().toHaveProperty("text");
```

**Step 4:** commit `feat(client-ts): snapshot openapi.json + generated schema.d.ts`.

### Task 3: Runtime smoke test with msw

**Files:**
- Create: `clients/typescript/tests/smoke.test.ts`

**Step 1:** msw mocks the SOMA server. Tests:
- `store then retrieve roundtrip` — POST /store → GET the same text back from POST /retrieve.
- `error response has typed detail` — server returns 401 → client's error path has `.detail: string`.
- `bearer token attached` — msw intercepts, asserts `Authorization: Bearer ...` header present.
- `no-auth mode works` — createClient without `token/apiKey` omits the header.

**Step 2:** implement msw handlers against the snapshotted `openapi.json`.

**Step 3:** commit `test(client-ts): msw runtime smoke for store/retrieve/error paths`.

### Task 4: CI workflow

**Files:**
- Create: `.github/workflows/client-ts.yml`

**Step 1:** workflow:
```yaml
name: client-ts
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    paths: ['src/soma/serve.py', 'clients/typescript/**']

jobs:
  build-publish:
    runs-on: ubuntu-latest
    permissions: { contents: read, id-token: write }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - uses: actions/setup-node@v4
        with: { node-version: '20', registry-url: 'https://registry.npmjs.org' }
      - name: Install SOMA
        run: pip install -e ".[serve]"
      - name: Launch server with stub embed
        run: |
          SOMA_EMBED_MODEL=stub uvicorn soma.serve:app --port 8420 &
          for i in {1..30}; do curl -sf http://127.0.0.1:8420/health && break; sleep 1; done
      - name: Snapshot OpenAPI
        run: curl -sf http://127.0.0.1:8420/openapi.json | jq . > clients/typescript/openapi.json
      - name: Spec drift check (PRs)
        if: github.event_name == 'pull_request'
        run: git diff --exit-code clients/typescript/openapi.json
      - name: Generate + test
        working-directory: clients/typescript
        run: |
          npm ci
          npm run generate
          npm run typecheck
          npm test
      - name: Publish on tag
        if: startsWith(github.ref, 'refs/tags/v')
        working-directory: clients/typescript
        run: npm publish --access public --provenance
        env: { NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }} }
```

**Step 2:** commit `ci(client-ts): generate + test + publish on tag`.

### Task 5: Claim `@soma-ai` scope + publish 0.1.0 placeholder

**Files:** (npm org operation; no repo change beyond docs)

**Step 1:** manual step — operator runs `npm org create soma-ai` (requires npm login). If scope taken, fall back to `@soma-memory` or `@soma-ml`.

**Step 2:** document the scope claim in `docs/clients.md` (new).

**Step 3:** Push a tag `v0.1.0`; CI publishes.

### Task 6: Docs + README

**Files:**
- Create: `docs/clients.md`
- Modify: `README.md` — add TS section under "REST API / Docker"
- Modify: `CHANGELOG.md`

**Step 1:** README snippet:
```markdown
## TypeScript client

```bash
npm install @soma-ai/client
```

```ts
import { createClient } from "@soma-ai/client";

const soma = createClient({
  baseUrl: "http://localhost:8420",
  token: process.env.SOMA_TOKEN,
});

await soma.POST("/store", { body: { text: "Paris is the capital of France." } });
const { data } = await soma.POST("/retrieve", { body: { query: "capital?", k: 3 } });
console.log(data?.hits);
```
```

**Step 2:** `docs/clients.md` — install, auth (JWT vs legacy), usage examples, error handling, retry middleware, browser vs Node notes.

**Step 3:** commit `docs: Phase 5 — TypeScript client publish + usage`.

---

## Risks

1. **npm scope squatting.** Claim `@soma-ai` before first publish. Script: `npm publish @soma-ai/client@0.0.0 --dry-run` first to verify availability.
2. **Spec drift without CI guard.** Without the PR-check step, serve.py edits silently change the client API. The `git diff --exit-code` on `openapi.json` in PRs prevents this.
3. **Stub embedder accidentally loads sbert.** Test matrix in Phase 5a already covers `SOMA_EMBED_MODEL=stub`; confirm the CI job respects it.
4. **Client auth header vs Phase 4 JWT.** Ensure both `token` (JWT) and `apiKey` (legacy) modes work; document the preference order.
5. **msw version churn.** msw v2 has breaking changes from v1. Pin major version in package.json.

## Related plans

- Phase 4 — `docs/plans/2026-04-16-phase-4-jwt-auth.md` (OpenAPI security scheme; this client consumes it)
- Phase 5a — landed in commit `7f03875` (serve.py schema polish: CORS, HTTPBearer, ErrorResponse, operation_ids, tags, field descriptions, stub embedder)
