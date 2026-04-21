/**
 * Runtime smoke test for the generated client. Uses msw to stub a
 * SOMA server and exercises the happy path + auth + error branches.
 *
 * Covers:
 *  - POST /store -> POST /retrieve round-trip with a typed hit.
 *  - 401 ErrorResponse carries a typed `detail` string.
 *  - Bearer token from `createClient({ token })` reaches the wire.
 *  - `apiKey` fallback also produces a Bearer header.
 *  - Open mode (no token, no apiKey) omits the Authorization header.
 */
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";
import { createClient } from "../src/index";

const BASE = "http://soma.test";
const sentHeaders: { last?: Headers } = {};

const STORE_ID = "node-abc123";
const STORED_TEXT = "Paris is the capital of France.";

const handlers = [
  http.post(`${BASE}/store`, async ({ request }) => {
    sentHeaders.last = request.headers;
    const body = (await request.json()) as { text: string };
    // Pretend we persisted the text; the retrieve handler below echoes
    // back whatever the store handler last saw so the round-trip test
    // can assert end-to-end parity.
    lastStoredText = body.text;
    return HttpResponse.json({ node_id: STORE_ID });
  }),
  http.post(`${BASE}/retrieve`, async ({ request }) => {
    sentHeaders.last = request.headers;
    const body = (await request.json()) as { query: string; k: number };
    return HttpResponse.json({
      hits: [
        {
          node_id: STORE_ID,
          text: lastStoredText ?? STORED_TEXT,
          score: 0.91,
          metadata: { q: body.query },
          timestamp_step: 0,
        },
      ],
    });
  }),
  // 401 path — a protected route that returns the typed ErrorResponse.
  http.get(`${BASE}/status`, ({ request }) => {
    sentHeaders.last = request.headers;
    if (!request.headers.get("authorization")) {
      return HttpResponse.json(
        { detail: "invalid or missing Authorization header" },
        { status: 401 },
      );
    }
    return HttpResponse.json({
      num_entries: 0,
      bundle_path: "/data/memory",
      embed_model: "stub",
    });
  }),
];

let lastStoredText: string | undefined;
const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  server.resetHandlers();
  sentHeaders.last = undefined;
  lastStoredText = undefined;
  server.use(...handlers);
});
afterAll(() => server.close());

describe("soma-memory runtime", () => {
  it("store then retrieve round-trip returns the same text", async () => {
    const soma = createClient({ baseUrl: BASE, token: "t1" });
    const store = await soma.POST("/store", {
      body: { text: STORED_TEXT, metadata: {} },
    });
    expect(store.error).toBeUndefined();
    expect(store.data?.node_id).toBe(STORE_ID);

    const retrieve = await soma.POST("/retrieve", {
      body: { query: "capital?", k: 3 },
    });
    expect(retrieve.error).toBeUndefined();
    expect(retrieve.data?.hits).toHaveLength(1);
    expect(retrieve.data?.hits[0]?.text).toBe(STORED_TEXT);
    expect(retrieve.data?.hits[0]?.node_id).toBe(STORE_ID);
  });

  it("401 response has typed detail string", async () => {
    const soma = createClient({ baseUrl: BASE });
    const res = await soma.GET("/status", {});
    expect(res.data).toBeUndefined();
    expect(res.response.status).toBe(401);
    expect(res.error).toBeDefined();
    expect((res.error as { detail: string }).detail).toContain(
      "Authorization",
    );
  });

  it("attaches Authorization: Bearer <token> when token is set", async () => {
    const soma = createClient({ baseUrl: BASE, token: "jwt-abc" });
    await soma.POST("/store", { body: { text: "x", metadata: {} } });
    expect(sentHeaders.last?.get("authorization")).toBe("Bearer jwt-abc");
  });

  it("falls back to apiKey when token is absent", async () => {
    const soma = createClient({ baseUrl: BASE, apiKey: "legacy-key" });
    await soma.POST("/store", { body: { text: "x", metadata: {} } });
    expect(sentHeaders.last?.get("authorization")).toBe("Bearer legacy-key");
  });

  it("prefers token over apiKey when both are set", async () => {
    const soma = createClient({
      baseUrl: BASE,
      token: "jwt-wins",
      apiKey: "legacy-loses",
    });
    await soma.POST("/store", { body: { text: "x", metadata: {} } });
    expect(sentHeaders.last?.get("authorization")).toBe("Bearer jwt-wins");
  });

  it("open mode omits the Authorization header entirely", async () => {
    const soma = createClient({ baseUrl: BASE });
    await soma.POST("/store", { body: { text: "x", metadata: {} } });
    expect(sentHeaders.last?.get("authorization")).toBeNull();
  });

  it("merges caller-provided extra headers", async () => {
    const soma = createClient({
      baseUrl: BASE,
      token: "t",
      headers: { "X-Trace-Id": "trace-42" },
    });
    await soma.POST("/store", { body: { text: "x", metadata: {} } });
    expect(sentHeaders.last?.get("authorization")).toBe("Bearer t");
    expect(sentHeaders.last?.get("x-trace-id")).toBe("trace-42");
  });
});
