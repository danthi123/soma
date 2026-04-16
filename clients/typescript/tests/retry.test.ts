/**
 * Tests for the `withRetry` fetch wrapper.
 *
 * Covers the contract from `docs/plans/2026-04-16-phase-15-polish-audit.md`:
 * retries on 503, gives up after `maxRetries`, respects `retryOn`,
 * doesn't retry on 4xx by default, and rethrows the last network error
 * when every attempt fails.
 */
import { describe, expect, it, vi } from "vitest";
import { withRetry } from "../src/retry";

// Sleeps in tests are no-ops — we assert call counts, not timing.
const instantSleep = (_ms: number): Promise<void> => Promise.resolve();

function okResponse(body: unknown = { ok: true }, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("withRetry", () => {
  it("returns a successful response on the first try without retrying", async () => {
    const fetchImpl = vi.fn(async () => okResponse());
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(200);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("retries on 503 and succeeds on the second attempt", async () => {
    let attempt = 0;
    const fetchImpl = vi.fn(async () => {
      attempt += 1;
      if (attempt === 1) {
        return okResponse({ detail: "unavailable" }, 503);
      }
      return okResponse({ ok: true });
    });
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 3,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(200);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("gives up after maxRetries and returns the last (still-failing) response", async () => {
    const fetchImpl = vi.fn(async () =>
      okResponse({ detail: "bad gateway" }, 502),
    );
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 3,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    // 1 initial try + 3 retries = 4 total calls.
    expect(fetchImpl).toHaveBeenCalledTimes(4);
    expect(res.status).toBe(502);
  });

  it("does NOT retry on 4xx by default", async () => {
    const fetchImpl = vi.fn(async () =>
      okResponse({ detail: "unauthorized" }, 401),
    );
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 3,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(401);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("respects a custom retryOn predicate", async () => {
    // Retry on 429 instead of the default 5xx set. A 503 should pass
    // straight through without retries.
    let attempt = 0;
    const fetchImpl = vi.fn(async () => {
      attempt += 1;
      if (attempt < 3) {
        return okResponse({ detail: "rate limited" }, 429);
      }
      return okResponse({ ok: true });
    });
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 5,
      retryOn: (res) => res.status === 429,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(200);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("custom retryOn lets 503 pass through when not in its set", async () => {
    const fetchImpl = vi.fn(async () => okResponse({}, 503));
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 3,
      retryOn: (res) => res.status === 429,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(503);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("rethrows the last error when every attempt throws a network error", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("ECONNRESET");
    });
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 2,
      sleep: instantSleep,
    });

    await expect(fetchWithRetry("http://x.test/y")).rejects.toThrow(
      "ECONNRESET",
    );
    // 1 initial + 2 retries = 3 calls.
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  it("recovers when a network error is followed by success", async () => {
    let attempt = 0;
    const fetchImpl = vi.fn(async () => {
      attempt += 1;
      if (attempt === 1) {
        throw new TypeError("ECONNRESET");
      }
      return okResponse();
    });
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 3,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(200);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("sleeps with exponential backoff by default between retries", async () => {
    const delays: number[] = [];
    const sleep = async (ms: number): Promise<void> => {
      delays.push(ms);
    };
    let attempt = 0;
    const fetchImpl = vi.fn(async () => {
      attempt += 1;
      if (attempt < 4) return okResponse({}, 503);
      return okResponse();
    });
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 5,
      initialDelayMs: 100,
      sleep,
    });
    await fetchWithRetry("http://x.test/y");
    // After 3 failing attempts: delays should be 100, 200, 400 (exponential).
    expect(delays).toEqual([100, 200, 400]);
  });

  it("sleeps with linear backoff when configured", async () => {
    const delays: number[] = [];
    const sleep = async (ms: number): Promise<void> => {
      delays.push(ms);
    };
    let attempt = 0;
    const fetchImpl = vi.fn(async () => {
      attempt += 1;
      if (attempt < 4) return okResponse({}, 503);
      return okResponse();
    });
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 5,
      backoff: "linear",
      initialDelayMs: 100,
      sleep,
    });
    await fetchWithRetry("http://x.test/y");
    // Linear: 100, 200, 300.
    expect(delays).toEqual([100, 200, 300]);
  });

  it("returns the raw response when maxRetries is 0 (no retries)", async () => {
    const fetchImpl = vi.fn(async () => okResponse({}, 503));
    const fetchWithRetry = withRetry(fetchImpl as unknown as typeof fetch, {
      maxRetries: 0,
      sleep: instantSleep,
    });

    const res = await fetchWithRetry("http://x.test/y");
    expect(res.status).toBe(503);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});
