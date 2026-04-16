/**
 * Optional retry middleware for the SOMA fetch layer.
 *
 * Wraps any `fetch`-compatible implementation and returns a new
 * function with the same signature that transparently retries
 * transient failures. Typical usage:
 *
 *     import { createClient, withRetry } from "@soma-ai/client";
 *
 *     const soma = createClient({
 *       baseUrl: "http://localhost:8420",
 *       token: process.env.SOMA_TOKEN,
 *       fetch: withRetry(fetch, { maxRetries: 5 }),
 *     });
 *
 * Design notes:
 *
 * - The wrapper is standalone on purpose — it does NOT fork or modify
 *   the generated `openapi-fetch` code. That keeps regeneration
 *   painless and leaves retry policy the caller's decision.
 * - Defaults retry only on 502 / 503 / 504 — the statuses a proxy or
 *   server sends when it's temporarily unable to serve the request.
 *   4xx is never retried by default (the caller's request was the
 *   problem; repeating it does not change the outcome).
 * - Network errors (a thrown exception from `fetchImpl`) are also
 *   retried up to `maxRetries` times. The last error is rethrown if
 *   every attempt fails so callers see the root cause, not a generic
 *   "retries exhausted".
 * - `retryOn` lets callers override the default status predicate
 *   (e.g. add 429 with a short backoff if the server uses it).
 */

export interface RetryOptions {
  /** Maximum number of retry attempts AFTER the initial request. Default 3. */
  maxRetries?: number;
  /** Backoff curve. `"linear"` = delay * attempt; `"exponential"` = delay * 2^attempt. Default `"exponential"`. */
  backoff?: "linear" | "exponential";
  /** Base delay in milliseconds for the first retry. Default 200 ms. */
  initialDelayMs?: number;
  /**
   * Predicate that decides whether a response should be retried.
   * Default: retries on 502 / 503 / 504.
   */
  retryOn?: (response: Response) => boolean;
  /**
   * Optional sleep function. Defaults to `setTimeout`-based delay.
   * Exposed so tests can run without real timers.
   */
  sleep?: (ms: number) => Promise<void>;
}

const DEFAULT_RETRY_STATUSES = new Set([502, 503, 504]);

const defaultRetryOn = (res: Response): boolean =>
  DEFAULT_RETRY_STATUSES.has(res.status);

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

function computeDelay(
  attempt: number,
  initialDelayMs: number,
  backoff: "linear" | "exponential",
): number {
  if (backoff === "linear") {
    return initialDelayMs * attempt;
  }
  // exponential: initial * 2^(attempt-1) so attempt 1 = initial, 2 = 2*initial, ...
  return initialDelayMs * 2 ** (attempt - 1);
}

/**
 * Wrap a `fetch` implementation with retry behaviour.
 *
 * The returned function matches the standard `fetch` signature, so
 * it plugs into `createClient({ fetch: ... })` without any other
 * plumbing.
 */
export function withRetry(
  fetchImpl: typeof fetch,
  opts: RetryOptions = {},
): typeof fetch {
  const maxRetries = opts.maxRetries ?? 3;
  const backoff = opts.backoff ?? "exponential";
  const initialDelayMs = opts.initialDelayMs ?? 200;
  const retryOn = opts.retryOn ?? defaultRetryOn;
  const sleep = opts.sleep ?? defaultSleep;

  const wrapped: typeof fetch = async (input, init) => {
    let lastError: unknown;
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      try {
        const response = await fetchImpl(input, init);
        // Final attempt: return whatever we got, even if retryable.
        if (attempt === maxRetries) {
          return response;
        }
        if (!retryOn(response)) {
          return response;
        }
      } catch (err) {
        lastError = err;
        if (attempt === maxRetries) {
          throw err;
        }
      }
      const delay = computeDelay(attempt + 1, initialDelayMs, backoff);
      await sleep(delay);
    }
    // Unreachable — the loop always either returns or throws above.
    // Included for exhaustiveness in case maxRetries is < 0, in which
    // case we rethrow the last error or surface a clear message.
    if (lastError !== undefined) {
      throw lastError;
    }
    throw new Error("withRetry: exhausted retries without a response");
  };

  return wrapped;
}
