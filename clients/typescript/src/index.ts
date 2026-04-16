/**
 * @soma-ai/client — TypeScript client for SOMA.
 *
 * Thin wrapper around `openapi-fetch` with typed paths generated from
 * the live SOMA `/openapi.json` spec. Works in Node 18+, browsers,
 * Deno, Bun, and Cloudflare Workers.
 *
 * Auth modes:
 *   - `token`  — preferred; a per-bundle JWT minted by `soma auth issue`.
 *   - `apiKey` — legacy `SOMA_API_KEY` admin escape hatch; the server
 *                responds with an `X-SOMA-Deprecated: use JWT` header.
 *
 * When neither is set the client sends no `Authorization` header, which
 * matches the server's open-mode default (no `SOMA_JWT_SECRET` / no
 * `SOMA_API_KEY` => every route is open).
 */
import createOpenApiClient from "openapi-fetch";
import type { paths } from "./schema";

export type SomaClientOptions = {
  /** Base URL of the SOMA server, e.g. `http://localhost:8420`. */
  baseUrl: string;
  /** Per-bundle JWT (preferred). Mints from `soma auth issue`. */
  token?: string;
  /** Legacy `SOMA_API_KEY` admin escape hatch. Prefer `token`. */
  apiKey?: string;
  /** Custom `fetch` implementation — e.g. for retries, logging, or
   *  platform-specific transports (undici, Cloudflare Workers). */
  fetch?: typeof fetch;
  /** Extra headers merged onto every request. */
  headers?: Record<string, string>;
};

/**
 * Build a fully typed SOMA client. Usage:
 *
 *     const soma = createClient({
 *       baseUrl: "http://localhost:8420",
 *       token: process.env.SOMA_TOKEN,
 *     });
 *     await soma.POST("/store", { body: { text: "hello" } });
 *     const { data } = await soma.POST("/retrieve", {
 *       body: { query: "hi?", k: 3 },
 *     });
 *     console.log(data?.hits);
 */
export function createClient(options: SomaClientOptions) {
  const auth = options.token ?? options.apiKey;
  const headers: Record<string, string> = { ...(options.headers ?? {}) };
  if (auth) {
    headers.Authorization = `Bearer ${auth}`;
  }
  // `openapi-fetch` uses `exactOptionalPropertyTypes` — only forward a
  // custom fetch when the caller actually provided one, otherwise let
  // the library fall back to the platform default.
  const clientOptions: Parameters<typeof createOpenApiClient<paths>>[0] = {
    baseUrl: options.baseUrl,
    headers,
  };
  if (options.fetch) {
    clientOptions.fetch = options.fetch;
  }
  return createOpenApiClient<paths>(clientOptions);
}

export type { paths, components, operations } from "./schema";
export { withRetry } from "./retry";
export type { RetryOptions } from "./retry";
