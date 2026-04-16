/**
 * Compile-time type-inference smoke test. No runtime assertions — if
 * the schema drifts from what `src/index.ts` expects, tsc fails here.
 *
 * Each `expectTypeOf` call is evaluated at compile time; a mismatch
 * surfaces as a TypeScript error during `npm run typecheck` /
 * `npm test`. The `.test(true)` calls also run at runtime but the
 * real signal is the type-level check above them.
 */
import { describe, it, expectTypeOf } from "vitest";
import type { paths, components } from "../src/schema";
import { createClient } from "../src/index";

describe("generated schema types", () => {
  it("exposes a POST /store body with text + metadata", () => {
    type StoreBody = NonNullable<
      paths["/store"]["post"]["requestBody"]
    >["content"]["application/json"];
    expectTypeOf<StoreBody>().toHaveProperty("text");
    expectTypeOf<StoreBody>().toHaveProperty("metadata");
    expectTypeOf<StoreBody["text"]>().toEqualTypeOf<string>();
  });

  it("exposes a POST /retrieve body with query + k", () => {
    type RetrieveBody = NonNullable<
      paths["/retrieve"]["post"]["requestBody"]
    >["content"]["application/json"];
    expectTypeOf<RetrieveBody>().toHaveProperty("query");
    expectTypeOf<RetrieveBody>().toHaveProperty("k");
    expectTypeOf<RetrieveBody["query"]>().toEqualTypeOf<string>();
  });

  it("types the hit array on the retrieve response", () => {
    type RetrieveResp =
      paths["/retrieve"]["post"]["responses"]["200"]["content"]["application/json"];
    expectTypeOf<RetrieveResp>().toHaveProperty("hits");
    type Hit = RetrieveResp["hits"][number];
    expectTypeOf<Hit>().toHaveProperty("node_id");
    expectTypeOf<Hit>().toHaveProperty("text");
    expectTypeOf<Hit>().toHaveProperty("score");
  });

  it("advertises the bearerAuth security scheme", () => {
    type SecuritySchemes = NonNullable<
      components["securitySchemes"]
    >;
    expectTypeOf<SecuritySchemes>().toHaveProperty("bearerAuth");
  });

  it("types multi-tenant bundle path params", () => {
    type BundleStoreParams = NonNullable<
      paths["/bundles/{name}/store"]["post"]["parameters"]["path"]
    >;
    expectTypeOf<BundleStoreParams>().toHaveProperty("name");
  });

  it("types a 401 error as ErrorResponse", () => {
    type Err401 =
      paths["/store"]["post"]["responses"]["401"]["content"]["application/json"];
    expectTypeOf<Err401>().toHaveProperty("detail");
  });

  it("createClient returns an object with HTTP method helpers", () => {
    const client = createClient({ baseUrl: "http://localhost:8420" });
    expectTypeOf(client).toHaveProperty("POST");
    expectTypeOf(client).toHaveProperty("GET");
  });
});
