import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { drop } from "@mswjs/data";
import dotenv from "dotenv";
import manifest from "../manifest.json";
import { db } from "./__mocks__/db";
import { setupTests } from "./__mocks__/helpers";
import { server } from "./__mocks__/node";

dotenv.config();

beforeAll(() => {
  server.listen();
});
afterEach(() => {
  server.resetHandlers();
  jest.clearAllMocks();
});
afterAll(() => server.close());

describe("Plugin tests", () => {
  beforeEach(async () => {
    drop(db);
    await setupTests();
  });

  it("Should serve the manifest file", async () => {
    const worker = (await import("../src/worker")).default;
    const response = await worker.fetch(new Request("http://localhost/manifest.json"), {
      OPENAI_API_KEY: "",
      TAVILY_API_KEY: "",
    });
    const content = await response.json();
    expect(content).toEqual(manifest);
  });
});
