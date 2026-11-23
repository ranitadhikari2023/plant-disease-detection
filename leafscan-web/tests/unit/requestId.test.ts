import { generateRequestId, isValidRequestId } from "@/lib/platform/requestId";
import { describe, expect, it } from "vitest";

describe("Request ID Utilities", () => {
  it("generates a valid UUID v4 format", () => {
    const id = generateRequestId();
    expect(typeof id).toBe("string");
    expect(id.length).toBe(36);
    expect(isValidRequestId(id)).toBe(true);
  });

  it("generates unique IDs across calls", () => {
    const set = new Set();
    for (let i = 0; i < 100; i++) {
      set.add(generateRequestId());
    }
    expect(set.size).toBe(100);
  });

  it("falls back to RFC4122 v4 generator when crypto.randomUUID is unavailable", () => {
    const original = crypto.randomUUID;
    try {
      // @ts-ignore
      crypto.randomUUID = undefined;
      const fallbackId = generateRequestId();
      expect(typeof fallbackId).toBe("string");
      expect(fallbackId.length).toBe(36);
      expect(isValidRequestId(fallbackId)).toBe(true);
    } finally {
      crypto.randomUUID = original;
    }
  });

  it("validates valid request IDs and rejects empty/null", () => {
    expect(isValidRequestId(null)).toBe(false);
    expect(isValidRequestId(undefined)).toBe(false);
    expect(isValidRequestId("")).toBe(false);
    expect(isValidRequestId("short")).toBe(false);
    expect(isValidRequestId("550e8400-e29b-41d4-a716-446655440000")).toBe(true);
  });
});
