import {
  MAX_UPLOAD_BYTES,
  prevalidateImage,
} from "@/features/diagnose/prevalidate";
import { describe, expect, it } from "vitest";

describe("Image Prevalidation", () => {
  it("rejects when no file is provided", async () => {
    const result = await prevalidateImage(null as any);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("No file provided");
  });

  it("rejects empty files (0 bytes)", async () => {
    const emptyFile = new File([], "empty.jpg", { type: "image/jpeg" });
    const result = await prevalidateImage(emptyFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("empty (0 bytes)");
  });

  it("rejects files exceeding 10MB limit", async () => {
    const oversizedBuffer = new Uint8Array(MAX_UPLOAD_BYTES + 1024);
    const largeFile = new File([oversizedBuffer], "large.jpg", {
      type: "image/jpeg",
    });
    const result = await prevalidateImage(largeFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("exceeds the 10 MB limit");
  });

  it("rejects unsupported MIME types and file extensions", async () => {
    const textFile = new File(["not an image"], "notes.txt", {
      type: "text/plain",
    });
    const result = await prevalidateImage(textFile);
    expect(result.valid).toBe(false);
    expect(result.error).toContain("Unsupported file format");
  });

  it("accepts valid JPEG, PNG, and WEBP files within limits", async () => {
    const validJpg = new File([new Uint8Array(2048)], "leaf.jpg", {
      type: "image/jpeg",
    });
    const validPng = new File([new Uint8Array(2048)], "leaf.png", {
      type: "image/png",
    });
    const validWebp = new File([new Uint8Array(2048)], "leaf.webp", {
      type: "image/webp",
    });

    expect((await prevalidateImage(validJpg)).valid).toBe(true);
    expect((await prevalidateImage(validPng)).valid).toBe(true);
    expect((await prevalidateImage(validWebp)).valid).toBe(true);
  });
});
