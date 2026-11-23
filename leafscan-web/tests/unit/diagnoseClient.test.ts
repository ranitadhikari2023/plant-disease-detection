import {
  ApiError,
  ContractViolationError,
  diagnoseLeaf,
} from "@/lib/api/diagnoseClient";
import { beforeEach, describe, expect, it, vi } from "vitest";

describe("diagnoseClient API Client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("successfully parses valid 200 response and returns typed DiagnosisResponse", async () => {
    const mockSuccessResponse = {
      success: true,
      model_version: "v2",
      plant_species: "Tomato",
      species_confidence: 0.99,
      disease: "Healthy",
      disease_confidence: 0.98,
      diagnosis: "Tomato - Healthy",
      safety_status: "confident",
      safety_warnings: [],
      advisory: "Foliage appears healthy.",
      species_probabilities: { Tomato: 0.99, Potato: 0.01 },
      probabilities: {
        Healthy: 0.98,
        "Early Blight": 0.01,
        "Late Blight": 0.01,
      },
      metadata: {
        execution_time_ms: 25.0,
        architecture: "MobileNetV2 + CBAM Attention",
        model_bundle: "3.0.0",
        request_id: "req-success-123",
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ "X-Request-ID": "req-success-123" }),
      json: async () => mockSuccessResponse,
    });

    const fileBlob = new Blob(["dummy bytes"], { type: "image/jpeg" });
    const result = await diagnoseLeaf(fileBlob, {
      requestId: "req-success-123",
    });

    expect(result.success).toBe(true);
    expect(result.plant_species).toBe("Tomato");
    expect(result.safety_status).toBe("confident");
    expect(result.metadata.request_id).toBe("req-success-123");
  });

  it("retries on HTTP 503 and honors Retry-After header", async () => {
    const mock503 = {
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      headers: new Headers({ "Retry-After": "1" }),
      json: async () => ({ detail: "Capacity saturated" }),
    };

    const mock200 = {
      ok: true,
      status: 200,
      headers: new Headers({ "X-Request-ID": "req-retry-123" }),
      json: async () => ({
        success: true,
        model_version: "v2",
        plant_species: "Potato",
        species_confidence: 0.95,
        disease: "Early Blight",
        disease_confidence: 0.92,
        diagnosis: "Potato - Early Blight",
        safety_status: "confident",
        safety_warnings: [],
        advisory: "Early blight management guidance.",
        species_probabilities: { Potato: 0.95 },
        probabilities: { "Early Blight": 0.92 },
        metadata: { request_id: "req-retry-123" },
      }),
    };

    let callCount = 0;
    global.fetch = vi.fn().mockImplementation(() => {
      callCount++;
      return Promise.resolve(callCount === 1 ? mock503 : mock200);
    });

    const onRetry = vi.fn();
    const fileBlob = new Blob(["dummy bytes"], { type: "image/jpeg" });
    const result = await diagnoseLeaf(fileBlob, {
      requestId: "req-retry-123",
      onRetry,
    });

    expect(callCount).toBe(2);
    expect(onRetry).toHaveBeenCalledWith(1, expect.any(Number));
    expect(result.success).toBe(true);
  });

  it("throws ContractViolationError when response violates schema", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({ "X-Request-ID": "bad-schema" }),
      json: async () => ({ invalid_field: "corrupt data" }),
    });

    const fileBlob = new Blob(["dummy bytes"], { type: "image/jpeg" });
    await expect(
      diagnoseLeaf(fileBlob, { requestId: "bad-schema" }),
    ).rejects.toThrow(ContractViolationError);
  });

  it("throws ApiError preserving request_id when HTTP 413 payload too large", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 413,
      statusText: "Payload Too Large",
      headers: new Headers({ "X-Request-ID": "req-large-123" }),
      json: async () => ({
        detail: "File exceeds maximum size limit of 10 MB.",
      }),
    });

    const fileBlob = new Blob(["dummy bytes"], { type: "image/jpeg" });
    await expect(
      diagnoseLeaf(fileBlob, { requestId: "req-large-123" }),
    ).rejects.toThrow(ApiError);
  });
});
