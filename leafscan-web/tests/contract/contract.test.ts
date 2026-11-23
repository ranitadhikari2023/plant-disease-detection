import { diagnoseLeaf } from "@/lib/api/diagnoseClient";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";

// Contract test fixtures matching actual LeafScan V3 Backend responses
const fixtures = {
  confident: {
    success: true,
    model_version: "v2",
    plant_species: "Potato",
    species_confidence: 0.9854,
    disease: "Early Blight",
    disease_confidence: 0.9542,
    diagnosis: "Potato - Early Blight",
    safety_status: "confident",
    safety_warnings: [],
    advisory:
      "Early Blight detected. Prune lower foliage and monitor humidity.",
    species_probabilities: { Potato: 0.9854, Tomato: 0.0146 },
    probabilities: {
      "Early Blight": 0.9542,
      "Late Blight": 0.03,
      Healthy: 0.0158,
    },
    metadata: {
      execution_time_ms: 32.5,
      architecture: "MobileNetV2 + CBAM Attention",
      model_bundle: "v3.0.0",
      prediction_set: ["Early Blight"],
      request_id: "req-contract-confident",
    },
  },
  speciesUncertain: {
    success: true,
    model_version: "v2",
    plant_species: "Potato",
    species_confidence: 0.8123,
    disease: null,
    disease_confidence: null,
    diagnosis: "Potato (Abstained - Species Uncertain)",
    safety_status: "species_uncertain",
    safety_warnings: [
      "Species confidence (81.23%) is below safety threshold (90%).",
    ],
    advisory:
      "SPECIES UNCERTAINTY: Specimen disease diagnosis has been abstained.",
    species_probabilities: { Potato: 0.8123, Tomato: 0.1877 },
    probabilities: {},
    metadata: {
      execution_time_ms: 18.4,
      architecture: "MobileNetV2 + CBAM Attention",
      model_bundle: "v3.0.0",
      request_id: "req-contract-species-unc",
    },
  },
  lateBlightUncertain: {
    success: true,
    model_version: "v2",
    plant_species: "Potato",
    species_confidence: 0.9912,
    disease: "Late Blight",
    disease_confidence: 0.8415,
    diagnosis: "Potato - Late Blight",
    safety_status: "late_blight_uncertain",
    safety_warnings: [
      "Late Blight confidence (84.15%) is below high-consequence threshold (90%).",
    ],
    advisory:
      "LATE BLIGHT UNCERTAINTY: Folio symptoms indicate potential early lesions.",
    species_probabilities: { Potato: 0.9912, Tomato: 0.0088 },
    probabilities: {
      "Late Blight": 0.8415,
      "Early Blight": 0.12,
      Healthy: 0.0385,
    },
    metadata: {
      execution_time_ms: 34.1,
      architecture: "MobileNetV2 + CBAM Attention",
      model_bundle: "v3.0.0",
      prediction_set: ["Late Blight", "Early Blight"],
      request_id: "req-contract-lb-unc",
    },
  },
};

const server = setupServer(
  http.post("*/v2/diagnose", ({ request }) => {
    const reqId = request.headers.get("X-Request-ID");
    if (reqId === "req-contract-species-unc") {
      return HttpResponse.json(fixtures.speciesUncertain, {
        headers: { "X-Request-ID": reqId },
      });
    }
    if (reqId === "req-contract-lb-unc") {
      return HttpResponse.json(fixtures.lateBlightUncertain, {
        headers: { "X-Request-ID": reqId },
      });
    }
    return HttpResponse.json(fixtures.confident, {
      headers: { "X-Request-ID": reqId || "req-contract-confident" },
    });
  }),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("Backend Contract Testing with MSW", () => {
  it("verifies contract compliance for Confident diagnosis response variant", async () => {
    const fileBlob = new Blob(["leaf image data"], { type: "image/jpeg" });
    const result = await diagnoseLeaf(fileBlob, {
      requestId: "req-contract-confident",
    });

    expect(result.safety_status).toBe("confident");
    expect(result.plant_species).toBe("Potato");
    expect(result.disease).toBe("Early Blight");
    expect(result.species_confidence).toBeGreaterThanOrEqual(0.9);
    expect(result.disease_confidence).toBeGreaterThanOrEqual(0.7);
  });

  it("verifies contract compliance for Species Uncertain response variant", async () => {
    const fileBlob = new Blob(["leaf image data"], { type: "image/jpeg" });
    const result = await diagnoseLeaf(fileBlob, {
      requestId: "req-contract-species-unc",
    });

    expect(result.safety_status).toBe("species_uncertain");
    expect(result.disease).toBeNull();
    expect(result.disease_confidence).toBeNull();
    expect(result.advisory).toContain("SPECIES UNCERTAINTY");
  });

  it("verifies contract compliance for Late Blight Uncertain response variant", async () => {
    const fileBlob = new Blob(["leaf image data"], { type: "image/jpeg" });
    const result = await diagnoseLeaf(fileBlob, {
      requestId: "req-contract-lb-unc",
    });

    expect(result.safety_status).toBe("late_blight_uncertain");
    expect(result.disease).toBe("Late Blight");
    expect(result.species_confidence).toBeGreaterThanOrEqual(0.9);
    expect(result.metadata.prediction_set).toContain("Late Blight");
  });
});
