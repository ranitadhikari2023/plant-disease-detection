import {
  DiagnosisResponseSchema,
  FeedbackRequestSchema,
  HealthResponseSchema,
  ModelManifestSchema,
  ReadyResponseSchema,
} from "@/lib/api/schemas";
import { describe, expect, it } from "vitest";

describe("API Zod Schemas", () => {
  it("validates a compliant DiagnosisResponse", () => {
    const mock = {
      success: true,
      model_version: "v2",
      plant_species: "Potato",
      species_confidence: 0.9854,
      disease: "Early Blight",
      disease_confidence: 0.9542,
      diagnosis: "Potato - Early Blight",
      safety_status: "confident",
      safety_warnings: [],
      advisory: "Early blight confirmed. Sanitation recommended.",
      species_probabilities: { Potato: 0.9854, Tomato: 0.0146 },
      probabilities: {
        "Early Blight": 0.9542,
        "Late Blight": 0.03,
        Healthy: 0.0158,
      },
      metadata: {
        execution_time_ms: 32.5,
        architecture: "MobileNetV2 + CBAM Attention",
        model_bundle: "3.0.0",
        prediction_set: ["Early Blight"],
        request_id: "test-req-1234",
      },
    };

    const parsed = DiagnosisResponseSchema.safeParse(mock);
    expect(parsed.success).toBe(true);
  });

  it("validates an abstained species_uncertain response", () => {
    const mock = {
      success: true,
      model_version: "v2",
      plant_species: "Potato",
      species_confidence: 0.82,
      disease: null,
      disease_confidence: null,
      diagnosis: "Potato (Abstained - Species Uncertain)",
      safety_status: "species_uncertain",
      safety_warnings: ["Species confidence (82.00%) below 90% threshold."],
      advisory:
        "SPECIES UNCERTAINTY: Specimen disease diagnosis has been abstained.",
      species_probabilities: { Potato: 0.82, Tomato: 0.18 },
      probabilities: {},
      metadata: {
        execution_time_ms: 18.2,
        architecture: "MobileNetV2 + CBAM Attention",
        model_bundle: "3.0.0",
        request_id: "req-uncertain-1",
      },
    };

    const parsed = DiagnosisResponseSchema.safeParse(mock);
    expect(parsed.success).toBe(true);
  });

  it("rejects malformed diagnosis schemas missing mandatory fields", () => {
    const invalidMock = {
      // missing success, diagnosis, safety_status
      plant_species: "Potato",
    };

    const parsed = DiagnosisResponseSchema.safeParse(invalidMock);
    expect(parsed.success).toBe(false);
  });

  it("validates feedback request and enforces predicted_species", () => {
    const valid = {
      predicted_species: "Potato",
      predicted_disease: "Early Blight",
      feedback_type: "agree",
    };
    expect(FeedbackRequestSchema.safeParse(valid).success).toBe(true);

    const invalid = {
      predicted_species: "", // empty
    };
    expect(FeedbackRequestSchema.safeParse(invalid).success).toBe(false);
  });

  it("validates health and ready responses", () => {
    expect(
      HealthResponseSchema.safeParse({
        status: "healthy",
        model_version: "v2",
      }).success,
    ).toBe(true);

    expect(
      ReadyResponseSchema.safeParse({
        status: "ready",
        bundle_version: "v3.0.0",
        engine: "OnnxEngine",
        execution_provider: "CPUExecutionProvider",
        models_verified: ["species_expert", "potato_expert", "tomato_expert"],
      }).success,
    ).toBe(true);
  });
});
