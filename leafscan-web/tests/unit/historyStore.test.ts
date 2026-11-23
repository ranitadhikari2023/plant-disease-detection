import {
  clearAllDiagnoses,
  deleteDiagnosis,
  getAllDiagnoses,
  getDiagnosis,
  getOfflineQueue,
  queueOfflineSubmission,
  removeOfflineSubmission,
  saveDiagnosis,
} from "@/features/history/historyStore";
import type { DiagnosisResponse } from "@/lib/api/schemas";
import { beforeEach, describe, expect, it } from "vitest";

const sampleResponse: DiagnosisResponse = {
  success: true,
  model_version: "v2",
  plant_species: "Potato",
  species_confidence: 0.98,
  disease: "Early Blight",
  disease_confidence: 0.94,
  diagnosis: "Potato - Early Blight",
  safety_status: "confident",
  safety_warnings: [],
  advisory: "Manage foliar lesions promptly.",
  species_probabilities: { Potato: 0.98, Tomato: 0.02 },
  probabilities: { "Early Blight": 0.94, "Late Blight": 0.04, Healthy: 0.02 },
  metadata: {
    execution_time_ms: 22.0,
    architecture: "MobileNetV2 + CBAM Attention",
    model_bundle: "3.0.0",
    prediction_set: ["Early Blight"],
    request_id: "req-hist-test-1",
  },
};

describe("historyStore", () => {
  beforeEach(async () => {
    await clearAllDiagnoses();
  });

  it("saves a diagnosis and retrieves it by request ID", async () => {
    const saved = await saveDiagnosis(
      sampleResponse,
      "data:image/jpeg;base64,sample",
    );
    expect(saved.requestId).toBe("req-hist-test-1");
    expect(saved.plantSpecies).toBe("Potato");

    const retrieved = await getDiagnosis("req-hist-test-1");
    expect(retrieved).not.toBeNull();
    expect(retrieved?.requestId).toBe("req-hist-test-1");
    expect(retrieved?.disease).toBe("Early Blight");
  });

  it("deletes a diagnosis correctly", async () => {
    await saveDiagnosis(sampleResponse);
    await deleteDiagnosis("req-hist-test-1");

    const retrieved = await getDiagnosis("req-hist-test-1");
    expect(retrieved).toBeNull();
  });

  it("queues offline submissions and retrieves them", async () => {
    const dummyBlob = new Blob(["offline image"], { type: "image/jpeg" });
    const item = await queueOfflineSubmission(dummyBlob, "thumb.jpg");

    expect(item.status).toBe("queued");
    expect(item.id).toMatch(/^queue_/);

    const queue = await getOfflineQueue();
    expect(queue.some((q) => q.id === item.id)).toBe(true);

    await removeOfflineSubmission(item.id);
    const queueAfter = await getOfflineQueue();
    expect(queueAfter.some((q) => q.id === item.id)).toBe(false);
  });
});
