import { diagnoseLeaf } from "@/lib/api/diagnoseClient";
import type { DiagnosisResponse, SafetyStatus } from "@/lib/api/schemas";
import { type DBSchema, type IDBPDatabase, openDB } from "idb";

export interface StoredDiagnosis {
  requestId: string;
  thumbnail?: string; // base64 or blob url
  plantSpecies: string | null;
  speciesConfidence: number;
  disease: string | null;
  diseaseConfidence: number | null;
  diagnosis: string;
  safetyStatus: SafetyStatus;
  advisory: string;
  safetyWarnings: string[];
  predictionSet?: string[] | null;
  timestamp: string;
  modelVersion: string;
  rawResponse: DiagnosisResponse;
}

export interface OfflineSubmission {
  id: string; // client UUID
  thumbnail: string;
  imageBlob: Blob;
  createdAt: string;
  status: "queued" | "syncing" | "failed";
  errorMessage?: string;
  retryCount: number;
}

interface LeafScanDB extends DBSchema {
  diagnoses: {
    key: string;
    value: StoredDiagnosis;
    indexes: {
      "by-timestamp": string;
      "by-status": string;
      "by-species": string;
    };
  };
  offline_queue: {
    key: string;
    value: OfflineSubmission;
    indexes: {
      "by-created": string;
    };
  };
}

const DB_NAME = "leafscan_v3_store";
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase<LeafScanDB>> | null = null;

function getDb(): Promise<IDBPDatabase<LeafScanDB>> {
  if (typeof window === "undefined") {
    return Promise.reject(
      new Error("IndexedDB is only available in the browser"),
    );
  }
  if (!dbPromise) {
    dbPromise = openDB<LeafScanDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        // Diagnoses store
        if (!db.objectStoreNames.contains("diagnoses")) {
          const diagStore = db.createObjectStore("diagnoses", {
            keyPath: "requestId",
          });
          diagStore.createIndex("by-timestamp", "timestamp");
          diagStore.createIndex("by-status", "safetyStatus");
          diagStore.createIndex("by-species", "plantSpecies");
        }

        // Offline queue store
        if (!db.objectStoreNames.contains("offline_queue")) {
          const queueStore = db.createObjectStore("offline_queue", {
            keyPath: "id",
          });
          queueStore.createIndex("by-created", "createdAt");
        }
      },
    });
  }
  return dbPromise;
}

// In-memory cache for fast hydration
const memoryDiagnosisCache = new Map<string, StoredDiagnosis>();

export async function saveDiagnosis(
  response: DiagnosisResponse,
  thumbnail?: string,
): Promise<StoredDiagnosis> {
  const reqId = response.metadata.request_id || `req_${Date.now()}`;
  const stored: StoredDiagnosis = {
    requestId: reqId,
    thumbnail,
    plantSpecies: response.plant_species || null,
    speciesConfidence: response.species_confidence,
    disease: response.disease || null,
    diseaseConfidence: response.disease_confidence ?? null,
    diagnosis: response.diagnosis,
    safetyStatus: response.safety_status,
    advisory: response.advisory,
    safetyWarnings: response.safety_warnings,
    predictionSet: response.metadata.prediction_set || null,
    timestamp: new Date().toISOString(),
    modelVersion:
      response.model_version || response.metadata.model_bundle || "v3.0.0",
    rawResponse: response,
  };

  memoryDiagnosisCache.set(reqId, stored);

  try {
    const db = await getDb();
    await db.put("diagnoses", stored);
  } catch (err) {
    console.warn("Failed to persist diagnosis to IndexedDB:", err);
  }

  return stored;
}

export async function getDiagnosis(
  requestId: string,
): Promise<StoredDiagnosis | null> {
  if (memoryDiagnosisCache.has(requestId)) {
    return memoryDiagnosisCache.get(requestId) || null;
  }

  try {
    const db = await getDb();
    const result = await db.get("diagnoses", requestId);
    if (result) {
      memoryDiagnosisCache.set(requestId, result);
      return result;
    }
  } catch (err) {
    console.warn("Failed to retrieve diagnosis from IndexedDB:", err);
  }

  return null;
}

export async function getAllDiagnoses(): Promise<StoredDiagnosis[]> {
  try {
    const db = await getDb();
    const items = await db.getAllFromIndex("diagnoses", "by-timestamp");
    return items.reverse(); // Newest first
  } catch (err) {
    console.warn("Failed to fetch diagnoses:", err);
    return Array.from(memoryDiagnosisCache.values()).reverse();
  }
}

export async function deleteDiagnosis(requestId: string): Promise<void> {
  memoryDiagnosisCache.delete(requestId);
  try {
    const db = await getDb();
    await db.delete("diagnoses", requestId);
  } catch (err) {
    console.warn("Failed to delete diagnosis:", err);
  }
}

export async function clearAllDiagnoses(): Promise<void> {
  memoryDiagnosisCache.clear();
  try {
    const db = await getDb();
    await db.clear("diagnoses");
  } catch (err) {
    console.warn("Failed to clear diagnoses:", err);
  }
}

// OFFLINE QUEUE
export async function queueOfflineSubmission(
  imageBlob: Blob,
  thumbnail: string,
): Promise<OfflineSubmission> {
  const id = `queue_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const item: OfflineSubmission = {
    id,
    thumbnail,
    imageBlob,
    createdAt: new Date().toISOString(),
    status: "queued",
    retryCount: 0,
  };

  try {
    const db = await getDb();
    await db.put("offline_queue", item);
  } catch (err) {
    console.error("Failed to enqueue offline submission:", err);
  }

  return item;
}

export async function getOfflineQueue(): Promise<OfflineSubmission[]> {
  try {
    const db = await getDb();
    const queue = await db.getAllFromIndex("offline_queue", "by-created");
    return queue;
  } catch (err) {
    return [];
  }
}

export async function removeOfflineSubmission(id: string): Promise<void> {
  try {
    const db = await getDb();
    await db.delete("offline_queue", id);
  } catch (err) {
    console.warn("Failed to remove offline queue item:", err);
  }
}

let isFlushing = false;

export async function flushOfflineQueue(
  onComplete?: (result: StoredDiagnosis) => void,
): Promise<{ processed: number; failed: number }> {
  if (isFlushing || typeof window === "undefined" || !navigator.onLine) {
    return { processed: 0, failed: 0 };
  }

  isFlushing = true;
  let processed = 0;
  let failed = 0;

  try {
    const items = await getOfflineQueue();
    for (const item of items) {
      try {
        const db = await getDb();
        await db.put("offline_queue", { ...item, status: "syncing" });

        const result = await diagnoseLeaf(item.imageBlob);
        const stored = await saveDiagnosis(result, item.thumbnail);
        await removeOfflineSubmission(item.id);
        processed++;
        onComplete?.(stored);
      } catch (err: any) {
        failed++;
        const db = await getDb();
        await db.put("offline_queue", {
          ...item,
          status: "failed",
          errorMessage: err.message,
          retryCount: item.retryCount + 1,
        });
      }
    }
  } finally {
    isFlushing = false;
  }

  return { processed, failed };
}
