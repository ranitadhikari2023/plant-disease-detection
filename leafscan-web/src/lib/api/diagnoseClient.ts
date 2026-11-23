import { generateRequestId } from "../platform/requestId";
import { telemetry } from "../platform/telemetry";
import {
  ApiErrorEnvelope,
  type DiagnosisResponse,
  DiagnosisResponseSchema,
  type FeedbackRequest,
  type FeedbackResponse,
  FeedbackResponseSchema,
  type HealthResponse,
  HealthResponseSchema,
  type ModelManifest,
  ModelManifestSchema,
  type ReadyResponse,
  ReadyResponseSchema,
} from "./schemas";

export class ApiError extends Error {
  public status: number;
  public code: string;
  public requestId: string;
  public details?: unknown;

  constructor(
    status: number,
    code: string,
    message: string,
    requestId: string,
    details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

export class ContractViolationError extends Error {
  public requestId: string;
  public rawPayload: unknown;

  constructor(message: string, requestId: string, rawPayload: unknown) {
    super(message);
    this.name = "ContractViolationError";
    this.requestId = requestId;
    this.rawPayload = rawPayload;
  }
}

export function getApiBaseUrl(): string {
  // Allow NEXT_PUBLIC_API_BASE_URL to override; fallback to empty string (same-origin) or http://localhost:8000
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    // If running in browser and no base URL specified, try localhost:8000 if running on Next dev server (port 3000)
    if (window.location.port === "3000") {
      return "http://localhost:8000";
    }
    return "";
  }
  return "http://127.0.0.1:8000";
}

function parseRetryAfter(headerValue: string | null): number {
  if (!headerValue) return 1000;
  const seconds = Number.parseInt(headerValue, 10);
  if (!Number.isNaN(seconds) && seconds > 0) {
    return seconds * 1000;
  }
  // Try HTTP Date
  const parsedDate = Date.parse(headerValue);
  if (!Number.isNaN(parsedDate)) {
    const diff = parsedDate - Date.now();
    return Math.max(1000, Math.min(diff, 60000));
  }
  return 1000;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function diagnoseLeaf(
  imageBlob: Blob,
  options?: {
    requestId?: string;
    onRetry?: (attempt: number, delayMs: number) => void;
  },
): Promise<DiagnosisResponse> {
  const reqId = options?.requestId || generateRequestId();
  const baseUrl = getApiBaseUrl();
  const endpoint = `${baseUrl}/v2/diagnose`;

  const formData = new FormData();
  formData.append("image", imageBlob, "specimen.jpg");

  let attempt = 0;
  const maxRetries = 3;

  while (attempt <= maxRetries) {
    try {
      const t0 = performance.now();
      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "X-Request-ID": reqId,
        },
        body: formData,
      });

      const elapsed = performance.now() - t0;
      telemetry.recordLatency("POST /v2/diagnose", elapsed);

      // Handle HTTP 503 Capacity load shedding
      if (res.status === 503 && attempt < maxRetries) {
        attempt++;
        const retryHeader = res.headers.get("Retry-After");
        const baseDelay = parseRetryAfter(retryHeader);
        const jitter = Math.random() * 500;
        const delay = Math.min(baseDelay + jitter, 10000);

        if (options?.onRetry) {
          options.onRetry(attempt, delay);
        }
        await sleep(delay);
        continue;
      }

      if (!res.ok) {
        let errData: any = {};
        try {
          errData = await res.json();
        } catch {
          errData = { message: res.statusText || "Diagnosis request failed" };
        }

        const returnedReqId =
          res.headers.get("X-Request-ID") ||
          errData?.error?.request_id ||
          reqId;
        const code = errData?.error?.code || `HTTP_${res.status}`;
        const message =
          errData?.error?.message ||
          (typeof errData?.detail === "string" ? errData.detail : undefined) ||
          errData?.message ||
          `Server returned error ${res.status}`;

        throw new ApiError(
          res.status,
          code,
          message,
          returnedReqId,
          errData?.error?.details || errData,
        );
      }

      const returnedReqId = res.headers.get("X-Request-ID");
      const rawJson = await res.json();

      // Check request ID matching
      const jsonReqId = rawJson?.metadata?.request_id || returnedReqId;
      if (jsonReqId && jsonReqId !== reqId) {
        telemetry.recordMismatchRequestId(reqId, jsonReqId);
      }

      // Strict Zod runtime validation
      const parseResult = DiagnosisResponseSchema.safeParse(rawJson);
      if (!parseResult.success) {
        telemetry.recordContractViolation(
          "/v2/diagnose",
          parseResult.error.format(),
          rawJson,
          reqId,
        );
        throw new ContractViolationError(
          "Diagnostic response schema mismatch. System prevented rendering unverified output.",
          reqId,
          rawJson,
        );
      }

      // Ensure metadata has request_id
      if (!parseResult.data.metadata.request_id) {
        parseResult.data.metadata.request_id = reqId;
      }

      return parseResult.data;
    } catch (err: any) {
      if (err instanceof ApiError || err instanceof ContractViolationError) {
        throw err;
      }
      if (
        attempt < maxRetries &&
        (err.name === "TypeError" || err.message?.includes("fetch"))
      ) {
        attempt++;
        const delay = 1000 * 2 ** attempt + Math.random() * 300;
        if (options?.onRetry) {
          options.onRetry(attempt, delay);
        }
        await sleep(delay);
        continue;
      }
      throw new ApiError(
        0,
        "NETWORK_ERROR",
        err.message || "Network connection failure",
        reqId,
      );
    }
  }

  throw new ApiError(
    503,
    "CAPACITY_EXHAUSTED",
    "Server is busy. Capacity exceeded after 3 retries.",
    reqId,
  );
}

export async function fetchHealth(): Promise<HealthResponse> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  const data = await res.json();
  return HealthResponseSchema.parse(data);
}

export async function fetchReady(): Promise<ReadyResponse> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/ready`);
  if (!res.ok) {
    throw new Error(`Readiness check failed with status ${res.status}`);
  }
  const data = await res.json();
  return ReadyResponseSchema.parse(data);
}

export async function fetchModelManifest(): Promise<ModelManifest> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/v2/model`);
  if (!res.ok) {
    throw new Error(`Model manifest request failed with status ${res.status}`);
  }
  const data = await res.json();
  return ModelManifestSchema.parse(data);
}

export async function submitFeedback(
  payload: FeedbackRequest,
): Promise<FeedbackResponse> {
  const baseUrl = getApiBaseUrl();
  const res = await fetch(`${baseUrl}/v2/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(payload.idempotency_key
        ? { "Idempotency-Key": payload.idempotency_key }
        : {}),
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errData: any = {};
    try {
      errData = await res.json();
    } catch {
      errData = { message: res.statusText };
    }
    const message =
      errData?.detail ||
      errData?.message ||
      `Feedback submission failed with status ${res.status}`;
    throw new Error(message);
  }

  const data = await res.json();
  return FeedbackResponseSchema.parse(data);
}
