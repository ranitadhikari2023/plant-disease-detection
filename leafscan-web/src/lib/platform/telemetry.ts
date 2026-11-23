export interface TelemetryEvent {
  name: string;
  attributes?: Record<string, string | number | boolean | undefined>;
}

class TelemetryService {
  private enabled = true;

  recordContractViolation(
    endpoint: string,
    error: unknown,
    payload: unknown,
    requestId?: string,
  ) {
    if (!this.enabled) return;
    const msg = `[Telemetry: Contract Violation] at ${endpoint} (Req ID: ${requestId || "unknown"}): ${String(error)}`;
    console.error(msg, { payload, error });
    // If Sentry is configured on window
    if (typeof window !== "undefined" && (window as any).Sentry) {
      (window as any).Sentry.captureMessage(msg, {
        level: "error",
        extra: { endpoint, payload, error, requestId },
      });
    }
  }

  recordMismatchRequestId(
    sentId: string,
    receivedId: string | null | undefined,
  ) {
    if (!this.enabled) return;
    const msg = `[Telemetry: Request ID Mismatch] Sent: ${sentId}, Received: ${receivedId}`;
    console.warn(msg);
    if (typeof window !== "undefined" && (window as any).Sentry) {
      (window as any).Sentry.captureMessage(msg, {
        level: "warning",
        extra: { sentId, receivedId },
      });
    }
  }

  recordLatency(operation: string, durationMs: number) {
    if (!this.enabled) return;
    if (durationMs > 2000) {
      console.warn(
        `[Telemetry: Slow Operation] ${operation} took ${durationMs.toFixed(1)}ms`,
      );
    }
  }

  recordError(error: unknown, context?: Record<string, unknown>) {
    if (!this.enabled) return;
    console.error("[Telemetry: Error]", error, context);
    if (typeof window !== "undefined" && (window as any).Sentry) {
      (window as any).Sentry.captureException(error, { extra: context });
    }
  }
}

export const telemetry = new TelemetryService();
