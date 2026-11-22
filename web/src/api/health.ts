/**
 * web/src/api/health.ts
 * API Client method for backend system health verification
 */

import { API_BASE_URL } from './client';
import type { HealthResponse } from './schema';

export interface HealthOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

/**
 * Probes GET /health with a 5-second timeout.
 * Safely handles unavailable states without crashing the caller.
 * Note: lazy loading of models (models_loaded: false) is normal on cold start.
 */
export async function checkHealth(options: HealthOptions = {}): Promise<HealthResponse> {
  const { signal: externalSignal, timeoutMs = 5000 } = options;
  const controller = new AbortController();

  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort());
  }

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });

    clearTimeout(timer);

    if (!response.ok) {
      return {
        status: 'unavailable',
        error: `Health check returned HTTP ${response.status}`,
      };
    }

    const data: HealthResponse = await response.json();
    return data;
  } catch (err: unknown) {
    clearTimeout(timer);
    const msg = timedOut
      ? 'Health check timed out after 5 seconds'
      : err instanceof Error
      ? err.message
      : 'Network error';
    return {
      status: 'unavailable',
      error: msg,
    };
  }
}
