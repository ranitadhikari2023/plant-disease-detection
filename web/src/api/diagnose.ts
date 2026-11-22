/**
 * web/src/api/diagnose.ts
 * API Client method for foliar specimen diagnosis
 */

import { API_BASE_URL } from './client';
import type { DiagnosisResponse } from './schema';

export interface DiagnoseOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export class DiagnosisError extends Error {
  status?: number;
  data?: unknown;

  constructor(message: string, status?: number, data?: unknown) {
    super(message);
    this.name = 'DiagnosisError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Executes a diagnosis by submitting an image file to POST /v2/diagnose.
 * Preserves the 30-second AbortController timeout and multipart/form-data browser boundary.
 */
export async function diagnoseImage(
  imageFile: File,
  options: DiagnoseOptions = {}
): Promise<DiagnosisResponse> {
  if (!imageFile) {
    throw new DiagnosisError('No image file provided for diagnosis.');
  }

  const { signal: externalSignal, timeoutMs = 30000 } = options;
  const controller = new AbortController();

  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  // Link external signal if provided
  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort());
  }

  const formData = new FormData();
  formData.append('image', imageFile, imageFile.name);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/v2/diagnose`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
      },
    });
  } catch (netErr: unknown) {
    clearTimeout(timer);
    if (timedOut || (netErr instanceof Error && netErr.name === 'AbortError')) {
      throw new DiagnosisError(
        'Diagnostic inference timed out after 30 seconds. Please check backend server status and retry.',
        408
      );
    }
    const message = netErr instanceof Error ? netErr.message : 'Server unreachable';
    throw new DiagnosisError(`Network error connecting to diagnosis service: ${message}`);
  } finally {
    clearTimeout(timer);
  }

  let data: DiagnosisResponse & { detail?: string };
  try {
    data = await response.json();
  } catch {
    throw new DiagnosisError(
      `Failed to parse server response as JSON (HTTP ${response.status})`,
      response.status
    );
  }

  if (!response.ok) {
    const errorMsg =
      data?.error?.message ||
      data?.detail ||
      `Diagnosis request failed (HTTP ${response.status})`;
    throw new DiagnosisError(errorMsg, response.status, data);
  }

  return data;
}
