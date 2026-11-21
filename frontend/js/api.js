/**
 * frontend/js/api.js
 * API Client Module for LeafScan — Plant Disease Detection V2
 * Handles multipart/form-data upload, HTTP error parsing, network timeouts, and health verification.
 */

const API_BASE_URL = ''; // Same-origin deployment

/**
 * Diagnostic submission to FastAPI inference endpoint.
 *
 * @param {File} imageFile - Browser File object (JPEG, PNG, WebP)
 * @returns {Promise<Object>} Structured prediction response
 * @throws {Error} Descriptive error on network failure or HTTP error
 */
export async function diagnoseImage(imageFile) {
  if (!imageFile) {
    throw new Error('No image file provided for diagnosis.');
  }

  const formData = new FormData();
  formData.append('image', imageFile, imageFile.name);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000); // 30-second timeout

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/v2/diagnose`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
      headers: {
        // Let the browser set Content-Type with multipart boundary
        'Accept': 'application/json'
      }
    });
  } catch (netErr) {
    if (netErr.name === 'AbortError') {
      throw new Error('Diagnostic inference timed out after 30 seconds. Please check backend server status and retry.');
    }
    throw new Error(`Network error connecting to diagnosis service: ${netErr.message || 'Server unreachable'}`);
  } finally {
    clearTimeout(timeoutId);
  }

  let data;
  try {
    data = await response.json();
  } catch (jsonErr) {
    throw new Error(`Failed to parse server response as JSON (HTTP ${response.status})`);
  }

  if (!response.ok) {
    const errorMsg = data?.error?.message || data?.detail || `Diagnosis request failed (HTTP ${response.status})`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.data = data;
    throw err;
  }

  return data;
}

/**
 * Health check endpoint probe.
 *
 * @returns {Promise<Object>} System health status and loaded model telemetry
 */
export async function checkHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout for health checks

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: controller.signal,
      headers: { 'Accept': 'application/json' }
    });
    if (!response.ok) {
      throw new Error(`Health check returned HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    return {
      status: 'unavailable',
      error: err.message
    };
  } finally {
    clearTimeout(timeoutId);
  }
}
