import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { diagnoseImage, DiagnosisError } from './diagnose';

describe('diagnoseImage API client', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('rejects if no image file provided', async () => {
    await expect(diagnoseImage(null as any)).rejects.toThrow(
      'No image file provided for diagnosis'
    );
  });

  it('submits valid FormData with field name "image" and Accept header', async () => {
    let capturedBody: any;
    let capturedHeaders: any;

    global.fetch = vi.fn().mockImplementation((_url, init) => {
      capturedBody = init.body;
      capturedHeaders = init.headers;
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          success: true,
          plant_species: 'Potato',
          disease: 'Early Blight',
          safety_status: 'confident',
        }),
      });
    });

    const file = new File(['fake-content'], 'leaf.jpg', { type: 'image/jpeg' });
    const result = await diagnoseImage(file);

    expect(result.success).toBe(true);
    expect(capturedBody instanceof FormData).toBe(true);
    expect(capturedBody.get('image')).toBeDefined();
    expect(capturedHeaders['Accept']).toBe('application/json');
  });

  it('handles external AbortSignal', async () => {
    const extController = new AbortController();
    global.fetch = vi.fn().mockImplementation((_url, init) => {
      return new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => {
          const err = new Error('AbortError');
          err.name = 'AbortError';
          reject(err);
        });
      });
    });

    const file = new File(['data'], 'leaf.jpg', { type: 'image/jpeg' });
    const promise = diagnoseImage(file, { signal: extController.signal });
    extController.abort();
    await expect(promise).rejects.toThrow('timed out after 30 seconds');
  });

  it('handles backend error response with detail string', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Invalid or corrupted image payload.' }),
    });

    const file = new File(['bad'], 'bad.jpg', { type: 'image/jpeg' });
    await expect(diagnoseImage(file)).rejects.toThrow(
      'Invalid or corrupted image payload.'
    );
  });

  it('handles non-JSON server error response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error('Unexpected token < in JSON');
      },
    });

    const file = new File(['bad'], 'bad.jpg', { type: 'image/jpeg' });
    await expect(diagnoseImage(file)).rejects.toThrow('Failed to parse server response as JSON');
  });

  it('handles structured 500 error response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({
        error: {
          type: 'InferenceExecutionError',
          message: 'Diagnostic inference encountered an internal error.',
        },
      }),
    });

    const file = new File(['bad'], 'bad.jpg', { type: 'image/jpeg' });
    await expect(diagnoseImage(file)).rejects.toThrow(
      'Diagnostic inference encountered an internal error.'
    );
  });

  it('handles network failure with non-Error', async () => {
    global.fetch = vi.fn().mockRejectedValue('String network error');

    const file = new File(['data'], 'leaf.jpg', { type: 'image/jpeg' });
    await expect(diagnoseImage(file)).rejects.toThrow('Network error connecting');
  });

  it('handles request timeout cleanly', async () => {
    global.fetch = vi.fn().mockImplementation((_url, init) => {
      return new Promise((_, reject) => {
        init.signal.addEventListener('abort', () => {
          const err = new Error('AbortError');
          err.name = 'AbortError';
          reject(err);
        });
      });
    });

    const file = new File(['data'], 'leaf.jpg', { type: 'image/jpeg' });
    await expect(diagnoseImage(file, { timeoutMs: 50 })).rejects.toThrow(
      'Diagnostic inference timed out after 30 seconds'
    );
  });
});
