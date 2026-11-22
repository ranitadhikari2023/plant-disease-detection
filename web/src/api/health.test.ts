import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { checkHealth } from './health';

describe('checkHealth API client', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('returns healthy status when endpoint returns 200 JSON', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'healthy',
        model_version: 'v2',
        architecture: 'MobileNetV2 + CBAM Attention',
        models_loaded: true,
      }),
    });

    const res = await checkHealth();
    expect(res.status).toBe('healthy');
    expect(res.models_loaded).toBe(true);
    expect(res.model_version).toBe('v2');
  });

  it('handles external AbortSignal', async () => {
    const extController = new AbortController();
    global.fetch = vi.fn().mockImplementation((_url, opts) => {
      return new Promise((_, reject) => {
        opts.signal.addEventListener('abort', () => {
          const err = new Error('AbortError');
          err.name = 'AbortError';
          reject(err);
        });
      });
    });

    const promise = checkHealth({ signal: extController.signal });
    extController.abort();
    const res = await promise;
    expect(res.status).toBe('unavailable');
  });

  it('handles models_loaded: false safely without marking service unhealthy', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'healthy',
        model_version: 'v2',
        architecture: 'MobileNetV2 + CBAM Attention',
        models_loaded: false,
      }),
    });

    const res = await checkHealth();
    expect(res.status).toBe('healthy');
    expect(res.models_loaded).toBe(false);
  });

  it('returns unavailable status on non-ok HTTP response', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
    });

    const res = await checkHealth();
    expect(res.status).toBe('unavailable');
    expect(res.error).toContain('HTTP 503');
  });

  it('returns unavailable status on non-Error network failure', async () => {
    global.fetch = vi.fn().mockRejectedValue('Generic network failure');

    const res = await checkHealth();
    expect(res.status).toBe('unavailable');
    expect(res.error).toBe('Network error');
  });

  it('aborts on timeout correctly', async () => {
    global.fetch = vi.fn().mockImplementation((_url, opts) => {
      return new Promise((_, reject) => {
        opts.signal.addEventListener('abort', () => {
          const err = new Error('AbortError');
          err.name = 'AbortError';
          reject(err);
        });
      });
    });

    const res = await checkHealth({ timeoutMs: 50 });
    expect(res.status).toBe('unavailable');
    expect(res.error).toContain('timed out');
  });
});
