import { describe, it, expect } from 'vitest';
import { validateFile, MAX_FILE_SIZE_BYTES } from './validateFile';

describe('validateFile', () => {
  it('rejects null and undefined file inputs', () => {
    expect(validateFile(null).valid).toBe(false);
    expect(validateFile(null).error).toContain('No file was selected');
    expect(validateFile(undefined).valid).toBe(false);
  });

  it('rejects empty files (0 bytes)', () => {
    const emptyFile = new File([], 'empty.jpg', { type: 'image/jpeg' });
    const res = validateFile(emptyFile);
    expect(res.valid).toBe(false);
    expect(res.error).toContain('0 bytes');
  });

  it('rejects oversized files (>10MB)', () => {
    const bigBlob = new Blob([new Uint8Array(MAX_FILE_SIZE_BYTES + 100)]);
    const bigFile = new File([bigBlob], 'huge.png', { type: 'image/png' });
    const res = validateFile(bigFile);
    expect(res.valid).toBe(false);
    expect(res.error).toContain('exceeds maximum allowed limit of 10 MB');
  });

  it('rejects unsupported image extensions and MIME types', () => {
    const textFile = new File(['hello'], 'document.pdf', { type: 'application/pdf' });
    const res = validateFile(textFile);
    expect(res.valid).toBe(false);
    expect(res.error).toContain('Unsupported image format');
  });

  it('accepts valid JPEG, PNG, and WebP files', () => {
    const validJpg = new File(['fake-jpg'], 'leaf.jpg', { type: 'image/jpeg' });
    expect(validateFile(validJpg)).toEqual({ valid: true, error: null });

    const validPng = new File(['fake-png'], 'leaf.png', { type: 'image/png' });
    expect(validateFile(validPng)).toEqual({ valid: true, error: null });

    const validWebp = new File(['fake-webp'], 'leaf.webp', { type: 'image/webp' });
    expect(validateFile(validWebp)).toEqual({ valid: true, error: null });
  });

  it('accepts valid file with missing MIME type but valid extension', () => {
    const validByExt = new File(['fake'], 'specimen.JPEG', { type: '' });
    expect(validateFile(validByExt).valid).toBe(true);
  });
});
