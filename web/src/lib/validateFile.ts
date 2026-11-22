/**
 * web/src/lib/validateFile.ts
 * Validation for uploaded foliar specimen files
 */

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 Megabytes

export const ALLOWED_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
]);

export const ALLOWED_EXTENSIONS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
]);

export interface FileValidationResult {
  valid: boolean;
  error: string | null;
}

/**
 * Validates file size (max 10MB) and format (JPEG, PNG, WebP).
 * Returns { valid: true, error: null } if file passes checks.
 */
export function validateFile(file: File | null | undefined): FileValidationResult {
  if (!file) {
    return { valid: false, error: 'No file was selected.' };
  }

  // 1. Enforce 10 MB size limit
  if (file.size > MAX_FILE_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `File size (${sizeMb} MB) exceeds maximum allowed limit of 10 MB.`,
    };
  }

  // 2. Reject empty files
  if (file.size === 0) {
    return {
      valid: false,
      error: 'Uploaded file is empty (0 bytes).',
    };
  }

  // 3. MIME and Extension validation
  const mime = (file.type || '').toLowerCase();
  const dotIndex = file.name.lastIndexOf('.');
  const ext = dotIndex !== -1 ? file.name.slice(dotIndex).toLowerCase() : '';

  const hasValidMime = ALLOWED_MIME_TYPES.has(mime);
  const hasValidExt = ALLOWED_EXTENSIONS.has(ext);

  if (!hasValidMime && !hasValidExt) {
    const displayFormat = file.type || ext || 'unknown';
    return {
      valid: false,
      error: `Unsupported image format (${displayFormat}). Supported formats: JPEG, PNG, WebP.`,
    };
  }

  return { valid: true, error: null };
}
