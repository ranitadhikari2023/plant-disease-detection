/**
 * web/src/hooks/useThumbnail.ts
 * Generates lightweight (~120px) base64 thumbnail data URLs for local persistence
 */

import { useCallback } from 'react';

export function useThumbnail() {
  const generateThumbnail = useCallback((img: HTMLImageElement, maxSize = 120): string | null => {
    try {
      const canvas = document.createElement('canvas');
      let w = img.naturalWidth || img.width || maxSize;
      let h = img.naturalHeight || img.height || maxSize;

      if (w > h) {
        if (w > maxSize) {
          h = Math.round((h * maxSize) / w);
          w = maxSize;
        }
      } else {
        if (h > maxSize) {
          w = Math.round((w * maxSize) / h);
          h = maxSize;
        }
      }

      canvas.width = Math.max(w, 1);
      canvas.height = Math.max(h, 1);

      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.7);
    } catch (err) {
      console.warn('Failed to generate thumbnail canvas:', err);
      return null;
    }
  }, []);

  return { generateThumbnail };
}
