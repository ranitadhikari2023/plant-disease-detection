import { describe, it, expect, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useThumbnail } from './useThumbnail';

describe('useThumbnail', () => {
  it('generates a thumbnail data url from an image element', () => {
    const { result } = renderHook(() => useThumbnail());

    const fakeImg = document.createElement('img');
    Object.defineProperty(fakeImg, 'naturalWidth', { value: 600 });
    Object.defineProperty(fakeImg, 'naturalHeight', { value: 400 });

    const thumb = result.current.generateThumbnail(fakeImg, 120);
    expect(thumb).toBe('data:image/jpeg;base64,mockThumbnailData');
  });

  it('handles image where height > width', () => {
    const { result } = renderHook(() => useThumbnail());

    const fakeImg = document.createElement('img');
    Object.defineProperty(fakeImg, 'naturalWidth', { value: 300 });
    Object.defineProperty(fakeImg, 'naturalHeight', { value: 900 });

    const thumb = result.current.generateThumbnail(fakeImg, 120);
    expect(thumb).toBe('data:image/jpeg;base64,mockThumbnailData');
  });

  it('returns null when canvas context is unavailable', () => {
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = () => null;

    const { result } = renderHook(() => useThumbnail());
    const fakeImg = document.createElement('img');
    const thumb = result.current.generateThumbnail(fakeImg, 120);
    expect(thumb).toBeNull();

    HTMLCanvasElement.prototype.getContext = origGetContext;
  });
});
