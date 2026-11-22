/**
 * web/src/test/setup.ts
 * Vitest and React Testing Library environment setup
 */

import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

// Automatically cleanup DOM after each test
afterEach(() => {
  cleanup();
});

// Polyfill window.scrollTo
Object.defineProperty(window, 'scrollTo', {
  value: () => {},
  writable: true,
});

// Polyfill navigator.clipboard
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: async () => {},
    readText: async () => '',
  },
  writable: true,
});

// Polyfill HTMLCanvasElement for JSDOM
HTMLCanvasElement.prototype.getContext = function () {
  return {
    drawImage: () => {},
  } as unknown as CanvasRenderingContext2D;
};

HTMLCanvasElement.prototype.toDataURL = function () {
  return 'data:image/jpeg;base64,mockThumbnailData';
};
