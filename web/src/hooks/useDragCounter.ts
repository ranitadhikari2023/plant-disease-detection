/**
 * web/src/hooks/useDragCounter.ts
 * Manages drag-over state without child element flickering using an integer counter
 */

import { useState, useCallback } from 'react';

export function useDragCounter() {
  const [dragCounter, setDragCounter] = useState(0);

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter((prev) => prev + 1);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter((prev) => Math.max(0, prev - 1));
  }, []);

  const onDrop = useCallback((e: React.DragEvent, handleFiles: (files: FileList) => void) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter(0);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const resetDrag = useCallback(() => {
    setDragCounter(0);
  }, []);

  return {
    isDragging: dragCounter > 0,
    onDragEnter,
    onDragOver,
    onDragLeave,
    onDrop,
    resetDrag,
  };
}
