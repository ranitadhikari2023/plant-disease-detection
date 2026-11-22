import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDragCounter } from './useDragCounter';

describe('useDragCounter', () => {
  it('increments and decrements dragCounter correctly', () => {
    const { result } = renderHook(() => useDragCounter());

    expect(result.current.isDragging).toBe(false);

    const mockEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as React.DragEvent;

    act(() => {
      result.current.onDragEnter(mockEvent);
    });
    expect(result.current.isDragging).toBe(true);

    act(() => {
      result.current.onDragOver(mockEvent);
    });
    expect(result.current.isDragging).toBe(true);

    act(() => {
      result.current.onDragEnter(mockEvent); // nested child enter
    });
    expect(result.current.isDragging).toBe(true);

    act(() => {
      result.current.onDragLeave(mockEvent); // first child leave
    });
    expect(result.current.isDragging).toBe(true);

    act(() => {
      result.current.onDragLeave(mockEvent); // container leave
    });
    expect(result.current.isDragging).toBe(false);
  });

  it('prevents dragCounter from going below zero', () => {
    const { result } = renderHook(() => useDragCounter());
    const mockEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as React.DragEvent;

    act(() => {
      result.current.onDragLeave(mockEvent);
    });
    expect(result.current.isDragging).toBe(false);
  });

  it('handles drop and resets counter to zero', () => {
    const { result } = renderHook(() => useDragCounter());
    const handleFiles = vi.fn();

    const mockFile = new File(['test'], 'leaf.jpg', { type: 'image/jpeg' });
    const mockDropEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      dataTransfer: {
        files: [mockFile],
      },
    } as unknown as React.DragEvent;

    act(() => {
      result.current.onDrop(mockDropEvent, handleFiles);
    });

    expect(result.current.isDragging).toBe(false);
    expect(handleFiles).toHaveBeenCalled();
  });

  it('resets drag state when resetDrag is called', () => {
    const { result } = renderHook(() => useDragCounter());
    const mockEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as React.DragEvent;

    act(() => {
      result.current.onDragEnter(mockEvent);
    });
    expect(result.current.isDragging).toBe(true);

    act(() => {
      result.current.resetDrag();
    });
    expect(result.current.isDragging).toBe(false);
  });
});
