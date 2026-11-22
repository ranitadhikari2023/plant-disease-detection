import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useLocalHistory, STORAGE_KEY, MAX_HISTORY_RECORDS } from './useLocalHistory';
import type { DiagnosisResponse } from '../api/schema';

describe('useLocalHistory', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const mockDiagnosis: DiagnosisResponse = {
    success: true,
    model_version: 'v2',
    plant_species: 'Potato',
    species_confidence: 0.96,
    disease: 'Early Blight',
    disease_confidence: 0.94,
    diagnosis: 'Potato - Early Blight',
    safety_status: 'confident',
    safety_warnings: [],
    advisory: 'Sample advisory',
    species_probabilities: { Potato: 0.96, Tomato: 0.04 },
    probabilities: { 'Early Blight': 0.94, 'Late Blight': 0.04, Healthy: 0.02 },
    metadata: { execution_time_ms: 120.5, architecture: 'MobileNetV2 + CBAM' },
  };

  it('starts with empty history and saves record successfully', () => {
    const { result } = renderHook(() => useLocalHistory());

    expect(result.current.history).toEqual([]);

    let record: any;
    act(() => {
      record = result.current.saveRecord('specimen1.jpg', mockDiagnosis, 'data:image/jpeg;base64,fake');
    });

    expect(record).toBeDefined();
    expect(record.filename).toBe('specimen1.jpg');
    expect(result.current.history.length).toBe(1);
    expect(result.current.history[0].id).toBe(record.id);

    // Verify localStorage persistence
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    expect(stored.length).toBe(1);
  });

  it('caps history records at 25', () => {
    const { result } = renderHook(() => useLocalHistory());

    act(() => {
      for (let i = 0; i < 30; i++) {
        result.current.saveRecord(`specimen_${i}.jpg`, mockDiagnosis, null);
      }
    });

    expect(result.current.history.length).toBe(MAX_HISTORY_RECORDS);
    expect(result.current.history[0].filename).toBe('specimen_29.jpg');
  });

  it('retrieves record by id with getRecord', () => {
    const { result } = renderHook(() => useLocalHistory());

    let saved: any;
    act(() => {
      saved = result.current.saveRecord('leaf.jpg', mockDiagnosis, null);
    });

    const retrieved = result.current.getRecord(saved.id);
    expect(retrieved).toBeDefined();
    expect(retrieved?.id).toBe(saved.id);

    act(() => {
      result.current.refreshHistory();
    });
    expect(result.current.history.length).toBe(1);
  });

  it('clears all history records', () => {
    const { result } = renderHook(() => useLocalHistory());

    act(() => {
      result.current.saveRecord('leaf.jpg', mockDiagnosis, null);
    });
    expect(result.current.history.length).toBe(1);

    act(() => {
      result.current.clearHistory();
    });
    expect(result.current.history.length).toBe(0);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('handles invalid JSON in localStorage gracefully', () => {
    localStorage.setItem(STORAGE_KEY, '{invalid json');
    const { result } = renderHook(() => useLocalHistory());
    expect(result.current.history).toEqual([]);
  });

  it('handles storage quota exceeded by gracefully trimming thumbnails', () => {
    const { result } = renderHook(() => useLocalHistory());

    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');
    let callCount = 0;
    setItemSpy.mockImplementation((key, val) => {
      callCount++;
      if (callCount === 1) {
        throw new DOMException('QuotaExceededError', 'QuotaExceededError');
      }
      // Second attempt succeeds
    });

    act(() => {
      result.current.saveRecord('heavy.jpg', mockDiagnosis, 'heavy_base64_data');
    });

    expect(result.current.history.length).toBe(1);
    setItemSpy.mockRestore();
  });
});
