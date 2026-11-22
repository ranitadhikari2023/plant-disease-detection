/**
 * web/src/hooks/useLocalHistory.ts
 * LocalStorage persistence for diagnostic history with 25-record cap and quota handling
 */

import { useState, useEffect, useCallback } from 'react';
import type { DiagnosisResponse } from '../api/schema';

export const STORAGE_KEY = 'leafscan_v2_history_v1';
export const MAX_HISTORY_RECORDS = 25;

export interface HistoryRecord {
  id: string;
  timestamp: string;
  filename: string;
  species: string;
  disease: string;
  confidence: number;
  safetyStatus: string;
  diagnosis: string;
  thumbnail: string | null;
  resultData: DiagnosisResponse;
}

export function useLocalHistory() {
  const [history, setHistory] = useState<HistoryRecord[]>([]);

  const loadHistory = useCallback((): HistoryRecord[] => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed;
      }
      return [];
    } catch (err) {
      console.warn('Failed to load local scan history:', err);
      return [];
    }
  }, []);

  useEffect(() => {
    setHistory(loadHistory());
  }, [loadHistory]);

  const saveRecord = useCallback(
    (filename: string, result: DiagnosisResponse, thumbnail: string | null): HistoryRecord | null => {
      try {
        const existing = loadHistory();
        const conf =
          result.disease_confidence != null
            ? result.disease_confidence
            : result.species_confidence;

        const newEntry: HistoryRecord = {
          id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
          timestamp: new Date().toISOString(),
          filename: filename || 'leaf_specimen.jpg',
          species: result.plant_species || 'Unknown',
          disease: result.disease || 'Abstained',
          confidence: conf,
          safetyStatus: result.safety_status || 'confident',
          diagnosis: result.diagnosis || 'Plant Screening',
          thumbnail: thumbnail || null,
          resultData: result,
        };

        const updated = [newEntry, ...existing];
        if (updated.length > MAX_HISTORY_RECORDS) {
          updated.length = MAX_HISTORY_RECORDS;
        }

        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
        } catch (storageError) {
          console.warn('Storage quota warning when writing full history:', storageError);
          // Gracefully strip thumbnails of older entries if quota exceeded
          const compact = updated.map((r, idx) =>
            idx === 0 ? r : { ...r, thumbnail: null }
          );
          try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
          } catch (e2) {
            console.error('LocalStorage write failed completely:', e2);
          }
        }

        setHistory(updated);
        return newEntry;
      } catch (err) {
        console.warn('Failed to save diagnosis to history:', err);
        return null;
      }
    },
    [loadHistory]
  );

  const clearHistory = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEY);
      setHistory([]);
    } catch (err) {
      console.warn('Error clearing localStorage:', err);
    }
  }, []);

  const getRecord = useCallback(
    (id: string): HistoryRecord | undefined => {
      const records = loadHistory();
      return records.find((r) => r.id === id);
    },
    [loadHistory]
  );

  return {
    history,
    saveRecord,
    clearHistory,
    getRecord,
    refreshHistory: () => setHistory(loadHistory()),
  };
}
