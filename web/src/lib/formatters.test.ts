import { describe, it, expect } from 'vitest';
import { formatBytes, formatRelativeTime, formatPercent, SPECIES_SCIENTIFIC_MAP } from './formatters';

describe('formatters', () => {
  it('formatBytes converts bytes to readable string', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1024 * 1024 * 2.5)).toBe('2.5 MB');
    expect(formatBytes(1024 * 1024 * 1024)).toBe('1.0 GB');
  });

  it('formatRelativeTime returns accurate relative time', () => {
    const now = new Date();
    expect(formatRelativeTime(now.toISOString())).toBe('Just now');

    const tenMinAgo = new Date(Date.now() - 10 * 60 * 1000);
    expect(formatRelativeTime(tenMinAgo.toISOString())).toBe('10m ago');

    const twoHoursAgo = new Date(Date.now() - 2 * 3600 * 1000);
    expect(formatRelativeTime(twoHoursAgo.toISOString())).toBe('2h ago');

    const threeDaysAgo = new Date(Date.now() - 3 * 86400 * 1000);
    expect(formatRelativeTime(threeDaysAgo.toISOString())).toBe('3d ago');

    const tenDaysAgo = new Date(Date.now() - 10 * 86400 * 1000);
    expect(formatRelativeTime(tenDaysAgo.toISOString())).toBe(tenDaysAgo.toLocaleDateString());

    expect(formatRelativeTime('invalid-date')).toBe('invalid-date');
  });

  it('formatPercent formats decimal numbers accurately', () => {
    expect(formatPercent(0.9542, 1)).toBe('95.4%');
    expect(formatPercent(0.9542, 2)).toBe('95.42%');
    expect(formatPercent(null)).toBe('--%');
    expect(formatPercent(undefined)).toBe('--%');
    expect(formatPercent(NaN)).toBe('--%');
  });

  it('SPECIES_SCIENTIFIC_MAP maps Potato and Tomato correctly', () => {
    expect(SPECIES_SCIENTIFIC_MAP['Potato']).toBe('Solanum tuberosum');
    expect(SPECIES_SCIENTIFIC_MAP['Tomato']).toBe('Solanum lycopersicum');
  });
});
