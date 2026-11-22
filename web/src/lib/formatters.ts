/**
 * web/src/lib/formatters.ts
 * Data formatting helpers for LeafScan V2
 */

/**
 * Formats a byte number into a human-readable size string (e.g. "1.2 MB").
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const val = bytes / Math.pow(k, i);
  return `${val.toFixed(i === 0 ? 0 : 1)} ${sizes[i]}`;
}

/**
 * Formats a relative timestamp from an ISO string (e.g. "2 hours ago", "Just now").
 */
export function formatRelativeTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return isoString;

    const now = new Date();
    const diffSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 3600) {
      const minutes = Math.floor(diffSeconds / 60);
      return `${minutes}m ago`;
    }
    if (diffSeconds < 86400) {
      const hours = Math.floor(diffSeconds / 3600);
      return `${hours}h ago`;
    }
    const days = Math.floor(diffSeconds / 86400);
    if (days < 7) return `${days}d ago`;

    return date.toLocaleDateString();
  } catch {
    return isoString;
  }
}

/**
 * Formats a decimal (0.0 to 1.0) into a percentage string (e.g. "94.2%").
 */
export function formatPercent(val: number | null | undefined, decimals = 1): string {
  if (val == null || isNaN(val)) return '--%';
  return `${(val * 100).toFixed(decimals)}%`;
}

/**
 * Maps plant species to their scientific botanical binomials.
 */
export const SPECIES_SCIENTIFIC_MAP: Record<string, string> = {
  Potato: 'Solanum tuberosum',
  Tomato: 'Solanum lycopersicum',
};
