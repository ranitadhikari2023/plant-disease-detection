/**
 * web/src/components/RecentScans.tsx
 * Horizontal strip / grid of recent diagnostic scans from local history
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { HistoryRecord } from '../hooks/useLocalHistory';
import { formatRelativeTime } from '../lib/formatters';
import styles from './RecentScans.module.css';

interface RecentScansProps {
  scans: HistoryRecord[];
}

export const RecentScans: React.FC<RecentScansProps> = ({ scans }) => {
  const navigate = useNavigate();
  const recentItems = scans.slice(0, 6);

  if (recentItems.length === 0) {
    return (
      <div className={styles.emptyContainer} role="region" aria-label="Recent Scans">
        <div className={styles.emptyIcon} aria-hidden="true">
          🌱
        </div>
        <div className={styles.emptyTitle}>No Foliar Scans Recorded Yet</div>
        <p className={styles.emptyText}>
          Upload or capture a leaf photo above to generate your first diagnostic pathology report.
        </p>
      </div>
    );
  }

  return (
    <div className={styles.container} role="region" aria-label="Recent Foliar Scans">
      <div className={styles.headerRow}>
        <div className={styles.titleWrap}>
          <h3 className={styles.sectionTitle}>Recent Foliar Scans</h3>
          <span className={styles.countBadge}>
            {scans.length} scan{scans.length === 1 ? '' : 's'} recorded
          </span>
        </div>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={() => navigate('/history')}
        >
          View All History
        </button>
      </div>

      <div className={styles.grid}>
        {recentItems.map((scan) => {
          const confVal =
            typeof scan.confidence === 'number'
              ? `${(scan.confidence * 100).toFixed(1)}%`
              : '--';

          let statusClass = styles.confident;
          if (scan.safetyStatus === 'late_blight_uncertain') {
            statusClass = styles.lateBlight;
          } else if (scan.safetyStatus === 'species_uncertain') {
            statusClass = styles.speciesUncertain;
          } else if (scan.safetyStatus === 'disease_uncertain') {
            statusClass = styles.diseaseUncertain;
          }

          return (
            <div
              key={scan.id}
              className={styles.card}
              role="button"
              tabIndex={0}
              aria-label={`View diagnostic report for ${scan.diagnosis || scan.species}`}
              onClick={() => navigate(`/result/${scan.id}`)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  navigate(`/result/${scan.id}`);
                }
              }}
            >
              <div className={styles.thumbWrap}>
                {scan.thumbnail ? (
                  <img
                    src={scan.thumbnail}
                    alt={scan.filename || 'Leaf thumbnail'}
                    className={styles.thumbImg}
                  />
                ) : (
                  <div className={styles.thumbFallback} aria-hidden="true">
                    🍃
                  </div>
                )}
              </div>

              <div className={styles.info}>
                <div className={styles.cardTitle}>{scan.diagnosis || scan.species}</div>
                <div className={styles.cardMeta}>
                  {formatRelativeTime(scan.timestamp)} · {confVal}
                </div>
                <div className={styles.cardFooter}>
                  <span className={`${styles.statusBadge} ${statusClass}`}>
                    {scan.safetyStatus}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
