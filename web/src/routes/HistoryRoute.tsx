/**
 * web/src/routes/HistoryRoute.tsx
 * Foliar scan history log and persistence management
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useLocalHistory } from '../hooks/useLocalHistory';
import { formatPercent } from '../lib/formatters';
import styles from './HistoryRoute.module.css';

interface HistoryRouteProps {
  announce: (message: string) => void;
}

export const HistoryRoute: React.FC<HistoryRouteProps> = ({ announce }) => {
  const navigate = useNavigate();
  const { history, clearHistory } = useLocalHistory();

  const handleClear = () => {
    if (window.confirm('Are you sure you want to clear all local diagnostic scan records?')) {
      clearHistory();
      announce('Diagnostic scan history cleared.');
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>Diagnostic History Log</h1>
          <p className={styles.subtitle}>
            Preserved locally in browser storage (capped at 25 recent foliar examinations).
          </p>
        </div>

        <div className={styles.headerActions}>
          <span className={styles.countBadge}>
            {history.length} scan{history.length === 1 ? '' : 's'} recorded
          </span>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={handleClear}
            disabled={history.length === 0}
          >
            Clear History
          </button>
        </div>
      </header>

      {history.length === 0 ? (
        <div className={styles.emptyCard}>
          <div className={styles.emptyIcon} aria-hidden="true">
            📂
          </div>
          <h3 className={styles.emptyTitle}>No Diagnostic Records Stored</h3>
          <p className={styles.emptyDesc}>
            Your local scan history is currently empty. Run an foliar specimen diagnosis to record
            predictions here.
          </p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => navigate('/scan')}
          >
            Start Foliar Scan
          </button>
        </div>
      ) : (
        <div className={styles.list}>
          {history.map((record) => {
            let statusClass = styles.confident;
            if (record.safetyStatus === 'late_blight_uncertain') {
              statusClass = styles.lateBlight;
            } else if (record.safetyStatus === 'species_uncertain') {
              statusClass = styles.speciesUncertain;
            } else if (record.safetyStatus === 'disease_uncertain') {
              statusClass = styles.diseaseUncertain;
            }

            const dateStr = new Date(record.timestamp).toLocaleString();

            return (
              <div key={record.id} className={styles.historyCard}>
                <div className={styles.leftGroup}>
                  <div className={styles.thumbWrap}>
                    {record.thumbnail ? (
                      <img
                        src={record.thumbnail}
                        alt={record.filename || 'Specimen thumbnail'}
                        className={styles.thumbImg}
                      />
                    ) : (
                      <div className={styles.thumbFallback} aria-hidden="true">
                        🍃
                      </div>
                    )}
                  </div>

                  <div className={styles.infoGroup}>
                    <div className={styles.timeText}>{dateStr}</div>
                    <h3 className={styles.cardHeading}>
                      {record.diagnosis || `${record.species} — ${record.disease}`}
                    </h3>
                    <div className={styles.metaText}>
                      File: <strong>{record.filename}</strong> · Confidence:{' '}
                      <strong>{formatPercent(record.confidence, 1)}</strong>
                    </div>
                  </div>
                </div>

                <div className={styles.rightGroup}>
                  <span className={`${styles.statusBadge} ${statusClass}`}>
                    {record.safetyStatus}
                  </span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => navigate(`/result/${record.id}`)}
                  >
                    View Report
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
