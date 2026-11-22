/**
 * web/src/components/TechnicalAccordion.tsx
 * Expandable technical metadata, latency, and raw JSON copy inspector
 */

import React, { useState } from 'react';
import type { DiagnosisResponse } from '../api/schema';
import styles from './TechnicalAccordion.module.css';

interface TechnicalAccordionProps {
  result: DiagnosisResponse;
}

export const TechnicalAccordion: React.FC<TechnicalAccordionProps> = ({ result }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const rawJson = JSON.stringify(result, null, 2);

  const handleCopyJson = async () => {
    try {
      await navigator.clipboard.writeText(rawJson);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Clipboard write failed:', err);
    }
  };

  return (
    <div className={styles.accordionWrap}>
      <button
        type="button"
        className={styles.headerButton}
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
      >
        <span className={styles.headerTitle}>
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
          Technical Telemetry & Model Verification
        </span>
        <span className={styles.arrowIcon} aria-hidden="true">
          {isOpen ? '▲' : '▼'}
        </span>
      </button>

      {isOpen && (
        <div className={styles.body}>
          <div className={styles.grid}>
            <div className={styles.metricItem}>
              <span className={styles.metricLabel}>Model Version</span>
              <span className={styles.metricValue}>{result.model_version || 'v2'}</span>
            </div>
            <div className={styles.metricItem}>
              <span className={styles.metricLabel}>Neural Architecture</span>
              <span className={styles.metricValue}>
                {result.metadata?.architecture || 'MobileNetV2 + CBAM Attention'}
              </span>
            </div>
            <div className={styles.metricItem}>
              <span className={styles.metricLabel}>Inference Latency</span>
              <span className={styles.metricValue}>
                {result.metadata?.execution_time_ms != null
                  ? `${result.metadata.execution_time_ms.toFixed(1)} ms`
                  : '-- ms'}
              </span>
            </div>
            <div className={styles.metricItem}>
              <span className={styles.metricLabel}>Safety Decision</span>
              <span className={styles.metricValue}>{result.safety_status}</span>
            </div>
          </div>

          <div className={styles.rawJsonSection}>
            <div className={styles.rawJsonHeader}>
              <span className={styles.rawJsonLabel}>Raw Diagnosis JSON Payload</span>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleCopyJson}
              >
                {copied ? '✓ Copied!' : 'Copy Raw JSON'}
              </button>
            </div>
            <pre className={styles.jsonPre}>
              <code>{rawJson}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
