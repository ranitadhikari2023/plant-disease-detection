/**
 * web/src/components/SafetyBanner.tsx
 * Hierarchical safety status banner with iconography, status pill, heading, message, and warnings
 */

import React from 'react';
import { getSafetyStatusConfig } from '../lib/safetyStatus';
import styles from './SafetyBanner.module.css';

interface SafetyBannerProps {
  status: string | null | undefined;
  warnings?: string[];
  species?: string | null;
  disease?: string | null;
}

export const SafetyBanner: React.FC<SafetyBannerProps> = ({
  status,
  warnings = [],
  species,
  disease,
}) => {
  const config = getSafetyStatusConfig(status);

  return (
    <div
      className={`${styles.banner} ${styles[config.className] || ''}`}
      role="region"
      aria-label="Diagnostic Safety Assessment"
    >
      <div className={styles.headerRow}>
        <div className={styles.iconCircle} aria-hidden="true">
          {config.icon}
        </div>
        <div className={styles.titleWrap}>
          <div className={styles.pillRow}>
            <span className={styles.pill}>{config.pillText}</span>
          </div>
          <h2 className={styles.heading}>{config.heading(species ?? null, disease ?? null)}</h2>
        </div>
      </div>

      <p className={styles.message}>{config.message}</p>

      {warnings && warnings.length > 0 && (
        <ul className={styles.warningsList} aria-label="Safety warnings">
          {warnings.map((warn, index) => (
            <li key={index}>{warn}</li>
          ))}
        </ul>
      )}
    </div>
  );
};
