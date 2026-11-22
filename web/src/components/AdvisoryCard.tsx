/**
 * web/src/components/AdvisoryCard.tsx
 * Actionable Agronomic Decision Support Card
 */

import React from 'react';
import { getSafetyStatusConfig } from '../lib/safetyStatus';
import styles from './AdvisoryCard.module.css';

interface AdvisoryCardProps {
  advisory: string;
  safetyStatus: string;
}

export const AdvisoryCard: React.FC<AdvisoryCardProps> = ({ advisory, safetyStatus }) => {
  const config = getSafetyStatusConfig(safetyStatus);
  const typeClass =
    config.advisoryType === 'urgent'
      ? styles.urgent
      : config.advisoryType === 'warning'
      ? styles.warning
      : styles.default;

  return (
    <div className={styles.card} role="region" aria-label="Agronomic Advisory Notice">
      <div className={styles.header}>
        <div className={styles.badgeWrap}>
          <span className={styles.icon} aria-hidden="true">
            🌱
          </span>
          <h3 className={styles.title}>Actionable Agronomic Advisory</h3>
        </div>
        <span className={styles.noticeTag}>Decision Support</span>
      </div>

      <div className={`${styles.textBox} ${typeClass}`}>
        <p className={styles.text}>{advisory || 'No specific foliar advisory returned.'}</p>
      </div>

      <div className={styles.disclaimer}>
        <strong>Agricultural Protocol Notice:</strong> Automated foliar screening provides
        diagnostic decision-support and does not replace on-site plant pathology consultation. Always
        strictly adhere to locally registered pesticide label instructions and regional agricultural
        extension directives.
      </div>
    </div>
  );
};
