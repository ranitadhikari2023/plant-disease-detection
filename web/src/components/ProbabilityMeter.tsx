/**
 * web/src/components/ProbabilityMeter.tsx
 * Accessible probability distribution progress meter
 */

import React from 'react';
import styles from './ProbabilityMeter.module.css';

interface ProbabilityMeterProps {
  label: string;
  probability: number; // 0.0 to 1.0 or 0 to 100
  barColor?: string;
  isNormalized?: boolean; // if true, input is 0..1
}

export const ProbabilityMeter: React.FC<ProbabilityMeterProps> = ({
  label,
  probability,
  barColor = 'var(--color-brand)',
  isNormalized = true,
}) => {
  const percent = isNormalized ? Math.max(0, Math.min(100, probability * 100)) : probability;
  const formattedPercent = `${percent.toFixed(1)}%`;

  return (
    <div className={styles.meterContainer}>
      <div className={styles.labelRow}>
        <span className={styles.label}>{label}</span>
        <span className={styles.value}>{formattedPercent}</span>
      </div>

      <div
        className={styles.track}
        role="progressbar"
        aria-valuenow={parseFloat(percent.toFixed(1))}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label} probability: ${formattedPercent}`}
      >
        <div
          className={styles.fill}
          style={{
            width: `${percent}%`,
            backgroundColor: barColor,
          }}
        />
      </div>
    </div>
  );
};
