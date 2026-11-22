/**
 * web/src/components/DiseaseCard.tsx
 * Foliar Pathology Specialist Card (Tier 2 CBAM Crop Specialist)
 */

import React from 'react';
import { formatPercent } from '../lib/formatters';
import { ProbabilityMeter } from './ProbabilityMeter';
import styles from './DiseaseCard.module.css';

interface DiseaseCardProps {
  disease: string | null;
  confidence: number | null;
  safetyStatus: string;
  probabilities?: Record<string, number>;
}

export const DiseaseCard: React.FC<DiseaseCardProps> = ({
  disease,
  confidence,
  safetyStatus,
  probabilities = {},
}) => {
  const isAbstained = !disease || safetyStatus === 'species_uncertain';

  let gatePillLabel = 'Specialist Confident';
  let gatePillClass = styles.confident;

  if (isAbstained) {
    gatePillLabel = 'Abstained by Gate 1';
    gatePillClass = styles.abstained;
  } else if (safetyStatus === 'late_blight_uncertain') {
    gatePillLabel = 'Late Blight Warning (<90%)';
    gatePillClass = styles.lateBlight;
  } else if (safetyStatus === 'disease_uncertain') {
    gatePillLabel = 'Marginal Confidence (<70%)';
    gatePillClass = styles.diseaseUncertain;
  }

  const ebProb = probabilities['Early Blight'] ?? 0;
  const lbProb = probabilities['Late Blight'] ?? 0;
  const healthyProb = probabilities['Healthy'] ?? 0;

  return (
    <div className={styles.card} role="region" aria-label="Tier 2 Foliar Pathology Diagnosis">
      <div className={styles.header}>
        <div className={styles.tierBadge}>Tier 2 · CBAM Crop Specialist</div>
        <span className={`${styles.gatePill} ${gatePillClass}`}>{gatePillLabel}</span>
      </div>

      {isAbstained ? (
        <div className={styles.abstainedContent}>
          <div className={styles.abstainedIcon} aria-hidden="true">
            🛡️
          </div>
          <h3 className={styles.abstainedHeading}>Pathology Assessment Withheld</h3>
          <p className={styles.abstainedText}>
            Because botanical species classification fell below the defensive 90% confidence
            boundary, the Tier 2 crop pathology specialist was purposefully abstained to prevent
            hazardous agronomic misdiagnosis.
          </p>
        </div>
      ) : (
        <>
          <div className={styles.mainInfo}>
            <div className={styles.subtext}>Detected Foliar Condition</div>
            <div className={styles.predictionRow}>
              <h3 className={styles.diseaseName}>{disease}</h3>
              <span className={styles.confidenceText}>
                {confidence != null ? formatPercent(confidence, 2) : '--%'}
              </span>
            </div>
          </div>

          <div className={styles.metersSection}>
            <div className={styles.metersTitle}>Specialist Foliar Class Distribution</div>
            <div className={styles.metersList}>
              <ProbabilityMeter
                label="Early Blight (Alternaria solani)"
                probability={ebProb}
                barColor="var(--meter-early-blight)"
              />
              <ProbabilityMeter
                label="Late Blight (Phytophthora infestans)"
                probability={lbProb}
                barColor="var(--meter-late-blight)"
              />
              <ProbabilityMeter
                label="Healthy Foliar Specimen"
                probability={healthyProb}
                barColor="var(--meter-healthy)"
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
};
