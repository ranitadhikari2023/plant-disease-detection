/**
 * web/src/components/SpeciesCard.tsx
 * Botanical Species Identification Card (Tier 1 Botanist Classification)
 */

import React from 'react';
import { SPECIES_SCIENTIFIC_MAP, formatPercent } from '../lib/formatters';
import { ProbabilityMeter } from './ProbabilityMeter';
import styles from './SpeciesCard.module.css';

interface SpeciesCardProps {
  species: string | null;
  confidence: number;
  safetyStatus: string;
  probabilities?: Record<string, number>;
}

export const SpeciesCard: React.FC<SpeciesCardProps> = ({
  species,
  confidence,
  safetyStatus,
  probabilities = {},
}) => {
  const isGate1Passed = safetyStatus !== 'species_uncertain';
  const displaySpecies = species || 'Unknown';
  const scientificName = SPECIES_SCIENTIFIC_MAP[displaySpecies] || 'Solanaceae family';

  const potatoProb = probabilities['Potato'] ?? 0;
  const tomatoProb = probabilities['Tomato'] ?? 0;

  return (
    <div className={styles.card} role="region" aria-label="Tier 1 Species Classification">
      <div className={styles.header}>
        <div className={styles.tierBadge}>Tier 1 · MobileNetV2 Botanist</div>
        <span
          className={`${styles.gatePill} ${
            isGate1Passed ? styles.gatePassed : styles.gateTriggered
          }`}
        >
          {isGate1Passed ? 'Gate 1 Passed (≥90%)' : 'Gate 1 Triggered (<90%)'}
        </span>
      </div>

      <div className={styles.mainInfo}>
        <div className={styles.scientific}>{scientificName}</div>
        <div className={styles.predictionRow}>
          <h3 className={styles.speciesName}>{displaySpecies}</h3>
          <span className={styles.confidenceText}>{formatPercent(confidence, 2)}</span>
        </div>
      </div>

      <div className={styles.metersSection}>
        <div className={styles.metersTitle}>Botanist Class Probabilities</div>
        <div className={styles.metersList}>
          <ProbabilityMeter
            label="Potato (Solanum tuberosum)"
            probability={potatoProb}
            barColor="var(--meter-species-potato)"
          />
          <ProbabilityMeter
            label="Tomato (Solanum lycopersicum)"
            probability={tomatoProb}
            barColor="var(--meter-species-tomato)"
          />
        </div>
      </div>
    </div>
  );
};
