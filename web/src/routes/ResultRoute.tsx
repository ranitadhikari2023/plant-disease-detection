/**
 * web/src/routes/ResultRoute.tsx
 * Comprehensive Diagnostic Pathology Report View
 */

import React, { useState } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useLocalHistory } from '../hooks/useLocalHistory';
import { SafetyBanner } from '../components/SafetyBanner';
import { SpeciesCard } from '../components/SpeciesCard';
import { DiseaseCard } from '../components/DiseaseCard';
import { AdvisoryCard } from '../components/AdvisoryCard';
import { TechnicalAccordion } from '../components/TechnicalAccordion';
import type { DiagnosisResponse } from '../api/schema';
import styles from './ResultRoute.module.css';

interface ResultRouteProps {
  announce: (message: string) => void;
}

export const ResultRoute: React.FC<ResultRouteProps> = ({ announce }) => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { getRecord } = useLocalHistory();

  const [copiedSummary, setCopiedSummary] = useState(false);

  // Retrieve record from history by id, or from router location state
  const record = id ? getRecord(id) : undefined;
  const stateResult = (location.state as { resultData?: DiagnosisResponse })?.resultData;
  const result: DiagnosisResponse | undefined = record?.resultData || stateResult;

  if (!result) {
    return (
      <div className={styles.notFoundCard}>
        <div className={styles.notFoundIcon} aria-hidden="true">
          🔍
        </div>
        <h2 className={styles.notFoundTitle}>Diagnostic Report Not Found</h2>
        <p className={styles.notFoundDesc}>
          The requested pathology report ID ({id}) could not be located in your browser session or
          local history log.
        </p>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => navigate('/scan')}
        >
          Return to Scan Workspace
        </button>
      </div>
    );
  }

  const filename = record?.filename || 'foliar_specimen.jpg';
  const timestamp = record?.timestamp
    ? new Date(record.timestamp).toLocaleString()
    : new Date().toLocaleString();

  const handleCopySummary = async () => {
    const summaryText = `[LeafScan V2 Diagnostic Pathology Report]
Evaluated: ${timestamp}
Specimen: ${filename}
Species: ${result.plant_species || 'Unknown'} (${((result.species_confidence || 0) * 100).toFixed(1)}% confidence)
Diagnosis: ${result.diagnosis || 'Abstained'}
Specialist Confidence: ${result.disease_confidence != null ? ((result.disease_confidence * 100).toFixed(1) + '%') : 'N/A'}
Safety Decision: ${result.safety_status}
Advisory: ${result.advisory || 'None'}
Architecture: MobileNetV2 + CBAM Attention (Opset 17)`;

    try {
      await navigator.clipboard.writeText(summaryText);
      setCopiedSummary(true);
      announce('Diagnostic summary copied to clipboard.');
      setTimeout(() => setCopiedSummary(false), 2000);
    } catch (err) {
      console.warn('Clipboard write failed:', err);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.timestampBadge}>Evaluated on {timestamp}</div>
          <h1 className={styles.title}>Foliar Pathology Report</h1>
        </div>

        <div className={styles.headerActions}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleCopySummary}
          >
            {copiedSummary ? '✓ Copied!' : 'Copy Summary'}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => navigate('/scan')}
          >
            New Scan
          </button>
        </div>
      </header>

      <div className={styles.reportGrid}>
        {/* Left Column: Specimen Overview */}
        <aside className={styles.specimenCol}>
          <div className={styles.specimenCard}>
            <div className={styles.imageWrap}>
              {record?.thumbnail ? (
                <img
                  src={record.thumbnail}
                  alt={`Analyzed foliar specimen: ${result.diagnosis || 'Leaf'}`}
                  className={styles.specimenImg}
                />
              ) : (
                <div className={styles.imageFallback} aria-hidden="true">
                  🍃
                </div>
              )}
            </div>

            <div className={styles.specimenDetails}>
              <div className={styles.specimenFilename} title={filename}>
                {filename}
              </div>
              <div className={styles.specimenMeta}>
                Classification Model: {result.model_version || 'v2'}
              </div>
            </div>
          </div>
        </aside>

        {/* Right Column: Diagnostic Cards */}
        <main className={styles.mainCol}>
          <SafetyBanner
            status={result.safety_status}
            warnings={result.safety_warnings}
            species={result.plant_species}
            disease={result.disease}
          />

          <SpeciesCard
            species={result.plant_species}
            confidence={result.species_confidence}
            safetyStatus={result.safety_status}
            probabilities={result.species_probabilities}
          />

          <DiseaseCard
            disease={result.disease}
            confidence={result.disease_confidence}
            safetyStatus={result.safety_status}
            probabilities={result.probabilities}
          />

          <AdvisoryCard advisory={result.advisory} safetyStatus={result.safety_status} />

          <TechnicalAccordion result={result} />
        </main>
      </div>
    </div>
  );
};
