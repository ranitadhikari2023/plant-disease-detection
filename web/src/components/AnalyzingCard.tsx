/**
 * web/src/components/AnalyzingCard.tsx
 * In-place indeterminate progress indicator during neural inference
 */

import React, { useState, useEffect } from 'react';
import styles from './AnalyzingCard.module.css';

export const INFERENCE_STEPS = [
  {
    title: 'Preparing Foliar Specimen...',
    desc: 'Verifying image resolution, normalization, and RGB raster integrity.',
  },
  {
    title: 'Extracting Visual Features...',
    desc: 'Executing MobileNetV2 botanist backbone to assess plant morphology.',
  },
  {
    title: 'Evaluating Botanical Identity (Gate 1)...',
    desc: 'Assessing Solanum tuberosum vs Solanum lycopersicum confidence threshold.',
  },
  {
    title: 'Processing CBAM Attention Submodules...',
    desc: 'Refining channel and spatial attention weights for foliar necrotic lesions.',
  },
  {
    title: 'Synthesizing Diagnostic Pathology Report...',
    desc: 'Evaluating Late Blight escalation boundaries and agronomic decision support.',
  },
];

export const AnalyzingCard: React.FC = () => {
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStepIndex((prev) => (prev + 1) % INFERENCE_STEPS.length);
    }, 1800);
    return () => clearInterval(timer);
  }, []);

  const currentStep = INFERENCE_STEPS[stepIndex];

  return (
    <div
      className={styles.card}
      role="status"
      aria-live="polite"
      aria-label="Analyzing foliar specimen"
    >
      <div className={styles.spinnerWrap}>
        <div className={styles.spinner} aria-hidden="true" />
      </div>

      <div className={styles.content}>
        <h3 className={styles.title}>{currentStep.title}</h3>
        <p className={styles.desc}>{currentStep.desc}</p>
      </div>

      <div className={styles.progressBarWrap} aria-hidden="true">
        <div className={styles.progressBarIndeterminate} />
      </div>

      <p className={styles.note}>
        Dual-gate safety validation active. Inference typically completes in 200–500 ms.
      </p>
    </div>
  );
};
