/**
 * web/src/components/PreviewPanel.tsx
 * Specimen preview panel with dimensions, file details, and workflow actions
 */

import React from 'react';
import { formatBytes } from '../lib/formatters';
import styles from './PreviewPanel.module.css';

interface PreviewPanelProps {
  file: File;
  previewUrl: string;
  dimensions: { width: number | null; height: number | null };
  onRunDiagnosis: () => void;
  onChangeImage: () => void;
  onRetakePhoto: () => void;
  onCancel: () => void;
  isDiagnosing?: boolean;
  isBackendHealthy?: boolean;
}

export const PreviewPanel: React.FC<PreviewPanelProps> = ({
  file,
  previewUrl,
  dimensions,
  onRunDiagnosis,
  onChangeImage,
  onRetakePhoto,
  onCancel,
  isDiagnosing = false,
  isBackendHealthy = true,
}) => {

  const dimText =
    dimensions.width && dimensions.height
      ? `${dimensions.width} × ${dimensions.height} px`
      : 'Calculating...';

  return (
    <div className={styles.card} role="region" aria-label="Selected leaf specimen preview">
      <div className={styles.mediaContainer}>
        <img
          src={previewUrl}
          alt={`Preview of selected foliar specimen: ${file.name}`}
          className={styles.image}
        />
      </div>

      <div className={styles.metaRow}>
        <div className={styles.metaInfo}>
          <span className={styles.filename} title={file.name}>
            {file.name}
          </span>
          <span className={styles.details}>
            {formatBytes(file.size)} · {dimText}
          </span>
        </div>
      </div>

      {!isBackendHealthy && (
        <div className={styles.warningBox} role="alert">
          <span className={styles.warningIcon} aria-hidden="true">
            ⚠️
          </span>
          <span className={styles.warningText}>
            Backend service appears offline (127.0.0.1:8000). Diagnosis is disabled until the
            API is reachable.
          </span>
        </div>
      )}

      <div className={styles.actionRow}>
        <button
          type="button"
          className="btn btn-primary btn-lg"
          onClick={onRunDiagnosis}
          disabled={isDiagnosing || !isBackendHealthy}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden="true"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          Run Diagnosis
        </button>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={onChangeImage}
          disabled={isDiagnosing}
        >
          Change Image
        </button>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={onRetakePhoto}
          disabled={isDiagnosing}
        >
          Retake Photo
        </button>

        <button
          type="button"
          className="btn btn-ghost"
          onClick={onCancel}
          disabled={isDiagnosing}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};
