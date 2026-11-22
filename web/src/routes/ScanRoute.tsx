/**
 * web/src/routes/ScanRoute.tsx
 * Main foliar scan workspace
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Dropzone } from '../components/Dropzone';
import { PreviewPanel } from '../components/PreviewPanel';
import { AnalyzingCard } from '../components/AnalyzingCard';
import { RecentScans } from '../components/RecentScans';
import { validateFile } from '../lib/validateFile';
import { useThumbnail } from '../hooks/useThumbnail';
import { useLocalHistory } from '../hooks/useLocalHistory';
import { diagnoseImage, DiagnosisError } from '../api/diagnose';
import styles from './ScanRoute.module.css';

interface ScanRouteProps {
  isBackendHealthy: boolean;
  announce: (message: string) => void;
}

export const ScanRoute: React.FC<ScanRouteProps> = ({ isBackendHealthy, announce }) => {
  const navigate = useNavigate();
  const { history, saveRecord } = useLocalHistory();
  const { generateThumbnail } = useThumbnail();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number | null; height: number | null }>({
    width: null,
    height: null,
  });
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);

  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [alertInfo, setAlertInfo] = useState<{ title: string; message: string } | null>(null);

  const [openAccordion, setOpenAccordion] = useState<number | null>(null);

  // Clean up object URL on unmount or file change
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleSelectFile = (file: File) => {
    setAlertInfo(null);

    const validation = validateFile(file);
    if (!validation.valid) {
      setAlertInfo({
        title: 'Invalid Image File',
        message: validation.error || 'Please provide a valid foliar image.',
      });
      announce(`Invalid file: ${validation.error}`);
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const objUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(objUrl);
    setThumbnailUrl(null);
    setDimensions({ width: null, height: null });

    const img = new Image();
    img.onload = () => {
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      const thumb = generateThumbnail(img);
      setThumbnailUrl(thumb);
    };
    img.src = objUrl;

    announce(`Selected leaf photo ${file.name} ready for diagnosis.`);
  };

  const handleReset = () => {
    setAlertInfo(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setThumbnailUrl(null);
    setDimensions({ width: null, height: null });
    announce('Upload area reset. Ready for new leaf image.');
  };

  const handleExecuteDiagnosis = async () => {
    if (!selectedFile || isDiagnosing) return;

    setAlertInfo(null);
    setIsDiagnosing(true);
    announce('Diagnostic inference started. Running MobileNetV2 and CBAM screening.');

    try {
      const result = await diagnoseImage(selectedFile);
      setIsDiagnosing(false);

      const record = saveRecord(selectedFile.name, result, thumbnailUrl);
      announce('Diagnostic pathology report loaded.');

      if (record) {
        navigate(`/result/${record.id}`);
      } else {
        // Fallback with timestamp id
        navigate(`/result/${Date.now()}`, { state: { resultData: result } });
      }
    } catch (err: unknown) {
      setIsDiagnosing(false);
      const msg =
        err instanceof DiagnosisError
          ? err.message
          : err instanceof Error
          ? err.message
          : 'Server error occurred during classification. Please retry.';
      setAlertInfo({
        title: 'Diagnosis Inference Failed',
        message: msg,
      });
      announce(`Diagnosis failed: ${msg}`);
    }
  };

  const toggleAccordion = (index: number) => {
    setOpenAccordion((prev) => (prev === index ? null : index));
  };

  return (
    <div className={styles.workspace}>
      <header className={styles.pageHeader}>
        <div className={styles.heroText}>
          <h1 className={styles.title}>LeafScan V2 Foliar Screening</h1>
          <p className={styles.subtitle}>
            Two-tier MobileNetV2 and CBAM neural network classification with dual-gate safety
            validation for Solanaceous crops.
          </p>
        </div>
      </header>

      {alertInfo && (
        <div className={styles.alertBox} role="alert">
          <div className={styles.alertHeader}>
            <span className={styles.alertIcon} aria-hidden="true">
              ⚠️
            </span>
            <strong className={styles.alertTitle}>{alertInfo.title}</strong>
          </div>
          <p className={styles.alertMessage}>{alertInfo.message}</p>
          <button
            type="button"
            className={styles.alertDismiss}
            onClick={() => setAlertInfo(null)}
            aria-label="Dismiss alert"
          >
            ✕
          </button>
        </div>
      )}

      <div className={styles.layoutGrid}>
        <div className={styles.mainCol}>
          {isDiagnosing ? (
            <AnalyzingCard />
          ) : selectedFile && previewUrl ? (
            <PreviewPanel
              file={selectedFile}
              previewUrl={previewUrl}
              dimensions={dimensions}
              onRunDiagnosis={handleExecuteDiagnosis}
              onChangeImage={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp';
                input.onchange = (e) => {
                  const f = (e.target as HTMLInputElement).files?.[0];
                  if (f) handleSelectFile(f);
                };
                input.click();
              }}
              onRetakePhoto={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/*';
                input.capture = 'environment';
                input.onchange = (e) => {
                  const f = (e.target as HTMLInputElement).files?.[0];
                  if (f) handleSelectFile(f);
                };
                input.click();
              }}
              onCancel={handleReset}
              isDiagnosing={isDiagnosing}
              isBackendHealthy={isBackendHealthy}
            />
          ) : (
            <Dropzone onFileSelect={handleSelectFile} />
          )}

          <section className={styles.recentSection}>
            <RecentScans scans={history} />
          </section>
        </div>

        <aside className={styles.sideCol}>
          <div className={styles.protocolCard}>
            <h3 className={styles.protocolTitle}>Diagnostic Imaging Protocol</h3>
            <p className={styles.protocolDesc}>
              Follow agronomic capture guidelines to maximize model accuracy and prevent safety gate
              abstentions.
            </p>

            <div className={styles.accordionGroup}>
              <div className={styles.accordionItem}>
                <button
                  type="button"
                  className={styles.accordionHeader}
                  onClick={() => toggleAccordion(1)}
                  aria-expanded={openAccordion === 1}
                >
                  <span>1. Natural Diffuse Illumination</span>
                  <span>{openAccordion === 1 ? '−' : '+'}</span>
                </button>
                {openAccordion === 1 && (
                  <div className={styles.accordionBody}>
                    Photograph foliage in even daytime shade or overcast sky. Avoid harsh direct
                    sunlight glare or heavy camera flash that washes out foliar lesion boundaries.
                  </div>
                )}
              </div>

              <div className={styles.accordionItem}>
                <button
                  type="button"
                  className={styles.accordionHeader}
                  onClick={() => toggleAccordion(2)}
                  aria-expanded={openAccordion === 2}
                >
                  <span>2. Single Leaf Framing</span>
                  <span>{openAccordion === 2 ? '−' : '+'}</span>
                </button>
                {openAccordion === 2 && (
                  <div className={styles.accordionBody}>
                    Center one symptomatic leaf against neutral soil or hand background. Ensure the
                    leaf fills at least 60% of the frame without obstructing fingers or background weeds.
                  </div>
                )}
              </div>

              <div className={styles.accordionItem}>
                <button
                  type="button"
                  className={styles.accordionHeader}
                  onClick={() => toggleAccordion(3)}
                  aria-expanded={openAccordion === 3}
                >
                  <span>3. Perpendicular Focal Plane</span>
                  <span>{openAccordion === 3 ? '−' : '+'}</span>
                </button>
                {openAccordion === 3 && (
                  <div className={styles.accordionBody}>
                    Hold camera flat parallel to foliar surface. Keep lesions in sharp optical focus.
                    MobileNetV2 and CBAM rely on concentric ring texture to distinguish Early vs Late
                    Blight.
                  </div>
                )}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};
