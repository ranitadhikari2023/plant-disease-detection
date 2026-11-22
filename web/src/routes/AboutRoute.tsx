/**
 * web/src/routes/AboutRoute.tsx
 * Comprehensive architectural and agronomic system documentation
 */

import React from 'react';
import styles from './AboutRoute.module.css';

export const AboutRoute: React.FC = () => {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>System Architecture & Methodology</h1>
        <p className={styles.subtitle}>
          Technical specifications for LeafScan V2 hierarchical foliar disease detection.
        </p>
      </header>

      <div className={styles.sectionsGrid}>
        {/* Section 1: Overview */}
        <section className={styles.sectionCard}>
          <div className={styles.cardHeader}>
            <span className={styles.badge}>Tiered Inference</span>
            <h2 className={styles.heading}>Two-Tier Hierarchical Classification</h2>
          </div>
          <p className={styles.paragraph}>
            LeafScan V2 resolves critical agronomic misclassification by decoupling botanical
            species identification from disease symptom diagnosis. Rather than executing flat
            classification over unverified foliar classes, the system evaluates incoming specimens
            through a sequential hierarchical decision tree.
          </p>
          <div className={styles.featureList}>
            <div className={styles.featureItem}>
              <strong>Tier 1 — Botanist Expert:</strong> Lightweight MobileNetV2 backbone trained to
              identify plant species (Potato vs Tomato) with high-confidence morphology extraction.
            </div>
            <div className={styles.featureItem}>
              <strong>Tier 2 — Crop Pathology Specialists:</strong> Dedicated CBAM-augmented models
              specialized exclusively in crop-specific necrotic lesions (Early Blight, Late Blight,
              Healthy).
            </div>
          </div>
        </section>

        {/* Section 2: Dual Confidence Gates */}
        <section className={styles.sectionCard}>
          <div className={styles.cardHeader}>
            <span className={styles.badge}>Diagnostic Safety</span>
            <h2 className={styles.heading}>Dual-Gate Safety & Diagnostic Abstention</h2>
          </div>
          <p className={styles.paragraph}>
            To protect growers from dangerous chemical misapplications, LeafScan V2 implements
            strict mathematical safety boundaries:
          </p>
          <div className={styles.featureList}>
            <div className={styles.featureItem}>
              <strong>Safety Gate 1 (Species Boundary):</strong> Requires ≥90% softmax confidence at
              Tier 1. When botanical certainty falls below 90%, the system halts inference and issues
              a diagnostic abstention (<code>species_uncertain</code>), refusing to predict disease on
              an ambiguous plant.
            </div>
            <div className={styles.featureItem}>
              <strong>Safety Gate 2 (Pathology Escalation):</strong> Late Blight (<em>Phytophthora
              infestans</em>) is a high-consequence agricultural threat. Predictions are subjected to
              a 90% confidence threshold, escalating uncertain detections as{' '}
              <code>late_blight_uncertain</code> and discouraging premature aggressive fungicides.
            </div>
          </div>
        </section>

        {/* Section 3: CBAM Attention */}
        <section className={styles.sectionCard}>
          <div className={styles.cardHeader}>
            <span className={styles.badge}>Deep Learning</span>
            <h2 className={styles.heading}>CBAM Convolutional Block Attention</h2>
          </div>
          <p className={styles.paragraph}>
            Crop specialist submodules integrate dual Channel and Spatial Attention mechanisms to
            focus on localized foliar lesions while suppressing background soil, fingers, and
            irrelevant leaf veins:
          </p>
          <div className={styles.featureList}>
            <div className={styles.featureItem}>
              <strong>Channel Attention Module:</strong> Explores inter-channel relationships of foliar
              feature maps via parallel Average and Max pooling, emphasizing diagnostic color and
              frequency features.
            </div>
            <div className={styles.featureItem}>
              <strong>Spatial Attention Module:</strong> Employs inter-spatial feature relationships
              to localize concentric ring lesions and chlorotic halos characteristic of Alternaria
              and Phytophthora pathogens.
            </div>
          </div>
        </section>

        {/* Section 4: Specifications */}
        <section className={styles.sectionCard}>
          <div className={styles.cardHeader}>
            <span className={styles.badge}>Technical Specifications</span>
            <h2 className={styles.heading}>Model & System Telemetry</h2>
          </div>
          <div className={styles.specGrid}>
            <div className={styles.specItem}>
              <span className={styles.specKey}>Input Dimensions</span>
              <span className={styles.specVal}>224 × 224 × 3 RGB</span>
            </div>
            <div className={styles.specItem}>
              <span className={styles.specKey}>Inference Framework</span>
              <span className={styles.specVal}>ONNX Runtime / TensorFlow</span>
            </div>
            <div className={styles.specItem}>
              <span className={styles.specKey}>Target Solanaceous Species</span>
              <span className={styles.specVal}>Potato, Tomato</span>
            </div>
            <div className={styles.specItem}>
              <span className={styles.specKey}>Diagnostic Categories</span>
              <span className={styles.specVal}>Early Blight, Late Blight, Healthy</span>
            </div>
            <div className={styles.specItem}>
              <span className={styles.specKey}>Mean Inference Latency</span>
              <span className={styles.specVal}>~180–350 ms</span>
            </div>
            <div className={styles.specItem}>
              <span className={styles.specKey}>Serving Framework</span>
              <span className={styles.specVal}>FastAPI / Uvicorn</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
