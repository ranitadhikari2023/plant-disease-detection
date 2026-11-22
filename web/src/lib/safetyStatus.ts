/**
 * web/src/lib/safetyStatus.ts
 * Centralized mapping for the four hierarchical safety statuses
 */

export interface SafetyStatusConfig {
  status: string;
  className: string;
  icon: string;
  pillText: string;
  heading: (species: string | null, disease: string | null) => string;
  message: string;
  advisoryType: 'default' | 'urgent' | 'warning';
}

export const SAFETY_STATUS_MAP: Record<string, SafetyStatusConfig> = {
  confident: {
    status: 'confident',
    className: 'confident',
    icon: '✓',
    pillText: 'Confident Diagnosis',
    heading: (species, disease) => `${species || 'Plant'} — ${disease || 'Foliar Screening'}`,
    message: 'Prediction satisfies all safety gates. Diagnostic confidence is optimal.',
    advisoryType: 'default',
  },
  species_uncertain: {
    status: 'species_uncertain',
    className: 'species_uncertain',
    icon: '⚠️',
    pillText: 'Species Identification Uncertain',
    heading: () => 'Safety Gate 1 Triggered (<90% Species Confidence)',
    message:
      'Botanical species identification is marginal. Disease classification has been abstained to avoid crop misdiagnosis. Do NOT apply chemicals to unverified plant species.',
    advisoryType: 'warning',
  },
  late_blight_uncertain: {
    status: 'late_blight_uncertain',
    className: 'late_blight_uncertain',
    icon: '🛑',
    pillText: 'High Consequence Risk',
    heading: () => 'Safety Gate 2: Late Blight Uncertainty (<90%)',
    message:
      'Potential foliar lesions detected, but Late Blight confidence is below the safety threshold. DO NOT apply aggressive systemic fungicides at this stage. Isolate specimen and re-inspect within 24 to 48 hours.',
    advisoryType: 'urgent',
  },
  disease_uncertain: {
    status: 'disease_uncertain',
    className: 'disease_uncertain',
    icon: '⚡',
    pillText: 'Marginal Confidence',
    heading: () => 'Diagnostic Confidence Below Threshold (<70%)',
    message:
      'Visual foliar symptoms are ambiguous. Re-photograph the leaf in diffuse daylight and monitor closely before initiating chemical treatment.',
    advisoryType: 'warning',
  },
};

export function getSafetyStatusConfig(status: string | null | undefined): SafetyStatusConfig {
  if (status && SAFETY_STATUS_MAP[status]) {
    return SAFETY_STATUS_MAP[status];
  }
  return {
    status: status || 'unknown',
    className: 'species_uncertain',
    icon: 'ℹ️',
    pillText: 'Notice',
    heading: (species) => `${species || 'Specimen'} Assessment`,
    message: `Result evaluated with safety status: ${status || 'unspecified'}.`,
    advisoryType: 'warning',
  };
}
