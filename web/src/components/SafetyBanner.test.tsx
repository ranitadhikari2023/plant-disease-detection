import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SafetyBanner } from './SafetyBanner';

describe('SafetyBanner component', () => {
  it('renders confident status correctly', () => {
    render(
      <SafetyBanner
        status="confident"
        warnings={[]}
        species="Potato"
        disease="Early Blight"
      />
    );

    expect(screen.getByText('Confident Diagnosis')).toBeInTheDocument();
    expect(screen.getByText('Potato — Early Blight')).toBeInTheDocument();
    expect(screen.getByText(/satisfies all safety gates/i)).toBeInTheDocument();
  });

  it('renders species_uncertain status with warnings', () => {
    render(
      <SafetyBanner
        status="species_uncertain"
        warnings={['Gate 1: Species confidence 84.2% < 90.0%']}
        species="Tomato"
        disease={null}
      />
    );

    expect(screen.getByText('Species Identification Uncertain')).toBeInTheDocument();
    expect(screen.getByText(/Safety Gate 1 Triggered/i)).toBeInTheDocument();
    expect(screen.getByText('Gate 1: Species confidence 84.2% < 90.0%')).toBeInTheDocument();
  });

  it('renders late_blight_uncertain status with high consequence warning', () => {
    render(
      <SafetyBanner
        status="late_blight_uncertain"
        warnings={['Late Blight confidence 78.4% below safety threshold 90.0%']}
        species="Potato"
        disease="Late Blight"
      />
    );

    expect(screen.getByText('High Consequence Risk')).toBeInTheDocument();
    expect(screen.getByText(/Safety Gate 2: Late Blight Uncertainty/i)).toBeInTheDocument();
    expect(screen.getByText(/DO NOT apply aggressive systemic fungicides/i)).toBeInTheDocument();
  });

  it('renders disease_uncertain status', () => {
    render(
      <SafetyBanner
        status="disease_uncertain"
        warnings={[]}
        species="Tomato"
        disease="Early Blight"
      />
    );

    expect(screen.getByText('Marginal Confidence')).toBeInTheDocument();
    expect(screen.getByText(/Diagnostic Confidence Below Threshold/i)).toBeInTheDocument();
  });
});
