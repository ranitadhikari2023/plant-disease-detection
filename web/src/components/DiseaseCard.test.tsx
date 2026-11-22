import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DiseaseCard } from './DiseaseCard';

describe('DiseaseCard component', () => {
  it('renders disease diagnosis and probability distribution meters', () => {
    render(
      <DiseaseCard
        disease="Late Blight"
        confidence={0.9412}
        safetyStatus="confident"
        probabilities={{ 'Early Blight': 0.02, 'Late Blight': 0.9412, Healthy: 0.0388 }}
      />
    );

    expect(screen.getByText('Late Blight')).toBeInTheDocument();
    expect(screen.getByText('94.12%')).toBeInTheDocument();
    expect(screen.getByText('Specialist Confident')).toBeInTheDocument();
  });

  it('renders abstained message when species is uncertain', () => {
    render(
      <DiseaseCard
        disease={null}
        confidence={null}
        safetyStatus="species_uncertain"
        probabilities={{}}
      />
    );

    expect(screen.getByText('Pathology Assessment Withheld')).toBeInTheDocument();
    expect(screen.getByText('Abstained by Gate 1')).toBeInTheDocument();
    expect(screen.getByText(/botanical species classification fell below/i)).toBeInTheDocument();
  });

  it('renders Late Blight warning pill on late_blight_uncertain', () => {
    render(
      <DiseaseCard
        disease="Late Blight"
        confidence={0.84}
        safetyStatus="late_blight_uncertain"
        probabilities={{ 'Early Blight': 0.1, 'Late Blight': 0.84, Healthy: 0.06 }}
      />
    );

    expect(screen.getByText('Late Blight Warning (<90%)')).toBeInTheDocument();
  });

  it('renders marginal confidence pill on disease_uncertain', () => {
    render(
      <DiseaseCard
        disease="Early Blight"
        confidence={0.65}
        safetyStatus="disease_uncertain"
        probabilities={{ 'Early Blight': 0.65, 'Late Blight': 0.25, Healthy: 0.1 }}
      />
    );

    expect(screen.getByText('Marginal Confidence (<70%)')).toBeInTheDocument();
  });
});
