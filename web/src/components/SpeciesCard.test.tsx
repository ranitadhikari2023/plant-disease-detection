import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SpeciesCard } from './SpeciesCard';

describe('SpeciesCard component', () => {
  it('renders species name, scientific name, and confidence percent', () => {
    render(
      <SpeciesCard
        species="Potato"
        confidence={0.9654}
        safetyStatus="confident"
        probabilities={{ Potato: 0.9654, Tomato: 0.0346 }}
      />
    );

    expect(screen.getByText('Potato')).toBeInTheDocument();
    expect(screen.getByText('Solanum tuberosum')).toBeInTheDocument();
    expect(screen.getByText('96.54%')).toBeInTheDocument();
    expect(screen.getByText('Gate 1 Passed (≥90%)')).toBeInTheDocument();
  });

  it('renders gate 1 triggered pill when species is uncertain', () => {
    render(
      <SpeciesCard
        species="Tomato"
        confidence={0.82}
        safetyStatus="species_uncertain"
        probabilities={{ Potato: 0.18, Tomato: 0.82 }}
      />
    );

    expect(screen.getByText('Gate 1 Triggered (<90%)')).toBeInTheDocument();
  });
});
