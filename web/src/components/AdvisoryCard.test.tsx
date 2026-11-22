import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AdvisoryCard } from './AdvisoryCard';

describe('AdvisoryCard component', () => {
  it('renders actionable advisory text and decision support notice', () => {
    render(
      <AdvisoryCard
        advisory="Maintain regular scouting and apply copper-based preventive spray."
        safetyStatus="confident"
      />
    );

    expect(screen.getByText(/Actionable Agronomic Advisory/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Maintain regular scouting and apply copper-based preventive spray/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Agricultural Protocol Notice:/i)).toBeInTheDocument();
  });
});
