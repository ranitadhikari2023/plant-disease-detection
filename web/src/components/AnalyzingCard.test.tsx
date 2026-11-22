import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnalyzingCard } from './AnalyzingCard';

describe('AnalyzingCard component', () => {
  it('renders indeterminate progress state and accessible status', () => {
    render(<AnalyzingCard />);

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText(/Preparing Foliar Specimen/i)).toBeInTheDocument();
    expect(screen.getByText(/Dual-gate safety validation active/i)).toBeInTheDocument();
  });
});
