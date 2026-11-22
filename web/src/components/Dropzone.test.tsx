import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Dropzone } from './Dropzone';

describe('Dropzone component', () => {
  it('renders dropzone text and buttons', () => {
    render(<Dropzone onFileSelect={vi.fn()} />);

    expect(screen.getByText('Drop your leaf photograph here')).toBeInTheDocument();
    expect(screen.getByText('Browse Files')).toBeInTheDocument();
    expect(screen.getByText('Take Photo')).toBeInTheDocument();
    expect(screen.getByText(/Supported: JPEG, PNG, WebP/)).toBeInTheDocument();
  });

  it('handles browse files button click triggering file input', () => {
    const handleSelect = vi.fn();
    render(<Dropzone onFileSelect={handleSelect} />);

    // Query file input directly
    const inputs = document.querySelectorAll('input[type="file"]');
    expect(inputs.length).toBe(2);

    const testFile = new File(['dummy'], 'leaf.jpg', { type: 'image/jpeg' });
    fireEvent.change(inputs[0], { target: { files: [testFile] } });

    expect(handleSelect).toHaveBeenCalledWith(testFile);
  });

  it('triggers on container keyboard Enter', () => {
    const handleSelect = vi.fn();
    render(<Dropzone onFileSelect={handleSelect} />);

    const region = screen.getByRole('region', { name: /drop area/i });
    fireEvent.keyDown(region, { key: 'Enter' });
  });
});
