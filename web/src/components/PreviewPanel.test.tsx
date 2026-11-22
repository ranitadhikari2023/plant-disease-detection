import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PreviewPanel } from './PreviewPanel';

describe('PreviewPanel component', () => {
  const mockFile = new File(['test'], 'specimen.jpg', { type: 'image/jpeg' });

  it('renders preview image, filename, dimensions, and action buttons', () => {
    const onRun = vi.fn();
    const onChange = vi.fn();
    const onRetake = vi.fn();
    const onCancel = vi.fn();

    render(
      <PreviewPanel
        file={mockFile}
        previewUrl="blob:http://localhost/fake-blob"
        dimensions={{ width: 800, height: 600 }}
        onRunDiagnosis={onRun}
        onChangeImage={onChange}
        onRetakePhoto={onRetake}
        onCancel={onCancel}
        isDiagnosing={false}
        isBackendHealthy={true}
      />
    );

    expect(screen.getByText('specimen.jpg')).toBeInTheDocument();
    expect(screen.getByText(/800 × 600 px/)).toBeInTheDocument();

    const runBtn = screen.getByRole('button', { name: /run diagnosis/i });
    expect(runBtn).toBeEnabled();

    fireEvent.click(runBtn);
    expect(onRun).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: /change image/i }));
    expect(onChange).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: /retake photo/i }));
    expect(onRetake).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    expect(onCancel).toHaveBeenCalled();
  });

  it('disables Run Diagnosis and displays warning when backend is offline', () => {
    render(
      <PreviewPanel
        file={mockFile}
        previewUrl="blob:http://localhost/fake-blob"
        dimensions={{ width: 800, height: 600 }}
        onRunDiagnosis={vi.fn()}
        onChangeImage={vi.fn()}
        onRetakePhoto={vi.fn()}
        onCancel={vi.fn()}
        isDiagnosing={false}
        isBackendHealthy={false}
      />
    );

    const runBtn = screen.getByRole('button', { name: /run diagnosis/i });
    expect(runBtn).toBeDisabled();
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/Backend service appears offline/i)).toBeInTheDocument();
  });
});
