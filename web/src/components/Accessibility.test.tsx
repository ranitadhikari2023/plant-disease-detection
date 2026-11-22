import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import axe from 'axe-core';
import { Dropzone } from './Dropzone';
import { PreviewPanel } from './PreviewPanel';
import { AnalyzingCard } from './AnalyzingCard';
import { SafetyBanner } from './SafetyBanner';
import { SpeciesCard } from './SpeciesCard';
import { DiseaseCard } from './DiseaseCard';
import { AdvisoryCard } from './AdvisoryCard';

describe('Accessibility audit (axe-core)', () => {
  it('Dropzone has zero critical violations', async () => {
    const { container } = render(<Dropzone onFileSelect={() => {}} />);
    const results = await axe.run(container);
    const criticals = results.violations.filter((v) => v.impact === 'critical');
    expect(criticals).toHaveLength(0);
  });

  it('PreviewPanel has zero critical violations', async () => {
    const file = new File(['fake'], 'leaf.jpg', { type: 'image/jpeg' });
    const { container } = render(
      <PreviewPanel
        file={file}
        previewUrl="data:image/jpeg;base64,mock"
        dimensions={{ width: 800, height: 600 }}
        onRunDiagnosis={() => {}}
        onChangeImage={() => {}}
        onRetakePhoto={() => {}}
        onCancel={() => {}}
      />
    );
    const results = await axe.run(container);
    const criticals = results.violations.filter((v) => v.impact === 'critical');
    expect(criticals).toHaveLength(0);
  });

  it('AnalyzingCard has zero critical violations', async () => {
    const { container } = render(<AnalyzingCard />);
    const results = await axe.run(container);
    const criticals = results.violations.filter((v) => v.impact === 'critical');
    expect(criticals).toHaveLength(0);
  });

  it('SafetyBanner across all safety statuses has zero critical violations', async () => {
    for (const st of ['confident', 'species_uncertain', 'late_blight_uncertain', 'disease_uncertain']) {
      const { container } = render(
        <SafetyBanner
          status={st}
          warnings={['Sample safety warning']}
          species="Potato"
          disease="Late Blight"
        />
      );
      const results = await axe.run(container);
      const criticals = results.violations.filter((v) => v.impact === 'critical');
      expect(criticals).toHaveLength(0);
    }
  });

  it('SpeciesCard and DiseaseCard have zero critical violations', async () => {
    const { container: speciesContainer } = render(
      <SpeciesCard
        species="Potato"
        confidence={0.96}
        safetyStatus="confident"
        probabilities={{ Potato: 0.96, Tomato: 0.04 }}
      />
    );
    const sResults = await axe.run(speciesContainer);
    expect(sResults.violations.filter((v) => v.impact === 'critical')).toHaveLength(0);

    const { container: diseaseContainer } = render(
      <DiseaseCard
        disease="Early Blight"
        confidence={0.92}
        safetyStatus="confident"
        probabilities={{ 'Early Blight': 0.92, 'Late Blight': 0.05, Healthy: 0.03 }}
      />
    );
    const dResults = await axe.run(diseaseContainer);
    expect(dResults.violations.filter((v) => v.impact === 'critical')).toHaveLength(0);
  });

  it('AdvisoryCard has zero critical violations', async () => {
    const { container } = render(
      <AdvisoryCard
        advisory="Follow local extension guidelines."
        safetyStatus="confident"
      />
    );
    const results = await axe.run(container);
    const criticals = results.violations.filter((v) => v.impact === 'critical');
    expect(criticals).toHaveLength(0);
  });
});
