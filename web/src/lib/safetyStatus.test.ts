import { describe, it, expect } from 'vitest';
import { getSafetyStatusConfig, SAFETY_STATUS_MAP } from './safetyStatus';

describe('safetyStatus', () => {
  it('contains configurations for all 4 safety statuses', () => {
    expect(SAFETY_STATUS_MAP['confident']).toBeDefined();
    expect(SAFETY_STATUS_MAP['species_uncertain']).toBeDefined();
    expect(SAFETY_STATUS_MAP['late_blight_uncertain']).toBeDefined();
    expect(SAFETY_STATUS_MAP['disease_uncertain']).toBeDefined();
  });

  it('maps confident status correctly', () => {
    const cfg = getSafetyStatusConfig('confident');
    expect(cfg.className).toBe('confident');
    expect(cfg.icon).toBe('✓');
    expect(cfg.pillText).toBe('Confident Diagnosis');
    expect(cfg.heading('Potato', 'Early Blight')).toBe('Potato — Early Blight');
    expect(cfg.heading(null, null)).toBe('Plant — Foliar Screening');
    expect(cfg.advisoryType).toBe('default');
  });

  it('maps species_uncertain status correctly', () => {
    const cfg = getSafetyStatusConfig('species_uncertain');
    expect(cfg.className).toBe('species_uncertain');
    expect(cfg.icon).toBe('⚠️');
    expect(cfg.pillText).toBe('Species Identification Uncertain');
    expect(cfg.heading(null, null)).toContain('Safety Gate 1');
    expect(cfg.message).toContain('Botanical species identification is marginal');
    expect(cfg.advisoryType).toBe('warning');
  });

  it('maps late_blight_uncertain status correctly', () => {
    const cfg = getSafetyStatusConfig('late_blight_uncertain');
    expect(cfg.className).toBe('late_blight_uncertain');
    expect(cfg.icon).toBe('🛑');
    expect(cfg.pillText).toBe('High Consequence Risk');
    expect(cfg.advisoryType).toBe('urgent');
    expect(cfg.message).toContain('DO NOT apply aggressive systemic fungicides');
  });

  it('maps disease_uncertain status correctly', () => {
    const cfg = getSafetyStatusConfig('disease_uncertain');
    expect(cfg.className).toBe('disease_uncertain');
    expect(cfg.icon).toBe('⚡');
    expect(cfg.pillText).toBe('Marginal Confidence');
    expect(cfg.advisoryType).toBe('warning');
  });

  it('falls back safely on unknown status', () => {
    const cfg = getSafetyStatusConfig('unknown_custom_status');
    expect(cfg.status).toBe('unknown_custom_status');
    expect(cfg.icon).toBe('ℹ️');
    expect(cfg.pillText).toBe('Notice');
    expect(cfg.heading('Tomato', null)).toBe('Tomato Assessment');
  });

  it('falls back safely on null/undefined status', () => {
    const cfg = getSafetyStatusConfig(null);
    expect(cfg.status).toBe('unknown');
    expect(cfg.heading(null, null)).toBe('Specimen Assessment');
  });
});
