import { test, expect } from '../../web/node_modules/@playwright/test';

test.describe('LeafScan V2 Modernized Frontend E2E Suite', () => {
  test.beforeEach(async ({ page }) => {
    // Mock /health endpoint by default to guarantee healthy state
    await page.route('**/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'healthy',
          model_version: 'v2',
          architecture: 'MobileNetV2 + CBAM Attention',
          models_loaded: true,
        }),
      });
    });

    // Mock /openapi.json for system route tests
    await page.route('**/openapi.json', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          openapi: '3.1.0',
          info: { title: 'LeafScan V2 API', version: '2.0.0' },
          paths: {
            '/health': {
              get: { summary: 'Lightweight System Health Check', description: 'Reports operational status.' },
            },
            '/v2/diagnose': {
              post: { summary: 'Execute Hierarchical Foliar Diagnosis', description: 'Two-tier foliar screening.' },
            },
            '/': {
              get: { summary: 'Serve LeafScan SPA', description: 'Serves frontend application.' },
            },
          },
        }),
      });
    });
  });

  // 1. Application loads successfully
  test('1. Application loads successfully with title, branding, and navigation', async ({ page }) => {
    await page.goto('/#/scan');
    await expect(page).toHaveTitle(/LeafScan V2/i);
    await expect(page.getByRole('link', { name: /LeafScan V2/i })).toBeVisible();
    // Verify "API Connected" badge is completely removed from the header
    await expect(page.locator('text=API Connected')).toHaveCount(0);
    await expect(page.locator('nav a', { hasText: 'Scan' })).toBeVisible();
    await expect(page.locator('nav a', { hasText: 'History' })).toBeVisible();
    await expect(page.locator('nav a', { hasText: 'About' })).toBeVisible();
    await expect(page.locator('nav a', { hasText: 'System' })).toBeVisible();
  });

  // 2. Image selection and dropzone interaction
  test('2. Image file browsing and selection works', async ({ page }) => {
    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'healthy_potato.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('fake-image-bytes'),
    });

    await expect(page.getByText('healthy_potato.jpg').first()).toBeVisible();
    await expect(page.locator('button:has-text("Run Diagnosis")')).toBeVisible();
  });

  // 3. Unsupported file is rejected
  test('3. Unsupported file format is rejected with in-page alert', async ({ page }) => {
    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'notes.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('fake-pdf-content'),
    });

    await expect(page.locator('text=Invalid Image File')).toBeVisible();
    await expect(page.locator('text=Unsupported image format')).toBeVisible();
  });

  // 4. Oversized file is rejected
  test('4. Oversized file (>10MB) is rejected', async ({ page }) => {
    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    const largeBuffer = Buffer.alloc(11 * 1024 * 1024);
    await fileChooser.setFiles({
      name: 'huge_leaf.jpg',
      mimeType: 'image/jpeg',
      buffer: largeBuffer,
    });

    await expect(page.locator('text=Invalid Image File')).toBeVisible();
    await expect(page.locator('text=exceeds maximum allowed limit of 10 MB')).toBeVisible();
  });

  // 5. Image preview is displayed
  test('5. Image preview panel renders filename and action buttons', async ({ page }) => {
    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'specimen_test.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('fake-image-data'),
    });

    await expect(page.getByText('specimen_test.jpg').first()).toBeVisible();
    await expect(page.locator('button:has-text("Run Diagnosis")')).toBeVisible();
    await expect(page.locator('button:has-text("Change Image")')).toBeVisible();
    await expect(page.locator('button:has-text("Retake Photo")')).toBeVisible();
    await expect(page.locator('button:has-text("Cancel")')).toBeVisible();
  });

  // 6. Change and cancel actions work
  test('6. Cancel action restores the initial idle Dropzone state', async ({ page }) => {
    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'test.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('fake-data'),
    });

    await expect(page.getByText('test.jpg').first()).toBeVisible();
    await page.click('button:has-text("Cancel")');
    await expect(page.locator('text=Drop your leaf photograph here')).toBeVisible();
  });

  // 7. Full diagnosis workflow completes using a controlled test response
  test('7. Full diagnosis workflow completes and renders result report', async ({ page }) => {
    await page.route('**/v2/diagnose', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          model_version: 'v2',
          plant_species: 'Potato',
          species_confidence: 0.985,
          disease: 'Early Blight',
          disease_confidence: 0.942,
          diagnosis: 'Potato - Early Blight',
          safety_status: 'confident',
          safety_warnings: [],
          advisory: 'Prune lower diseased foliage and ensure adequate spacing.',
          species_probabilities: { Potato: 0.985, Tomato: 0.015 },
          probabilities: { 'Early Blight': 0.942, 'Late Blight': 0.038, Healthy: 0.02 },
          metadata: { execution_time_ms: 215.4, architecture: 'MobileNetV2 + CBAM Attention' },
        }),
      });
    });

    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'potato_specimen.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('fake-image-bytes'),
    });

    await page.click('button:has-text("Run Diagnosis")');

    await expect(page.locator('h1', { hasText: 'Foliar Pathology Report' })).toBeVisible();
    await expect(page.locator('text=Confident Diagnosis').first()).toBeVisible();
    await expect(page.locator('h2', { hasText: 'Potato — Early Blight' })).toBeVisible();
    await expect(page.getByText('Solanum tuberosum', { exact: true })).toBeVisible();
    await expect(page.locator('text=98.50%')).toBeVisible();
    await expect(page.locator('text=94.20%')).toBeVisible();
  });

  // 8. All four safety states render correctly
  test('8. All 4 safety statuses render appropriate UI banners', async ({ page }) => {
    const statuses = [
      {
        status: 'confident',
        expectedPill: 'Confident Diagnosis',
        species: 'Potato',
        disease: 'Early Blight',
      },
      {
        status: 'species_uncertain',
        expectedPill: 'Species Identification Uncertain',
        species: 'Tomato',
        disease: null,
      },
      {
        status: 'late_blight_uncertain',
        expectedPill: 'High Consequence Risk',
        species: 'Potato',
        disease: 'Late Blight',
      },
      {
        status: 'disease_uncertain',
        expectedPill: 'Marginal Confidence',
        species: 'Tomato',
        disease: 'Early Blight',
      },
    ];

    for (const item of statuses) {
      await page.route('**/v2/diagnose', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            model_version: 'v2',
            plant_species: item.species,
            species_confidence: item.status === 'species_uncertain' ? 0.75 : 0.95,
            disease: item.disease,
            disease_confidence: item.disease ? 0.85 : null,
            diagnosis: item.disease ? `${item.species} - ${item.disease}` : `${item.species} (Abstained)`,
            safety_status: item.status,
            safety_warnings: [],
            advisory: 'Advisory notice text.',
            species_probabilities: { Potato: 0.5, Tomato: 0.5 },
            probabilities: item.disease ? { 'Early Blight': 0.85, 'Late Blight': 0.1, Healthy: 0.05 } : {},
            metadata: { execution_time_ms: 180, architecture: 'MobileNetV2 + CBAM' },
          }),
        });
      });

      await page.goto('/#/scan');
      const fileChooserPromise = page.waitForEvent('filechooser');
      await page.click('button:has-text("Browse Files")');
      const fileChooser = await fileChooserPromise;
      await fileChooser.setFiles({
        name: `test_${item.status}.jpg`,
        mimeType: 'image/jpeg',
        buffer: Buffer.from('test-bytes'),
      });
      await page.click('button:has-text("Run Diagnosis")');

      await expect(page.locator(`text=${item.expectedPill}`).first()).toBeVisible();
    }
  });

  // 9 & 10. Diagnosis history is saved and survives reload
  test('9 & 10. Diagnosis history is saved and survives page reload', async ({ page }) => {
    await page.route('**/v2/diagnose', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          model_version: 'v2',
          plant_species: 'Potato',
          species_confidence: 0.99,
          disease: 'Healthy',
          disease_confidence: 0.97,
          diagnosis: 'Potato - Healthy',
          safety_status: 'confident',
          safety_warnings: [],
          advisory: 'Crop appears healthy.',
          species_probabilities: { Potato: 0.99, Tomato: 0.01 },
          probabilities: { 'Early Blight': 0.01, 'Late Blight': 0.02, Healthy: 0.97 },
          metadata: { execution_time_ms: 190.0, architecture: 'MobileNetV2 + CBAM' },
        }),
      });
    });

    await page.goto('/#/scan');
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'healthy_leaf_history.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('leaf-data'),
    });
    await page.click('button:has-text("Run Diagnosis")');

    await expect(page.locator('h1', { hasText: 'Foliar Pathology Report' })).toBeVisible();

    // Navigate to History
    await page.click('nav a:has-text("History")');
    await expect(page.locator('text=Potato - Healthy').first()).toBeVisible();
    await expect(page.locator('text=1 scan recorded').first()).toBeVisible();

    // Reload page
    await page.reload();
    await expect(page.locator('text=Potato - Healthy').first()).toBeVisible();
    await expect(page.locator('text=1 scan recorded').first()).toBeVisible();
  });

  // 11. Historical report opens correctly
  test('11. Historical report reopens from history view', async ({ page }) => {
    // Populate localStorage directly
    await page.goto('/#/history');
    await page.evaluate(() => {
      const entry = {
        id: 'hist_test_1',
        timestamp: new Date().toISOString(),
        filename: 'persisted_leaf.jpg',
        species: 'Tomato',
        disease: 'Early Blight',
        confidence: 0.93,
        safetyStatus: 'confident',
        diagnosis: 'Tomato - Early Blight',
        thumbnail: null,
        resultData: {
          success: true,
          model_version: 'v2',
          plant_species: 'Tomato',
          species_confidence: 0.95,
          disease: 'Early Blight',
          disease_confidence: 0.93,
          diagnosis: 'Tomato - Early Blight',
          safety_status: 'confident',
          safety_warnings: [],
          advisory: 'Sample advisory for tomato.',
          species_probabilities: { Potato: 0.05, Tomato: 0.95 },
          probabilities: { 'Early Blight': 0.93, 'Late Blight': 0.04, Healthy: 0.03 },
          metadata: { execution_time_ms: 210, architecture: 'MobileNetV2 + CBAM' },
        },
      };
      localStorage.setItem('leafscan_v2_history_v1', JSON.stringify([entry]));
    });

    await page.reload();
    await expect(page.locator('text=Tomato - Early Blight').first()).toBeVisible();
    await page.click('button:has-text("View Report")');

    await expect(page.locator('h1', { hasText: 'Foliar Pathology Report' })).toBeVisible();
    await expect(page.locator('h2', { hasText: 'Tomato — Early Blight' })).toBeVisible();
    await expect(page.getByText('Solanum lycopersicum', { exact: true })).toBeVisible();
  });

  // 12. Browser hash navigation works
  test('12. Browser hash navigation works for all main routes', async ({ page }) => {
    await page.goto('/#/scan');
    await expect(page.locator('h1', { hasText: 'LeafScan V2 Foliar Screening' })).toBeVisible();

    await page.click('nav a:has-text("History")');
    await expect(page).toHaveURL(/#\/history/);
    await expect(page.locator('h1', { hasText: 'Diagnostic History Log' })).toBeVisible();

    await page.click('nav a:has-text("About")');
    await expect(page).toHaveURL(/#\/about/);
    await expect(page.locator('h1', { hasText: 'System Architecture & Methodology' })).toBeVisible();

    await page.click('nav a:has-text("System")');
    await expect(page).toHaveURL(/#\/system/);
    await expect(page.locator('h1', { hasText: 'Backend System & API Architecture' })).toBeVisible();

    // Browser back button
    await page.goBack();
    await expect(page).toHaveURL(/#\/about/);
  });

  // 13. Backend unavailable state is handled
  test('13. Backend unavailable state disables diagnosis and displays notice', async ({ page }) => {
    await page.route('**/health', async (route) => {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'unavailable', error: 'Service unavailable' }),
      });
    });

    await page.goto('/#/scan');
    // Verify status indicator badge is not present in header
    await expect(page.locator('text=API Connected')).toHaveCount(0);
    await expect(page.locator('text=API Offline')).toHaveCount(0);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('button:has-text("Browse Files")');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'test_offline.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('fake'),
    });

    const runBtn = page.locator('button:has-text("Run Diagnosis")');
    await expect(runBtn).toBeDisabled();
    await expect(page.locator('text=Backend service appears offline')).toBeVisible();
  });

  // 14. System route displays backend API inventory
  test('14. System route displays OpenAPI inventory and documentation links', async ({ page }) => {
    await page.goto('/#/system');
    await expect(page.locator('h1', { hasText: 'Backend System & API Architecture' })).toBeVisible();
    await expect(page.locator('text=FastAPI Service Health')).toBeVisible();
    await expect(page.locator('text=Frontend Connection Configuration')).toBeVisible();
    await expect(page.locator('text=Discovered API Endpoints')).toBeVisible();

    // Check discovered routes
    await expect(page.locator('code:has-text("/health")')).toBeVisible();
    await expect(page.locator('code:has-text("/v2/diagnose")')).toBeVisible();

    // Check documentation links
    await expect(page.locator('a:has-text("Swagger UI")')).toBeVisible();
    await expect(page.locator('a:has-text("ReDoc UI")')).toBeVisible();
  });
});
