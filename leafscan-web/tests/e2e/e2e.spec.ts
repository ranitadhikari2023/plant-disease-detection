import { test, expect } from "@playwright/test";

test.describe("LeafScan V3 End-to-End Suite", () => {
  test("marketing homepage renders hero and navigation to diagnose", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/LeafScan V3/i);
    await expect(page.locator("h1")).toContainText("Scientific Foliar Screening for Solanaceous Crops");

    const launchBtn = page.getByRole("link", { name: /Launch Field Diagnostic/i });
    await expect(launchBtn).toBeVisible();
    await launchBtn.click();
    await expect(page).toHaveURL(/\/diagnose/);
  });

  test("diagnose page displays capture card and field photography guidelines", async ({ page }) => {
    await page.goto("/diagnose");
    await expect(page.getByRole("heading", { name: "Field Foliar Screening" })).toBeVisible();
    await expect(page.getByText("Submit Leaf Specimen")).toBeVisible();
    await expect(page.getByText("Field Photography Best Practices")).toBeVisible();
    await expect(page.getByRole("button", { name: /Capture with Camera/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Browse Device/i })).toBeVisible();
  });

  test("model transparency page renders verified cryptographic manifest", async ({ page }) => {
    await page.goto("/model");
    await expect(page.getByRole("heading", { name: "Model Transparency & Verification" })).toBeVisible();
    await expect(page.getByText("Cryptographically Verified")).toBeVisible();
    await expect(page.getByText("Deterministic Safety Thresholds")).toBeVisible();
    await expect(page.getByText("Gate 1: Species Confidence")).toBeVisible();
    await expect(page.getByText("Gate 2: Late Blight Escalation")).toBeVisible();
  });

  test("history ledger displays empty state on fresh profile", async ({ page }) => {
    await page.goto("/history");
    await expect(page.getByRole("heading", { name: "Field Diagnosis Ledger" })).toBeVisible();
    await expect(page.getByText(/No foliar scans recorded in local history yet/i)).toBeVisible();
  });

  test("feedback page renders input fields and agreements", async ({ page }) => {
    await page.goto("/feedback");
    await expect(page.getByRole("heading", { name: "Field Validation Feedback" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Submit Diagnostic Feedback/i })).toBeVisible();
  });
});
