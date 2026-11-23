import { SafetyBanner } from "@/features/safety/SafetyBanner";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";

describe("SafetyBanner Component", () => {
  it("renders confident safety banner with verified badges", () => {
    render(
      <SafetyBanner
        status="confident"
        diagnosisTitle="Potato - Early Blight"
      />,
    );
    expect(screen.getByText("Verified Diagnosis")).toBeInTheDocument();
    expect(screen.getByText("Potato - Early Blight")).toBeInTheDocument();
  });

  it("renders late_blight_uncertain banner with assertive role and critical warning", () => {
    render(
      <SafetyBanner
        status="late_blight_uncertain"
        warnings={[
          "Potential early lesion, Late Blight confidence is below 90%.",
        ]}
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(
      screen.getByText("Late Blight Warning: Confidence Below 90%"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Potential early lesion/)).toBeInTheDocument();
  });

  it("renders species_uncertain banner with diagnostic abstained notification", () => {
    render(
      <SafetyBanner
        status="species_uncertain"
        warnings={["Species confidence below 90% threshold."]}
      />,
    );
    expect(
      screen.getByText("Crop Identification Below 90% Threshold"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Species confidence below 90%/),
    ).toBeInTheDocument();
  });
});
