import { StageTimeline } from "@/features/diagnose/StageTimeline";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";

describe("StageTimeline Component", () => {
  it("renders logical stages header and initial stages", () => {
    render(<StageTimeline isLoading={true} isComplete={false} />);

    expect(screen.getByText("Logical Execution Sequence")).toBeInTheDocument();
    expect(screen.getByText("1. Image Intake")).toBeInTheDocument();
    expect(screen.getByText("2. Species Classifier")).toBeInTheDocument();
    expect(screen.getByText("3. Safety Gate 1")).toBeInTheDocument();
    expect(screen.getByText("4. Crop Specialist")).toBeInTheDocument();
    expect(screen.getByText("5. Safety Gate 2")).toBeInTheDocument();
    expect(screen.getByText("6. Response Formulation")).toBeInTheDocument();
  });

  it("displays retry badge when retrying due to capacity load shedding", () => {
    render(
      <StageTimeline
        isLoading={true}
        isComplete={false}
        retryAttempt={2}
        retryMessage="Service busy. Retrying in 1.5s..."
      />,
    );

    expect(
      screen.getByText("Service busy. Retrying in 1.5s..."),
    ).toBeInTheDocument();
  });

  it("shows skipped stages and abstention notice when species is uncertain", () => {
    render(
      <StageTimeline
        isLoading={false}
        isComplete={true}
        safetyStatus="species_uncertain"
      />,
    );

    expect(
      screen.getByText(
        /Species confidence fell below the 90% threshold. Subsequent specialist inference was skipped/,
      ),
    ).toBeInTheDocument();
  });
});
