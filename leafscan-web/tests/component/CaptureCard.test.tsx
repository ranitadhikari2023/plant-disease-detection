import { CaptureCard } from "@/features/diagnose/CaptureCard";
import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

describe("CaptureCard Component", () => {
  it("renders upload buttons and field guidelines in initial state", () => {
    render(
      <CaptureCard
        onFileSelect={vi.fn()}
        previewUrl={null}
        onClear={vi.fn()}
        isLoading={false}
        onDiagnose={vi.fn()}
      />,
    );

    expect(screen.getByText("Submit Leaf Specimen")).toBeInTheDocument();
    expect(screen.getByText("Capture with Camera")).toBeInTheDocument();
    expect(screen.getByText("Browse Device")).toBeInTheDocument();
    expect(
      screen.getByText("Field Photography Best Practices"),
    ).toBeInTheDocument();
  });

  it("renders preview image and execute button when file is selected", () => {
    const onDiagnose = vi.fn();
    render(
      <CaptureCard
        onFileSelect={vi.fn()}
        previewUrl="blob:http://localhost/test-preview"
        onClear={vi.fn()}
        isLoading={false}
        onDiagnose={onDiagnose}
      />,
    );

    expect(
      screen.getByAltText("Leaf specimen ready for diagnostic evaluation"),
    ).toBeInTheDocument();
    const executeBtn = screen.getByText("Execute Diagnosis");
    expect(executeBtn).toBeInTheDocument();

    fireEvent.click(executeBtn);
    expect(onDiagnose).toHaveBeenCalled();
  });

  it("displays loading state and disables diagnose button during inference", () => {
    render(
      <CaptureCard
        onFileSelect={vi.fn()}
        previewUrl="blob:http://localhost/test-preview"
        onClear={vi.fn()}
        isLoading={true}
        onDiagnose={vi.fn()}
      />,
    );

    expect(screen.getByText("Running Foliar Screening...")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Running Foliar Screening/ }),
    ).toBeDisabled();
  });

  it("displays error banner when error prop is provided", () => {
    render(
      <CaptureCard
        onFileSelect={vi.fn()}
        previewUrl={null}
        onClear={vi.fn()}
        isLoading={false}
        onDiagnose={vi.fn()}
        error="File size exceeds the 10 MB limit."
      />,
    );

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(
      screen.getByText("File size exceeds the 10 MB limit."),
    ).toBeInTheDocument();
  });
});
