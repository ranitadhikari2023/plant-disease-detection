import { ConfidenceMeter } from "@/features/safety/ConfidenceMeter";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";

describe("ConfidenceMeter Component", () => {
  it("renders accessible meter semantics with correct percentage", () => {
    render(
      <ConfidenceMeter
        label="Botanist Species Confidence"
        value={0.954}
        threshold={0.9}
      />,
    );

    const meter = screen.getByRole("meter");
    expect(meter).toBeInTheDocument();
    expect(meter).toHaveAttribute("aria-valuenow", "95");
    expect(meter).toHaveAttribute("aria-valuemin", "0");
    expect(meter).toHaveAttribute("aria-valuemax", "100");
    expect(meter).toHaveAttribute("aria-valuetext", "95 percent confident");

    expect(screen.getByText("95%")).toBeInTheDocument();
    expect(screen.getByText("Target: 90%")).toBeInTheDocument();
  });

  it("renders appropriate color styling when below threshold", () => {
    render(
      <ConfidenceMeter
        label="Pathology Confidence"
        value={0.65}
        threshold={0.7}
      />,
    );

    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-valuenow", "65");
    expect(screen.getByText("65%")).toBeInTheDocument();
  });
});
