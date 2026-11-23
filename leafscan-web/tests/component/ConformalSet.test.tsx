import { ConformalSet } from "@/features/safety/ConformalSet";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";

describe("ConformalSet Component", () => {
  it("renders highlighted included classes and muted excluded classes", () => {
    render(
      <ConformalSet
        allClasses={["Early Blight", "Late Blight", "Healthy"]}
        includedClasses={["Early Blight", "Late Blight"]}
        coverageLevel={0.9}
      />,
    );

    expect(screen.getByText("90% Theoretical Coverage")).toBeInTheDocument();
    expect(screen.getByText("Early Blight")).toBeInTheDocument();
    expect(screen.getByText("Late Blight")).toBeInTheDocument();
    expect(screen.getByText("Healthy")).toBeInTheDocument();

    const includedBadges = screen.getAllByLabelText("Included in set");
    expect(includedBadges.length).toBe(2);

    const excludedBadges = screen.getAllByLabelText("Excluded from set");
    expect(excludedBadges.length).toBe(1);
  });
});
