import { AdvisoryCard } from "@/features/safety/AdvisoryCard";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";

describe("AdvisoryCard Component", () => {
  it("renders advisory text verbatim as provided by the backend", () => {
    const backendMessage =
      "Symptoms are characteristic of Early Blight (Alternaria solani). Prune lower senescent leaves.";
    render(<AdvisoryCard body={backendMessage} title="Field Guidance" />);

    expect(screen.getByText("Field Guidance")).toBeInTheDocument();
    expect(screen.getByText(backendMessage)).toBeInTheDocument();
  });

  it("enforces chemical suppression restriction and displays pesticide warning", () => {
    const chemicalText =
      "Consider registered fungicide applications according to extension schedules.";
    render(<AdvisoryCard code="chemical_suppression" body={chemicalText} />);

    // Must show explicit pesticide warning and forbid recipes
    expect(screen.getByText("Pesticide Safety Warning:")).toBeInTheDocument();
    expect(
      screen.getByText(
        /This application does not provide dosage recipes, chemical mixtures, or spray schedules./,
      ),
    ).toBeInTheDocument();
  });

  it("renders external referral links and phone actions correctly", () => {
    render(
      <AdvisoryCard
        body="Contact local agronomy office."
        actions={[
          { label: "Extension Help", tel: "1-800-AGRI-HELP" },
          { label: "USDA Portal", url: "https://www.usda.gov" },
        ]}
      />,
    );

    expect(screen.getByText(/Extension Help/)).toBeInTheDocument();
    expect(screen.getByText(/USDA Portal/)).toBeInTheDocument();
  });
});
