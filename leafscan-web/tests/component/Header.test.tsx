import { Header } from "@/components/layout/Header";
import { act, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/diagnose",
}));

// Mock api client
const mockFetchHealth = vi.fn();
const mockFetchReady = vi.fn();
vi.mock("@/lib/api/diagnoseClient", () => ({
  fetchHealth: () => mockFetchHealth(),
  fetchReady: () => mockFetchReady(),
}));

// Mock history store
vi.mock("@/features/history/historyStore", () => ({
  flushOfflineQueue: vi.fn().mockResolvedValue(undefined),
}));

describe("Header Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchHealth.mockResolvedValue({ status: "healthy" });
    mockFetchReady.mockResolvedValue({
      status: "ready",
      engine: "ONNX Runtime",
    });
  });

  it("renders branding, desktop/mobile navigation links, and V3 Ready status", async () => {
    await act(async () => {
      render(<Header />);
    });

    // 1. Branding
    expect(screen.getByText("LeafScan V3")).toBeInTheDocument();
    expect(screen.getByText("Botanical Field Console")).toBeInTheDocument();

    // 2. Navigation links (both desktop and mobile navigation bars are rendered)
    const navs = screen.getAllByRole("navigation");
    expect(navs.length).toBe(2); // Main Navigation & Mobile Navigation

    const diagnoseLinks = screen.getAllByRole("link", { name: /Diagnose/i });
    expect(diagnoseLinks.length).toBe(2);

    const historyLinks = screen.getAllByRole("link", { name: /History/i });
    expect(historyLinks.length).toBe(2);

    const modelLinks = screen.getAllByRole("link", { name: /Model/i });
    expect(modelLinks.length).toBe(2);

    const feedbackLinks = screen.getAllByRole("link", { name: /Feedback/i });
    expect(feedbackLinks.length).toBe(2);

    // 3. Status indicator & Engine tooltip
    await waitFor(() => {
      expect(screen.getByText("V3 Ready")).toBeInTheDocument();
    });
    const statusRegion = screen.getByRole("status");
    expect(statusRegion).toBeInTheDocument();
    expect(statusRegion).toHaveAttribute(
      "title",
      "Inference Engine: ONNX Runtime",
    );
  });

  it("completely omits globe icon, language label, and language dropdown menu", async () => {
    const { container } = await act(async () => {
      return render(<Header />);
    });

    // Language selector controls must NOT be present
    expect(screen.queryByLabelText("Select Language")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByText("EN")).not.toBeInTheDocument();
    expect(screen.queryByText("HI")).not.toBeInTheDocument();
    expect(screen.queryByText("ES")).not.toBeInTheDocument();

    // Verify no select element exists anywhere in the header
    const selectElements = container.querySelectorAll("select");
    expect(selectElements.length).toBe(0);

    // Verify the status container only contains the status badge
    const statusBadge = screen.getByRole("status");
    expect(statusBadge.parentElement?.childElementCount).toBe(1);
  });

  it("renders connecting and degraded states properly", async () => {
    mockFetchReady.mockResolvedValueOnce({ status: "not_ready" });
    mockFetchHealth.mockResolvedValueOnce({ status: "unhealthy" });

    await act(async () => {
      render(<Header />);
    });

    await waitFor(() => {
      expect(screen.getByText("Degraded")).toBeInTheDocument();
    });
  });
});
