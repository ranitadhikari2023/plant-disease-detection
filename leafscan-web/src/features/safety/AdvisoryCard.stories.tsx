import type { Meta, StoryObj } from "@storybook/react";
import { AdvisoryCard } from "./AdvisoryCard";

const meta: Meta<typeof AdvisoryCard> = {
  title: "Safety/AdvisoryCard",
  component: AdvisoryCard,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof AdvisoryCard>;

export const Info: Story = {
  args: {
    severity: "info",
    title: "Sanitation Protocol",
    body: "Ensure proper spacing and prune lower senescent leaves to improve air circulation.",
  },
};

export const ChemicalSuppressionRestriction: Story = {
  args: {
    code: "chemical_suppression",
    severity: "warning",
    title: "Pesticide Warning",
    body: "Consult certified extension guidelines. Do not apply unregistered chemical mixtures.",
    actions: [
      { label: "Contact Extension Agent", tel: "1-800-AGRI-SAFE" },
      { label: "USDA Pest Guide", url: "https://www.usda.gov" },
    ],
  },
};
