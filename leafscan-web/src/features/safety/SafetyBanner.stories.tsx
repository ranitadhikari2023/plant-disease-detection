import type { Meta, StoryObj } from "@storybook/react";
import { SafetyBanner } from "./SafetyBanner";

const meta: Meta<typeof SafetyBanner> = {
  title: "Safety/SafetyBanner",
  component: SafetyBanner,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof SafetyBanner>;

export const Confident: Story = {
  args: {
    status: "confident",
    diagnosisTitle: "Potato - Early Blight",
    warnings: [],
  },
};

export const LateBlightUncertain: Story = {
  args: {
    status: "late_blight_uncertain",
    diagnosisTitle: "Potato - Late Blight",
    warnings: [
      "Potential early lesion, Late Blight confidence (84.15%) is below high-consequence safety threshold (90%).",
    ],
  },
};

export const SpeciesUncertain: Story = {
  args: {
    status: "species_uncertain",
    warnings: [
      "Species confidence (81.23%) is below the safety threshold (90%). Disease classification abstained.",
    ],
  },
};
