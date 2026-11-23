import type { Meta, StoryObj } from "@storybook/react";
import { ConfidenceMeter } from "./ConfidenceMeter";

const meta: Meta<typeof ConfidenceMeter> = {
  title: "Safety/ConfidenceMeter",
  component: ConfidenceMeter,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof ConfidenceMeter>;

export const AboveThreshold: Story = {
  args: {
    label: "Species Confidence",
    value: 0.985,
    threshold: 0.90,
  },
};

export const BelowThreshold: Story = {
  args: {
    label: "Late Blight Confidence",
    value: 0.84,
    threshold: 0.90,
  },
};
