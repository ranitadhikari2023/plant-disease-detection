import type { Preview } from "@storybook/react";
import "../src/styles/globals.css";

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: "^on[A-Z].*" },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    backgrounds: {
      default: "leaf-50",
      values: [
        { name: "leaf-50", value: "#F2F7F1" },
        { name: "white", value: "#FFFFFF" },
        { name: "bark-900", value: "#1F2420" },
      ],
    },
  },
};

export default preview;
