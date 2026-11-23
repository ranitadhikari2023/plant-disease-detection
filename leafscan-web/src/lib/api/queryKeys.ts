export const queryKeys = {
  health: ["system", "health"] as const,
  ready: ["system", "ready"] as const,
  model: ["system", "model"] as const,
  metrics: ["system", "metrics"] as const,
  diagnosis: (id: string) => ["diagnosis", id] as const,
  history: ["history"] as const,
};
