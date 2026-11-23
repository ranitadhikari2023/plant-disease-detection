import { z } from "zod";

export const SafetyStatusSchema = z.enum([
  "confident",
  "species_uncertain",
  "late_blight_uncertain",
  "disease_uncertain",
  "out_of_scope",
  "error",
]);

export type SafetyStatus = z.infer<typeof SafetyStatusSchema>;

export const MetadataSchema = z.object({
  execution_time_ms: z.number().optional(),
  architecture: z.string().optional(),
  model_bundle: z.string().optional(),
  prediction_set: z.array(z.string()).nullable().optional(),
  request_id: z.string().nullable().optional(),
});

export const DiagnosisResponseSchema = z.object({
  success: z.boolean(),
  model_version: z.string().optional(),
  plant_species: z.string().nullable().optional(),
  species_confidence: z.number().default(0),
  disease: z.string().nullable().optional(),
  disease_confidence: z.number().nullable().optional(),
  diagnosis: z.string(),
  safety_status: SafetyStatusSchema,
  safety_warnings: z.array(z.string()).default([]),
  advisory: z.string().default(""),
  species_probabilities: z.record(z.string(), z.number()).default({}),
  probabilities: z.record(z.string(), z.number()).default({}),
  metadata: MetadataSchema.default({}),
});

export type DiagnosisResponse = z.infer<typeof DiagnosisResponseSchema>;

export const FeedbackRequestSchema = z.object({
  request_id: z.string().nullable().optional(),
  predicted_species: z.string().min(1, "Predicted species is required"),
  predicted_disease: z.string().nullable().optional(),
  observed_species: z.string().nullable().optional(),
  observed_disease: z.string().nullable().optional(),
  feedback_type: z.enum(["agree", "disagree", "unclear", "expert_verified"]),
  agronomic_notes: z.string().max(2000).nullable().optional(),
  idempotency_key: z.string().nullable().optional(),
});

export type FeedbackRequest = z.infer<typeof FeedbackRequestSchema>;

export const FeedbackResponseSchema = z.object({
  success: z.boolean(),
  feedback_id: z.string(),
  status: z.string(),
  message: z.string(),
  idempotency_key: z.string().nullable().optional(),
});

export type FeedbackResponse = z.infer<typeof FeedbackResponseSchema>;

export const ModelManifestSchema = z.object({
  bundle_name: z.string(),
  bundle_version: z.string(),
  runtime: z.string(),
  opset: z.number(),
  disclaimer: z.string(),
  models: z.record(
    z.string(),
    z.object({
      filename: z.string(),
      architecture: z.string(),
      sha256: z.string(),
      classes: z.array(z.string()),
      input_node: z.string(),
      output_node: z.string(),
    }),
  ),
  safety_gates: z.object({
    species_confidence_threshold: z.number(),
    late_blight_confidence_threshold: z.number(),
    general_disease_confidence_threshold: z.number(),
  }),
  input_spec: z.record(z.string(), z.unknown()),
  engine_info: z.object({
    name: z.string(),
    provider: z.string(),
    intra_op_threads: z.number(),
    inter_op_threads: z.number(),
  }),
  metrics: z.record(
    z.string(),
    z.object({
      accuracy: z.number(),
      macro_f1: z.number(),
    }),
  ),
});

export type ModelManifest = z.infer<typeof ModelManifestSchema>;

export const HealthResponseSchema = z.object({
  status: z.string(),
  model_version: z.string().optional(),
  architecture: z.string().optional(),
  models_loaded: z.boolean().optional(),
});

export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export const ReadyResponseSchema = z.object({
  status: z.string(),
  bundle_version: z.string().optional(),
  engine: z.string().optional(),
  execution_provider: z.string().optional(),
  models_verified: z.array(z.string()).optional(),
});

export type ReadyResponse = z.infer<typeof ReadyResponseSchema>;

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    request_id?: string;
    details?: unknown;
  };
  status: number;
}
