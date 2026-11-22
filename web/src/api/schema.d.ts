/**
 * web/src/api/schema.d.ts
 * TypeScript Type Definitions for LeafScan V2 OpenAPI Schema
 */

export interface paths {
  "/health": {
    get: {
      summary: "Lightweight System Health Check";
      description: "Lightweight health check endpoint. Reports operational status and model caching telemetry without triggering heavy deep learning model deserialization.";
      responses: {
        200: {
          content: {
            "application/json": HealthResponse;
          };
        };
      };
    };
  };
  "/v2/diagnose": {
    post: {
      summary: "Execute Hierarchical Foliar Diagnosis";
      description: "Receives an uploaded leaf photograph and executes two-tier hierarchical screening.";
      requestBody: {
        content: {
          "multipart/form-data": {
            image: Blob | File;
          };
        };
      };
      responses: {
        200: {
          content: {
            "application/json": DiagnosisResponse;
          };
        };
        400: {
          content: {
            "application/json": HTTPErrorResponse;
          };
        };
        413: {
          content: {
            "application/json": HTTPErrorResponse;
          };
        };
        415: {
          content: {
            "application/json": HTTPErrorResponse;
          };
        };
        422: {
          content: {
            "application/json": HTTPValidationError;
          };
        };
        500: {
          content: {
            "application/json": DiagnosisErrorResponse;
          };
        };
      };
    };
  };
  "/": {
    get: {
      summary: "Serve LeafScan SPA";
      responses: {
        200: {
          content: {
            "text/html": string;
          };
        };
      };
    };
  };
}

export type SafetyStatus =
  | "confident"
  | "species_uncertain"
  | "late_blight_uncertain"
  | "disease_uncertain"
  | "error";

export interface DiagnosisMetadata {
  execution_time_ms: number | null;
  architecture: string;
}

export interface DiagnosisResponse {
  success: boolean;
  model_version: string;
  plant_species: string | null;
  species_confidence: number;
  disease: string | null;
  disease_confidence: number | null;
  diagnosis: string;
  safety_status: SafetyStatus | string;
  safety_warnings: string[];
  advisory: string;
  species_probabilities: Record<string, number>;
  probabilities: Record<string, number>;
  metadata: DiagnosisMetadata;
  error?: {
    type: string;
    message: string;
  };
}

export interface DiagnosisErrorResponse {
  success: false;
  model_version: string;
  plant_species: null;
  species_confidence: 0.0;
  disease: null;
  disease_confidence: 0.0;
  diagnosis: string;
  safety_status: "error" | string;
  safety_warnings: string[];
  advisory: string;
  species_probabilities: Record<string, number>;
  probabilities: Record<string, number>;
  error?: {
    type: string;
    message: string;
  };
}

export interface HealthResponse {
  status: "healthy" | "unavailable" | string;
  model_version?: string;
  architecture?: string;
  models_loaded?: boolean;
  error?: string;
}

export interface HTTPErrorResponse {
  detail: string;
}

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}

export interface HTTPValidationError {
  detail: ValidationError[];
}
