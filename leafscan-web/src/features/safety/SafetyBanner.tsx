"use client";

import type { SafetyStatus } from "@/lib/api/schemas";
import {
  AlertOctagon,
  AlertTriangle,
  HelpCircle,
  ShieldCheck,
} from "lucide-react";
import type React from "react";

export interface SafetyBannerProps {
  status: SafetyStatus;
  warnings?: string[];
  diagnosisTitle?: string;
  className?: string;
}

export const SafetyBanner: React.FC<SafetyBannerProps> = ({
  status,
  warnings = [],
  diagnosisTitle,
  className = "",
}) => {
  if (status === "confident") {
    return (
      <div
        role="alert"
        aria-live="polite"
        className={`rounded-xl border border-teal-500/30 bg-teal-500/10 p-5 text-bark-900 shadow-sm ${className}`}
      >
        <div className="flex items-start gap-3.5">
          <div className="rounded-full bg-teal-500/20 p-2 text-teal-700">
            <ShieldCheck className="h-6 w-6" aria-hidden="true" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-teal-800 bg-teal-500/20 px-2 py-0.5 rounded">
                Verified Diagnosis
              </span>
              <span className="text-xs text-bark-500">
                Safety Gates Cleared
              </span>
            </div>
            <h2 className="mt-1 text-xl font-serif font-bold text-bark-900">
              {diagnosisTitle || "High-Confidence Foliar Assessment"}
            </h2>
            <p className="mt-1 text-sm text-bark-600">
              Both species and crop pathology models exceeded required safety
              gating thresholds (&ge;90%).
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (status === "late_blight_uncertain") {
    return (
      <div
        role="alert"
        aria-live="assertive"
        className={`rounded-xl border-2 border-rust-600/40 bg-rust-600/10 p-5 text-bark-900 shadow-float ${className}`}
      >
        <div className="flex items-start gap-3.5">
          <div className="rounded-full bg-rust-600/20 p-2 text-rust-600">
            <AlertOctagon className="h-6 w-6" aria-hidden="true" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-white bg-rust-600 px-2 py-0.5 rounded">
                High-Consequence Alert
              </span>
              <span className="text-xs font-semibold text-rust-600">
                Late Blight Uncertainty
              </span>
            </div>
            <h2 className="mt-1 text-xl font-serif font-bold text-rust-600">
              Late Blight Warning: Confidence Below 90%
            </h2>
            <p className="mt-1.5 text-sm font-medium text-bark-800">
              Potential foliar lesions detected, but Late Blight confidence is
              below safety thresholds. Do not apply systemic fungicides without
              certified agronomic validation.
            </p>
            {warnings.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs text-bark-700 border-t border-rust-600/20 pt-2 list-disc list-inside">
                {warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (status === "species_uncertain" || status === "out_of_scope") {
    return (
      <div
        role="alert"
        aria-live="polite"
        className={`rounded-xl border border-amber-500/40 bg-amber-500/10 p-5 text-bark-900 shadow-sm ${className}`}
      >
        <div className="flex items-start gap-3.5">
          <div className="rounded-full bg-amber-500/20 p-2 text-amber-700">
            <AlertTriangle className="h-6 w-6" aria-hidden="true" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-amber-900 bg-amber-500/30 px-2 py-0.5 rounded">
                Diagnostic Abstained
              </span>
              <span className="text-xs font-medium text-amber-800">
                Species Uncertainty
              </span>
            </div>
            <h2 className="mt-1 text-xl font-serif font-bold text-amber-900">
              Crop Identification Below 90% Threshold
            </h2>
            <p className="mt-1 text-sm text-bark-700">
              The botanist classifier could not definitively confirm whether
              this specimen is Potato or Tomato. Disease classification was
              abstained to prevent crop misdiagnosis.
            </p>
            {warnings.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs text-bark-600 border-t border-amber-500/20 pt-2 list-disc list-inside">
                {warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    );
  }

  // General uncertainty or error
  return (
    <div
      role="alert"
      className={`rounded-xl border border-bark-300 bg-bark-100 p-5 text-bark-900 shadow-sm ${className}`}
    >
      <div className="flex items-start gap-3.5">
        <div className="rounded-full bg-bark-200 p-2 text-bark-600">
          <HelpCircle className="h-6 w-6" aria-hidden="true" />
        </div>
        <div>
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-bark-700 bg-bark-200 px-2 py-0.5 rounded">
            Marginal Confidence
          </span>
          <h2 className="mt-1 text-xl font-serif font-bold text-bark-900">
            Ambiguous Diagnostic Indications
          </h2>
          <p className="mt-1 text-sm text-bark-600">
            Confidence is below the deterministic safety threshold (70%).
            Re-photograph the specimen in daylight.
          </p>
        </div>
      </div>
    </div>
  );
};
