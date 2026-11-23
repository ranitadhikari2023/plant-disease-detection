"use client";

import { fetchModelManifest } from "@/lib/api/diagnoseClient";
import { queryKeys } from "@/lib/api/queryKeys";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Check, CheckCircle2, Copy } from "lucide-react";
import React, { useState } from "react";

export function ModelTransparencyView() {
  const [copiedSha, setCopiedSha] = useState<string | null>(null);

  const {
    data: manifest,
    isLoading,
    error,
  } = useQuery({
    queryKey: queryKeys.model,
    queryFn: fetchModelManifest,
    staleTime: 1000 * 60 * 5,
  });

  const handleCopySha = (sha: string) => {
    navigator.clipboard.writeText(sha);
    setCopiedSha(sha);
    setTimeout(() => setCopiedSha(null), 2000);
  };

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-leaf-600 border-t-transparent mb-4" />
        <p className="text-sm font-medium text-bark-600">
          Loading verified model manifest...
        </p>
      </div>
    );
  }

  if (error || !manifest) {
    return (
      <div className="max-w-md mx-auto text-center py-12">
        <div className="rounded-full bg-rust-600/10 p-4 text-rust-600 w-fit mx-auto mb-4">
          <AlertTriangle className="h-8 w-8" />
        </div>
        <h2 className="text-xl font-serif font-bold text-bark-900">
          Failed to Load Model Manifest
        </h2>
        <p className="mt-2 text-xs text-bark-500">
          {(error as Error)?.message ||
            "Could not retrieve operational bundle specifications."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="pb-4 border-b border-bark-200">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-teal-800 bg-teal-500/20 px-2 py-0.5 rounded">
            Cryptographically Verified
          </span>
          <span className="text-xs font-mono text-bark-500">
            Bundle Version: {manifest.bundle_version}
          </span>
        </div>
        <h1 className="text-3xl font-serif font-extrabold text-bark-900">
          Model Transparency & Verification
        </h1>
        <p className="mt-1 text-sm text-bark-600">
          Live technical specifications, SHA-256 model digests, and
          deterministic safety thresholds fetched directly from the backend.
        </p>
      </div>

      {/* Production Disclaimer */}
      <div className="rounded-2xl border border-leaf-200 bg-leaf-50/60 p-5 text-xs text-bark-700 leading-relaxed shadow-xs">
        <strong className="block font-semibold text-leaf-900 mb-1">
          Production Serving Disclaimer
        </strong>
        {manifest.disclaimer}
      </div>

      {/* Verified Model Artifacts */}
      <section aria-labelledby="models-heading" className="space-y-4">
        <h2
          id="models-heading"
          className="text-lg font-serif font-bold text-bark-900"
        >
          Production ONNX Models
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(manifest.models).map(([key, model]) => (
            <div
              key={key}
              className="rounded-2xl border border-bark-200 bg-white p-5 shadow-xs flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-bold text-leaf-700 uppercase">
                    {key.replace("_", " ")}
                  </span>
                  <CheckCircle2 className="h-4 w-4 text-teal-600" />
                </div>
                <h3 className="text-base font-serif font-bold text-bark-900 mb-1">
                  {model.architecture}
                </h3>
                <span className="text-xs font-mono text-bark-500 block mb-3">
                  {model.filename}
                </span>

                <div className="space-y-1 mb-4">
                  <span className="text-[11px] font-mono text-bark-400 block">
                    Classes ({model.classes.length}):
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {model.classes.map((cls) => (
                      <span
                        key={cls}
                        className="text-[10px] font-mono bg-bark-100 px-1.5 py-0.5 rounded text-bark-700"
                      >
                        {cls}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-bark-100">
                <span className="text-[10px] font-mono text-bark-400 block mb-1">
                  SHA-256 Digest:
                </span>
                <div className="flex items-center justify-between bg-bark-50 p-1.5 rounded border border-bark-200">
                  <code
                    className="text-[9px] font-mono truncate max-w-[170px]"
                    title={model.sha256}
                  >
                    {model.sha256}
                  </code>
                  <button
                    type="button"
                    onClick={() => handleCopySha(model.sha256)}
                    className="p-1 hover:text-bark-900 text-bark-500"
                    aria-label="Copy SHA-256 hash"
                  >
                    {copiedSha === model.sha256 ? (
                      <Check className="h-3 w-3 text-teal-600" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Safety Gate Thresholds */}
      <section aria-labelledby="gates-heading" className="space-y-4">
        <h2
          id="gates-heading"
          className="text-lg font-serif font-bold text-bark-900"
        >
          Deterministic Safety Thresholds
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="rounded-xl border border-bark-200 bg-white p-4 shadow-xs">
            <span className="text-xs font-mono text-bark-500 block mb-1">
              Gate 1: Species Confidence
            </span>
            <span className="text-2xl font-serif font-bold text-bark-900">
              {(
                manifest.safety_gates.species_confidence_threshold * 100
              ).toFixed(0)}
              %
            </span>
            <p className="mt-1 text-[11px] text-bark-500">
              Abstains disease inference if species is uncertain.
            </p>
          </div>

          <div className="rounded-xl border border-rust-600/30 bg-rust-50/30 p-4 shadow-xs">
            <span className="text-xs font-mono text-rust-700 block mb-1">
              Gate 2: Late Blight Escalation
            </span>
            <span className="text-2xl font-serif font-bold text-rust-800">
              {(
                manifest.safety_gates.late_blight_confidence_threshold * 100
              ).toFixed(0)}
              %
            </span>
            <p className="mt-1 text-[11px] text-bark-500">
              High-consequence pathogen threshold.
            </p>
          </div>

          <div className="rounded-xl border border-bark-200 bg-white p-4 shadow-xs">
            <span className="text-xs font-mono text-bark-500 block mb-1">
              Gate 2: General Disease
            </span>
            <span className="text-2xl font-serif font-bold text-bark-900">
              {(
                manifest.safety_gates.general_disease_confidence_threshold * 100
              ).toFixed(0)}
              %
            </span>
            <p className="mt-1 text-[11px] text-bark-500">
              Minimum certainty for foliar diagnosis.
            </p>
          </div>
        </div>
      </section>

      {/* Runtime & Engine Specs */}
      <section aria-labelledby="runtime-heading" className="space-y-4">
        <h2
          id="runtime-heading"
          className="text-lg font-serif font-bold text-bark-900"
        >
          Inference Engine Execution Specs
        </h2>

        <div className="rounded-2xl border border-bark-200 bg-white p-5 shadow-xs">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
            <div>
              <span className="text-bark-400 block">Runtime:</span>
              <span className="font-semibold text-bark-900">
                {manifest.runtime} (Opset {manifest.opset})
              </span>
            </div>
            <div>
              <span className="text-bark-400 block">Execution Provider:</span>
              <span className="font-semibold text-bark-900">
                {manifest.engine_info.provider}
              </span>
            </div>
            <div>
              <span className="text-bark-400 block">Intra-Op Threads:</span>
              <span className="font-semibold text-bark-900">
                {manifest.engine_info.intra_op_threads}
              </span>
            </div>
            <div>
              <span className="text-bark-400 block">Inter-Op Threads:</span>
              <span className="font-semibold text-bark-900">
                {manifest.engine_info.inter_op_threads}
              </span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
