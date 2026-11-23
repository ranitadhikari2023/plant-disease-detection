"use client";

import {
  type StoredDiagnosis,
  getDiagnosis,
} from "@/features/history/historyStore";
import { AdvisoryCard } from "@/features/safety/AdvisoryCard";
import { ConfidenceMeter } from "@/features/safety/ConfidenceMeter";
import { ConformalSet } from "@/features/safety/ConformalSet";
import { SafetyBanner } from "@/features/safety/SafetyBanner";
import {
  AlertTriangle,
  ArrowLeft,
  Calendar,
  Check,
  Clock,
  Copy,
  Cpu,
  MessageSquare,
  RotateCcw,
} from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import React, { useEffect, useState, Suspense } from "react";

function ResultContent() {
  const searchParams = useSearchParams();
  const id = searchParams?.get("id");

  const [diagnosis, setDiagnosis] = useState<StoredDiagnosis | null>(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!id) {
      setLoading(false);
      return;
    }

    getDiagnosis(id).then((stored) => {
      setDiagnosis(stored);
      setLoading(false);
    });
  }, [id]);

  const copyRequestId = () => {
    if (!diagnosis?.requestId) return;
    navigator.clipboard.writeText(diagnosis.requestId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-leaf-600 border-t-transparent mb-4" />
        <p className="text-sm font-medium text-bark-600">
          Retrieving diagnostic audit record...
        </p>
      </div>
    );
  }

  if (!diagnosis) {
    return (
      <div className="max-w-md mx-auto text-center py-12">
        <div className="rounded-full bg-bark-100 p-4 text-bark-600 w-fit mx-auto mb-4">
          <AlertTriangle className="h-8 w-8" />
        </div>
        <h2 className="text-xl font-serif font-bold text-bark-900">
          Diagnosis Record Not Found
        </h2>
        <p className="mt-2 text-xs text-bark-500">
          The requested diagnosis ID could not be loaded from local storage.
        </p>
        <div className="mt-6">
          <Link
            href="/diagnose"
            className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-4 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-leaf-700"
          >
            <ArrowLeft className="h-4 w-4" />
            Perform New Diagnosis
          </Link>
        </div>
      </div>
    );
  }

  const {
    safetyStatus,
    plantSpecies,
    disease,
    speciesConfidence,
    diseaseConfidence,
    rawResponse,
  } = diagnosis;
  const isSpeciesUncertain =
    safetyStatus === "species_uncertain" || safetyStatus === "out_of_scope";
  const isLateBlightUncertain = safetyStatus === "late_blight_uncertain";

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Navigation & Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-bark-200">
        <Link
          href="/diagnose"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-bark-600 hover:text-leaf-700 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Scan Another Leaf
        </Link>

        {/* Copyable Request ID */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-bark-500">ID:</span>
          <code className="text-xs font-mono font-bold text-bark-800 bg-bark-100 px-2 py-0.5 rounded">
            {diagnosis.requestId}
          </code>
          <button
            type="button"
            onClick={copyRequestId}
            className="p-1 rounded text-bark-500 hover:text-bark-900 hover:bg-bark-100 transition-colors"
            aria-label="Copy request ID"
            title="Copy Request ID"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-teal-600" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* 1. Deterministic Safety Banner (Highest Priority) */}
      <SafetyBanner
        status={safetyStatus}
        warnings={diagnosis.safetyWarnings}
        diagnosisTitle={diagnosis.diagnosis}
      />

      {/* Main Grid: Specimen Image & Primary Classification */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Thumbnail & Specimen Details */}
        <div className="rounded-2xl border border-bark-200 bg-white p-5 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-serif font-bold text-bark-900 mb-3">
              Analyzed Leaf Specimen
            </h3>
            {diagnosis.thumbnail ? (
              <div className="relative aspect-video sm:aspect-square w-full rounded-xl overflow-hidden bg-bark-900 border border-bark-200">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={diagnosis.thumbnail}
                  alt="Examined foliar specimen"
                  className="h-full w-full object-contain"
                />
              </div>
            ) : (
              <div className="aspect-video sm:aspect-square w-full rounded-xl bg-bark-50 border border-bark-200 flex items-center justify-center text-xs text-bark-400">
                No preview image stored
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-bark-100 flex flex-wrap justify-between text-xs font-mono text-bark-500 gap-2">
            <span className="flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" />
              {new Date(diagnosis.timestamp).toLocaleDateString()}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" />
              {rawResponse.metadata.execution_time_ms
                ? `${rawResponse.metadata.execution_time_ms} ms`
                : "Synchronous"}
            </span>
            <span className="flex items-center gap-1">
              <Cpu className="h-3.5 w-3.5" />
              {diagnosis.modelVersion}
            </span>
          </div>
        </div>

        {/* Right: Diagnostic Card (strictly conditional on safety_status) */}
        <div className="flex flex-col gap-5">
          {/* Species Assessment Card */}
          <div className="rounded-2xl border border-bark-200 bg-white p-5 shadow-xs">
            <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-bark-500 mb-2">
              Tier 1: Botanist Species Classifier
            </h3>
            <div className="flex items-baseline justify-between mb-4">
              <span className="text-2xl font-serif font-bold text-bark-900">
                {plantSpecies || "Unknown Crop"}
              </span>
              <span className="text-xs text-bark-500">
                {plantSpecies === "Potato"
                  ? "Solanum tuberosum"
                  : plantSpecies === "Tomato"
                    ? "Solanum lycopersicum"
                    : "Unidentified"}
              </span>
            </div>

            <ConfidenceMeter
              label="Species Confidence"
              value={speciesConfidence}
              threshold={0.9}
            />
          </div>

          {/* Disease Assessment Card: HIDDEN if species_uncertain to prevent misleading output */}
          {!isSpeciesUncertain ? (
            <div
              className={`rounded-2xl border p-5 shadow-xs ${
                isLateBlightUncertain
                  ? "border-rust-600/30 bg-rust-50/20"
                  : "border-bark-200 bg-white"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-bark-500">
                  Tier 2: CBAM Crop Specialist
                </h3>
                {isLateBlightUncertain && (
                  <span className="text-[10px] font-mono font-bold uppercase text-rust-600 bg-rust-600/10 px-2 py-0.5 rounded">
                    Unconfirmed Lesion
                  </span>
                )}
              </div>

              <div className="flex items-baseline justify-between mb-4">
                <span className="text-2xl font-serif font-bold text-bark-900">
                  {disease || "Indeterminate"}
                </span>
                {diseaseConfidence !== null && (
                  <span className="text-xs font-mono text-bark-500">
                    Threshold: {disease === "Late Blight" ? "90%" : "70%"}
                  </span>
                )}
              </div>

              {diseaseConfidence !== null && (
                <ConfidenceMeter
                  label="Pathology Confidence"
                  value={diseaseConfidence}
                  threshold={disease === "Late Blight" ? 0.9 : 0.7}
                />
              )}
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-amber-500/40 bg-amber-500/5 p-5 text-center">
              <h4 className="text-sm font-serif font-bold text-amber-900">
                Specialist Screening Abstained
              </h4>
              <p className="mt-1 text-xs text-bark-600 max-w-sm mx-auto">
                Because species identification confidence fell below the 90%
                boundary, foliar disease inference was not executed.
              </p>
              <div className="mt-4">
                <Link
                  href="/diagnose"
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white border border-amber-300 text-xs font-semibold text-amber-900 hover:bg-amber-50"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Retake with Better Lighting
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 2. Conformal Prediction Set (if provided by backend) */}
      {!isSpeciesUncertain && diagnosis.predictionSet && (
        <ConformalSet
          includedClasses={diagnosis.predictionSet}
          coverageLevel={0.9}
        />
      )}

      {/* 3. Deterministic Agronomic Advisory (Verbatim from Backend) */}
      {diagnosis.advisory && (
        <AdvisoryCard
          code={
            isSpeciesUncertain
              ? "species_uncertain"
              : isLateBlightUncertain
                ? "late_blight_uncertain"
                : undefined
          }
          severity={
            isLateBlightUncertain
              ? "critical"
              : isSpeciesUncertain
                ? "caution"
                : "info"
          }
          title="Field Management Advisory"
          body={diagnosis.advisory}
          actions={[
            {
              label: "Consult Agricultural Extension",
              url: "https://www.ars.usda.gov",
            },
          ]}
        />
      )}

      {/* Footer Actions */}
      <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-bark-200">
        <Link
          href={`/feedback?request_id=${encodeURIComponent(diagnosis.requestId)}&species=${encodeURIComponent(plantSpecies || "")}&disease=${encodeURIComponent(disease || "")}`}
          className="inline-flex items-center gap-2 text-xs font-semibold text-leaf-700 hover:text-leaf-800 transition-colors"
        >
          <MessageSquare className="h-4 w-4" />
          Provide Agronomist Field Feedback
        </Link>

        <Link
          href="/diagnose"
          className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-5 py-2.5 text-xs font-semibold text-white shadow-sm hover:bg-leaf-700"
        >
          Diagnose Another Leaf
        </Link>
      </div>
    </div>
  );
}

export default function ResultPage() {
  return (
    <Suspense
      fallback={
        <div className="text-center py-12 text-sm text-bark-500">
          Loading diagnostic result...
        </div>
      }
    >
      <ResultContent />
    </Suspense>
  );
}
