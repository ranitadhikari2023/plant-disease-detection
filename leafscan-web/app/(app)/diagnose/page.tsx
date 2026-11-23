"use client";

import { CaptureCard } from "@/features/diagnose/CaptureCard";
import { StageTimeline } from "@/features/diagnose/StageTimeline";
import { useDiagnose } from "@/features/diagnose/useDiagnose";
import React from "react";

export default function DiagnosePage() {
  const {
    previewUrl,
    isLoading,
    error,
    retryAttempt,
    retryMessage,
    isOfflineQueued,
    handleSelectFile,
    handleDiagnose,
    handleReset,
  } = useDiagnose();

  return (
    <div className="space-y-8 max-w-3xl mx-auto">
      <div>
        <h1 className="text-3xl font-serif font-extrabold tracking-tight text-bark-900">
          Field Foliar Screening
        </h1>
        <p className="mt-1.5 text-sm text-bark-600">
          Capture or upload an individual Potato or Tomato leaf. Images are
          evaluated through verified two-tier safety gates.
        </p>
      </div>

      {/* Capture Card */}
      <CaptureCard
        onFileSelect={handleSelectFile}
        previewUrl={previewUrl}
        onClear={handleReset}
        isLoading={isLoading}
        onDiagnose={handleDiagnose}
        error={error}
        isOfflineQueued={isOfflineQueued}
      />

      {/* Processing Timeline during analysis */}
      {isLoading && (
        <div className="pt-2">
          <StageTimeline
            isLoading={isLoading}
            isComplete={false}
            retryAttempt={retryAttempt}
            retryMessage={retryMessage}
          />
        </div>
      )}
    </div>
  );
}
