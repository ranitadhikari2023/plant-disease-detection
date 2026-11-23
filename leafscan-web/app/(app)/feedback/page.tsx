"use client";

import { submitFeedback } from "@/lib/api/diagnoseClient";
import { type FeedbackRequest, FeedbackRequestSchema } from "@/lib/api/schemas";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle,
  MessageSquare,
  Send,
} from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import React, { useState, Suspense } from "react";
import { useForm } from "react-hook-form";

function FeedbackFormContent() {
  const searchParams = useSearchParams();
  const reqId = searchParams?.get("request_id") || undefined;
  const initialSpecies = searchParams?.get("species") || "";
  const initialDisease = searchParams?.get("disease") || "";

  const [submittedId, setSubmittedId] = useState<string | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    reset,
  } = useForm<FeedbackRequest>({
    resolver: zodResolver(FeedbackRequestSchema),
    defaultValues: {
      request_id: reqId,
      predicted_species: initialSpecies || "Potato",
      predicted_disease: initialDisease || undefined,
      observed_species: initialSpecies || undefined,
      feedback_type: "agree",
      agronomic_notes: "",
    },
  });

  const onSubmit = async (data: FeedbackRequest) => {
    setSubmissionError(null);
    try {
      // Add idempotency key
      const idempotencyKey = `fb_key_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      const res = await submitFeedback({
        ...data,
        idempotency_key: idempotencyKey,
      });
      setSubmittedId(res.feedback_id);
      reset();
    } catch (err: any) {
      setSubmissionError(err.message || "Failed to submit feedback to backend");
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="pb-4 border-b border-bark-200">
        <Link
          href="/diagnose"
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-bark-600 hover:text-leaf-700 transition-colors mb-3"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Scanner
        </Link>
        <h1 className="text-3xl font-serif font-extrabold text-bark-900">
          Field Validation Feedback
        </h1>
        <p className="mt-1 text-sm text-bark-600">
          Submit certified agronomist observations or laboratory PCR
          confirmations to audit model inference.
        </p>
      </div>

      {submittedId ? (
        <div className="rounded-2xl border border-teal-500/30 bg-teal-500/10 p-6 text-center shadow-xs">
          <div className="rounded-full bg-teal-500/20 p-3 text-teal-700 w-fit mx-auto mb-3">
            <CheckCircle className="h-8 w-8" />
          </div>
          <h2 className="text-xl font-serif font-bold text-bark-900">
            Feedback Recorded
          </h2>
          <p className="mt-1.5 text-xs text-bark-600 max-w-md mx-auto">
            Your field validation report has been saved with confirmation ID:
          </p>
          <code className="mt-2 inline-block font-mono font-bold text-xs bg-white px-3 py-1 rounded border border-teal-200 text-teal-800">
            {submittedId}
          </code>
          <p className="mt-3 text-[11px] text-bark-500">
            Submissions require certified agricultural extension validation
            before dataset curation.
          </p>
          <div className="mt-6 flex justify-center gap-3">
            <button
              type="button"
              onClick={() => setSubmittedId(null)}
              className="px-4 py-2 rounded-xl bg-white border border-bark-300 text-xs font-semibold text-bark-800 hover:bg-bark-50"
            >
              Submit Another Report
            </button>
            <Link
              href="/diagnose"
              className="px-4 py-2 rounded-xl bg-leaf-600 text-xs font-semibold text-white shadow-xs hover:bg-leaf-700"
            >
              Return to Diagnosis
            </Link>
          </div>
        </div>
      ) : (
        <form
          onSubmit={handleSubmit(onSubmit)}
          className="space-y-5 rounded-2xl border border-bark-200 bg-white p-6 shadow-sm"
        >
          {reqId && (
            <div className="rounded-lg bg-bark-50 p-3 border border-bark-200 text-xs">
              <span className="font-mono text-bark-500">
                Associated Request ID:{" "}
              </span>
              <code className="font-mono font-semibold text-bark-800">
                {reqId}
              </code>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-bark-900 mb-1">
                Predicted Species
              </label>
              <input
                type="text"
                {...register("predicted_species")}
                className="w-full rounded-xl border border-bark-200 bg-bark-50/50 px-3.5 py-2.5 text-xs text-bark-900 focus:border-leaf-600 focus:outline-none"
              />
              {errors.predicted_species && (
                <span className="text-[11px] text-rust-600 mt-1 block">
                  {errors.predicted_species.message}
                </span>
              )}
            </div>

            <div>
              <label className="block text-xs font-semibold text-bark-900 mb-1">
                Predicted Disease
              </label>
              <input
                type="text"
                {...register("predicted_disease")}
                className="w-full rounded-xl border border-bark-200 bg-bark-50/50 px-3.5 py-2.5 text-xs text-bark-900 focus:border-leaf-600 focus:outline-none"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-bark-900 mb-1">
                Observed Field Species
              </label>
              <select
                {...register("observed_species")}
                className="w-full rounded-xl border border-bark-200 bg-white px-3.5 py-2.5 text-xs text-bark-900 focus:border-leaf-600 focus:outline-none"
              >
                <option value="Potato">Potato (Solanum tuberosum)</option>
                <option value="Tomato">Tomato (Solanum lycopersicum)</option>
                <option value="Other">Other / Non-Solanaceous</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-bark-900 mb-1">
                Observed Pathology
              </label>
              <select
                {...register("observed_disease")}
                className="w-full rounded-xl border border-bark-200 bg-white px-3.5 py-2.5 text-xs text-bark-900 focus:border-leaf-600 focus:outline-none"
              >
                <option value="Healthy">Healthy (No blight)</option>
                <option value="Early Blight">
                  Early Blight (Alternaria solani)
                </option>
                <option value="Late Blight">
                  Late Blight (Phytophthora infestans)
                </option>
                <option value="Abiotic">Abiotic Injury / Scorching</option>
                <option value="Other">Other Pathogen</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-bark-900 mb-1">
              Field Assessment Agreement
            </label>
            <select
              {...register("feedback_type")}
              className="w-full rounded-xl border border-bark-200 bg-white px-3.5 py-2.5 text-xs text-bark-900 focus:border-leaf-600 focus:outline-none"
            >
              <option value="agree">Agree with Model Diagnosis</option>
              <option value="disagree">Disagree with Model Diagnosis</option>
              <option value="unclear">
                Ambiguous / Unclear Field Specimen
              </option>
              <option value="expert_verified">
                Certified Laboratory / PCR Verified
              </option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-bark-900 mb-1">
              Agronomic Observations / Laboratory Findings
            </label>
            <textarea
              rows={4}
              {...register("agronomic_notes")}
              placeholder="Detail symptom progression, weather context (humidity/rainfall), spore microscopy, or crop stage..."
              className="w-full rounded-xl border border-bark-200 bg-white p-3 text-xs text-bark-900 placeholder:text-bark-400 focus:border-leaf-600 focus:outline-none"
            />
          </div>

          {submissionError && (
            <div className="flex items-center gap-2 rounded-lg border border-rust-600/30 bg-rust-50 p-3 text-xs text-rust-700">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{submissionError}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-leaf-600 px-6 py-3 text-xs font-semibold text-white shadow-sm hover:bg-leaf-700 active:scale-98 disabled:opacity-60 transition-all min-h-[44px]"
          >
            <Send className="h-4 w-4" />
            {isSubmitting
              ? "Transmitting Report..."
              : "Submit Diagnostic Feedback"}
          </button>
        </form>
      )}
    </div>
  );
}

export default function FeedbackPage() {
  return (
    <Suspense
      fallback={
        <div className="text-center py-12 text-sm text-bark-500">
          Loading feedback form...
        </div>
      }
    >
      <FeedbackFormContent />
    </Suspense>
  );
}
