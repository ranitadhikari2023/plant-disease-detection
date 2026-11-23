"use client";

import type { SafetyStatus } from "@/lib/api/schemas";
import { AlertCircle, CheckCircle2, Circle, SkipForward } from "lucide-react";
import type React from "react";
import { useEffect, useState } from "react";

export interface StageTimelineProps {
  isLoading: boolean;
  isComplete: boolean;
  safetyStatus?: SafetyStatus;
  retryAttempt?: number;
  retryMessage?: string;
}

interface Stage {
  id: string;
  label: string;
  description: string;
}

const STAGES: Stage[] = [
  {
    id: "intake",
    label: "Image Intake",
    description:
      "Sanitizing raster, matting background & validating dimensions",
  },
  {
    id: "species",
    label: "Species Classifier",
    description: "Botanist screening: Potato vs Tomato",
  },
  {
    id: "gate1",
    label: "Safety Gate 1",
    description: "Evaluating 90% species confidence boundary",
  },
  {
    id: "specialist",
    label: "Crop Specialist",
    description: "CBAM attention screening for Early/Late Blight",
  },
  {
    id: "gate2",
    label: "Safety Gate 2",
    description:
      "Evaluating Late Blight (90%) and general disease (70%) safety",
  },
  {
    id: "serialization",
    label: "Response Formulation",
    description: "Assembling prediction sets, advisories & cryptographic audit",
  },
];

export const StageTimeline: React.FC<StageTimelineProps> = ({
  isLoading,
  isComplete,
  safetyStatus,
  retryAttempt,
  retryMessage,
}) => {
  const [activeStageIndex, setActiveStageIndex] = useState(0);

  useEffect(() => {
    if (!isLoading) {
      if (isComplete) {
        setActiveStageIndex(STAGES.length);
      }
      return;
    }

    setActiveStageIndex(0);
    // Simulate natural progress curve through logical stages
    const intervals = [300, 500, 400, 600, 400, 300];
    let currentIndex = 0;

    const timer = setInterval(() => {
      currentIndex++;
      if (currentIndex < STAGES.length - 1) {
        setActiveStageIndex(currentIndex);
      } else {
        clearInterval(timer);
      }
    }, intervals[currentIndex] || 400);

    return () => clearInterval(timer);
  }, [isLoading, isComplete]);

  // Determine stage status
  const getStageStatus = (index: number) => {
    if (isLoading) {
      if (index < activeStageIndex) return "completed";
      if (index === activeStageIndex) return "active";
      return "pending";
    }

    if (isComplete) {
      // If species uncertain or out of scope, stages 3, 4, 5 are skipped because processing abstained at Gate 1
      if (
        safetyStatus === "species_uncertain" ||
        safetyStatus === "out_of_scope"
      ) {
        if (index <= 2) return "completed";
        if (index === 2) return "abstained";
        return "skipped";
      }
      return "completed";
    }

    return "pending";
  };

  return (
    <div
      role="region"
      aria-label="Diagnostic Processing Stages"
      className="w-full bg-white rounded-xl border border-bark-200/60 p-6 shadow-sm"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-serif font-semibold text-bark-900">
          Logical Execution Sequence
        </h3>
        {retryAttempt && retryAttempt > 0 ? (
          <span className="text-xs font-mono font-medium px-2.5 py-1 rounded bg-amber-500/10 text-amber-700 animate-pulse">
            {retryMessage || `Service busy. Retry ${retryAttempt}/3...`}
          </span>
        ) : null}
      </div>

      <p className="text-xs text-bark-500 mb-6">
        The backend executes synchronous two-tier verification. Stages below
        illustrate the server evaluation pipeline.
      </p>

      <ol className="relative border-s border-bark-200 ms-3 space-y-6">
        {STAGES.map((stage, idx) => {
          const status = getStageStatus(idx);

          return (
            <li key={stage.id} className="ms-6">
              <span className="absolute -start-3 flex h-6 w-6 items-center justify-center rounded-full bg-white ring-4 ring-white">
                {status === "completed" && (
                  <CheckCircle2
                    className="h-5 w-5 text-leaf-600"
                    aria-label="Completed"
                  />
                )}
                {status === "active" && (
                  <span
                    className="h-3 w-3 rounded-full bg-leaf-600 animate-ping"
                    aria-label="In Progress"
                  />
                )}
                {status === "pending" && (
                  <Circle
                    className="h-4 w-4 text-bark-300"
                    aria-label="Pending"
                  />
                )}
                {status === "skipped" && (
                  <SkipForward
                    className="h-4 w-4 text-bark-400"
                    aria-label="Skipped"
                  />
                )}
                {status === "abstained" && (
                  <AlertCircle
                    className="h-5 w-5 text-amber-500"
                    aria-label="Abstained"
                  />
                )}
              </span>

              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span
                    className={`text-sm font-semibold ${
                      status === "active"
                        ? "text-leaf-700 font-bold"
                        : status === "skipped"
                          ? "text-bark-400 line-through"
                          : "text-bark-900"
                    }`}
                  >
                    {idx + 1}. {stage.label}
                  </span>
                  {status === "skipped" && (
                    <span className="text-[10px] uppercase tracking-wider font-mono font-semibold px-1.5 py-0.5 rounded bg-bark-100 text-bark-600">
                      Skipped (Abstained)
                    </span>
                  )}
                  {status === "abstained" && (
                    <span className="text-[10px] uppercase tracking-wider font-mono font-semibold px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-800">
                      Gate Abstained
                    </span>
                  )}
                </div>
                <p className="text-xs text-bark-500 mt-0.5">
                  {stage.description}
                </p>
              </div>
            </li>
          );
        })}
      </ol>

      {safetyStatus === "species_uncertain" && (
        <div className="mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-900">
          <strong>Processing Halted at Safety Gate 1:</strong> Species
          confidence fell below the 90% threshold. Subsequent specialist
          inference was skipped to prevent misdiagnosis.
        </div>
      )}
    </div>
  );
};
