"use client";

import { Check, X } from "lucide-react";
import type React from "react";

export interface ConformalSetProps {
  allClasses?: string[];
  includedClasses: string[];
  coverageLevel?: number; // e.g. 0.90 for 90% coverage
  className?: string;
}

const DEFAULT_SOLANACEOUS_CLASSES = ["Early Blight", "Late Blight", "Healthy"];

export const ConformalSet: React.FC<ConformalSetProps> = ({
  allClasses = DEFAULT_SOLANACEOUS_CLASSES,
  includedClasses,
  coverageLevel = 0.9,
  className = "",
}) => {
  const coveragePercent = Math.round(coverageLevel * 100);

  return (
    <div
      role="region"
      aria-label="Conformal Prediction Set"
      className={`rounded-xl border border-bark-200 bg-white p-5 shadow-sm ${className}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-serif font-bold text-bark-900">
          Conformal Prediction Set
        </h4>
        <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-leaf-50 text-leaf-700 border border-leaf-200">
          {coveragePercent}% Theoretical Coverage
        </span>
      </div>

      <p className="text-xs text-bark-500 mb-4">
        Guarantees that the true pathology is contained within the highlighted
        candidates with {coveragePercent}% statistical confidence.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {allClasses.map((cls) => {
          const isIncluded = includedClasses.some(
            (inc) => inc.toLowerCase() === cls.toLowerCase(),
          );

          return (
            <div
              key={cls}
              className={`flex items-center justify-between p-3 rounded-lg border text-sm transition-colors ${
                isIncluded
                  ? "border-leaf-600/40 bg-leaf-50 text-leaf-900 font-semibold"
                  : "border-bark-200 bg-bark-50/50 text-bark-400 line-through"
              }`}
            >
              <span>{cls}</span>
              {isIncluded ? (
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-leaf-600 text-white">
                  <Check className="h-3.5 w-3.5" aria-label="Included in set" />
                </span>
              ) : (
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-bark-200 text-bark-500">
                  <X className="h-3 w-3" aria-label="Excluded from set" />
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
