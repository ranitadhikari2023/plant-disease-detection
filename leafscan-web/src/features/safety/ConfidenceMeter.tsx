"use client";

import type React from "react";

export interface ConfidenceMeterProps {
  label: string;
  value: number; // 0 to 1
  threshold?: number; // e.g. 0.90 or 0.70
  className?: string;
  testId?: string;
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  label,
  value,
  threshold,
  className = "",
  testId = "confidence-meter",
}) => {
  const percentage = Math.min(100, Math.max(0, Math.round(value * 100)));
  const thresholdPercentage = threshold ? Math.round(threshold * 100) : null;
  const isAboveThreshold = threshold ? value >= threshold : true;

  // Meter styling tokens
  const barColor = isAboveThreshold ? "bg-teal-500" : "bg-amber-500";
  const textColor = isAboveThreshold ? "text-teal-700" : "text-amber-700";

  return (
    <div className={`w-full ${className}`} data-testid={testId}>
      <div className="flex justify-between items-baseline mb-1.5">
        <span className="text-sm font-medium text-bark-900">{label}</span>
        <div className="flex items-center gap-2">
          {thresholdPercentage && (
            <span className="text-xs text-bark-500 font-mono">
              Target: {thresholdPercentage}%
            </span>
          )}
          <span
            className={`text-sm font-mono font-bold tabular-nums ${textColor}`}
            aria-hidden="true"
          >
            {percentage}%
          </span>
        </div>
      </div>

      <div
        role="meter"
        aria-label={`${label} confidence`}
        aria-valuenow={percentage}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuetext={`${percentage} percent confident`}
        className="relative h-3 w-full overflow-hidden rounded-full bg-bark-100 border border-bark-200"
      >
        <div
          className={`h-full ${barColor} transition-all duration-600 ease-out`}
          style={{ width: `${percentage}%` }}
        />
        {thresholdPercentage && (
          <div
            title={`Threshold: ${thresholdPercentage}%`}
            style={{ left: `${thresholdPercentage}%` }}
            className="absolute top-0 bottom-0 w-0.5 bg-bark-900/60 z-10"
          />
        )}
      </div>

      <div className="mt-1 flex justify-between text-[11px] text-bark-400 font-mono">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
