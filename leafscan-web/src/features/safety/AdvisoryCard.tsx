"use client";

import {
  AlertCircle,
  AlertTriangle,
  ExternalLink,
  Info,
  PhoneCall,
  ShieldAlert,
} from "lucide-react";
import type React from "react";

export interface AdvisoryAction {
  label: string;
  url?: string;
  tel?: string;
}

export interface AdvisoryCardProps {
  code?: string;
  severity?: "info" | "caution" | "warning" | "critical";
  title?: string;
  body: string;
  actions?: AdvisoryAction[];
  className?: string;
}

export const AdvisoryCard: React.FC<AdvisoryCardProps> = ({
  code,
  severity = "info",
  title,
  body,
  actions = [],
  className = "",
}) => {
  // Chemical suppression safety check: Section 11.3
  // Must never generate recipes, dosages, or quantities.
  const isChemicalSuppression =
    code === "chemical_suppression" ||
    body.toLowerCase().includes("chemical") ||
    body.toLowerCase().includes("fungicide");

  const effectiveSeverity =
    isChemicalSuppression && severity === "info" ? "warning" : severity;

  const severityConfigs = {
    info: {
      border: "border-leaf-200",
      bg: "bg-leaf-50",
      iconColor: "text-leaf-700",
      badgeColor: "bg-leaf-200 text-leaf-800",
      icon: Info,
      defaultTitle: "Pathology Management Guidance",
    },
    caution: {
      border: "border-amber-300",
      bg: "bg-amber-50/80",
      iconColor: "text-amber-700",
      badgeColor: "bg-amber-200 text-amber-800",
      icon: AlertTriangle,
      defaultTitle: "Cautionary Agronomic Note",
    },
    warning: {
      border: "border-amber-400",
      bg: "bg-amber-50",
      iconColor: "text-amber-700",
      badgeColor: "bg-amber-300 text-amber-900",
      icon: AlertCircle,
      defaultTitle: "Important Safety Advisory",
    },
    critical: {
      border: "border-rust-600/30",
      bg: "bg-rust-50",
      iconColor: "text-rust-600",
      badgeColor: "bg-rust-600 text-white",
      icon: ShieldAlert,
      defaultTitle: "Critical Intervention Protocol",
    },
  };

  const config = severityConfigs[effectiveSeverity] || severityConfigs.info;
  const IconComponent = config.icon;

  return (
    <article
      aria-label="Agronomic Advisory"
      className={`rounded-xl border ${config.border} ${config.bg} p-5 shadow-sm ${className}`}
    >
      <div className="flex items-start gap-3.5">
        <div
          className={`rounded-full p-2 ${config.iconColor} bg-white shadow-xs`}
        >
          <IconComponent className="h-5 w-5" aria-hidden="true" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`text-[10px] font-mono font-bold uppercase tracking-wider px-2 py-0.5 rounded ${config.badgeColor}`}
            >
              {effectiveSeverity}
            </span>
            {code && (
              <span className="text-[11px] font-mono text-bark-500">
                Code: {code}
              </span>
            )}
          </div>

          <h3 className="text-base font-serif font-bold text-bark-900">
            {title || config.defaultTitle}
          </h3>

          <div className="mt-2 text-sm leading-relaxed text-bark-800 whitespace-pre-line">
            {body}
          </div>

          {/* Chemical suppression explicit disclaimer */}
          {isChemicalSuppression && (
            <div className="mt-3 rounded-lg border border-rust-600/20 bg-white/80 p-3 text-xs text-rust-800">
              <strong className="block font-semibold mb-1">
                Pesticide Safety Warning:
              </strong>
              This application does not provide dosage recipes, chemical
              mixtures, or spray schedules. Always consult a certified local
              agricultural extension officer and strictly adhere to registered
              label instructions.
            </div>
          )}

          {/* Action buttons / contact referral links */}
          {actions.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2.5 pt-2 border-t border-bark-200/50">
              {actions.map((act, idx) => (
                <span key={idx}>
                  {act.tel ? (
                    <a
                      href={`tel:${act.tel}`}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-bark-300 text-xs font-semibold text-bark-900 hover:bg-bark-50"
                    >
                      <PhoneCall className="h-3.5 w-3.5 text-leaf-600" />
                      {act.label} ({act.tel})
                    </a>
                  ) : act.url ? (
                    <a
                      href={act.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-bark-300 text-xs font-semibold text-bark-900 hover:bg-bark-50"
                    >
                      <ExternalLink className="h-3.5 w-3.5 text-leaf-600" />
                      {act.label}
                    </a>
                  ) : (
                    <span className="inline-flex items-center px-3 py-1.5 rounded-lg bg-white border border-bark-300 text-xs font-medium text-bark-800">
                      {act.label}
                    </span>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </article>
  );
};
