"use client";

import { ShieldCheck } from "lucide-react";
import Link from "next/link";
import type React from "react";

export const Footer: React.FC = () => {
  return (
    <footer className="mt-auto border-t border-bark-200/60 bg-bark-100/40 py-8 px-4 sm:px-6">
      <div className="mx-auto max-w-7xl">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-leaf-600" />
              <span className="text-xs font-bold font-serif text-bark-900">
                LeafScan V3 Agricultural Decision Support
              </span>
            </div>
            <p className="mt-1 max-w-2xl text-[11px] text-bark-500 leading-relaxed">
              Automated screening assistance for educational and
              decision-support purposes only. Does not replace professional
              on-site plant pathology diagnosis. Always consult certified
              agricultural extension officers and follow registered pesticide
              label directions.
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-bark-500">
            <Link
              href="/model"
              className="hover:text-leaf-700 transition-colors"
            >
              Model Specs
            </Link>
            <span>&bull;</span>
            <Link
              href="/feedback"
              className="hover:text-leaf-700 transition-colors"
            >
              Field Feedback
            </Link>
            <span>&bull;</span>
            <span>Version 3.0.0</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
