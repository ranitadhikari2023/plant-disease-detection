"use client";

import { telemetry } from "@/lib/platform/telemetry";
import { AlertCircle, RefreshCw } from "lucide-react";
import Link from "next/link";
import React, { useEffect } from "react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    telemetry.recordError(error, { digest: error.digest });
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4">
      <div className="rounded-full bg-rust-600/10 p-4 text-rust-600 mb-4">
        <AlertCircle className="h-10 w-10" />
      </div>
      <h2 className="text-2xl font-serif font-bold text-bark-900">
        Application Error
      </h2>
      <p className="mt-2 max-w-md text-sm text-bark-600">
        {error.message ||
          "An unexpected error occurred while rendering the interface."}
      </p>
      {error.digest && (
        <span className="mt-2 text-xs font-mono text-bark-400">
          Error Digest: {error.digest}
        </span>
      )}
      <div className="mt-6 flex gap-3">
        <button
          type="button"
          onClick={() => reset()}
          className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-leaf-700"
        >
          <RefreshCw className="h-4 w-4" />
          Try Again
        </button>
        <Link
          href="/diagnose"
          className="inline-flex items-center rounded-xl border border-bark-300 bg-white px-5 py-2.5 text-sm font-semibold text-bark-900 shadow-xs hover:bg-bark-50"
        >
          Return to Scanner
        </Link>
      </div>
    </div>
  );
}
