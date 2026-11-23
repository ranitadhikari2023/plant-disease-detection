"use client";

import {
  type OfflineSubmission,
  type StoredDiagnosis,
  clearAllDiagnoses,
  flushOfflineQueue,
  getAllDiagnoses,
  getOfflineQueue,
} from "@/features/history/historyStore";
import {
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  Calendar,
  Filter,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  WifiOff,
} from "lucide-react";
import Link from "next/link";
import React, { useEffect, useState } from "react";

export default function HistoryPage() {
  const [diagnoses, setDiagnoses] = useState<StoredDiagnosis[]>([]);
  const [offlineQueue, setOfflineQueue] = useState<OfflineSubmission[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<
    "all" | "confident" | "uncertain"
  >("all");
  const [isSyncing, setIsSyncing] = useState(false);

  const loadData = async () => {
    const [diagList, queueList] = await Promise.all([
      getAllDiagnoses(),
      getOfflineQueue(),
    ]);
    setDiagnoses(diagList);
    setOfflineQueue(queueList);
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleClearHistory = async () => {
    if (
      window.confirm(
        "Are you sure you want to permanently clear all stored scan records?",
      )
    ) {
      await clearAllDiagnoses();
      loadData();
    }
  };

  const handleSyncQueue = async () => {
    setIsSyncing(true);
    await flushOfflineQueue();
    await loadData();
    setIsSyncing(false);
  };

  // Filter diagnoses
  const filteredDiagnoses = diagnoses.filter((d) => {
    const matchesSearch =
      (d.plantSpecies || "")
        .toLowerCase()
        .includes(searchQuery.toLowerCase()) ||
      (d.disease || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.requestId.toLowerCase().includes(searchQuery.toLowerCase());

    if (!matchesSearch) return false;

    if (statusFilter === "confident") {
      return d.safetyStatus === "confident";
    }
    if (statusFilter === "uncertain") {
      return d.safetyStatus !== "confident";
    }
    return true;
  });

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-bark-200">
        <div>
          <h1 className="text-3xl font-serif font-extrabold text-bark-900">
            Field Diagnosis Ledger
          </h1>
          <p className="mt-1 text-sm text-bark-600">
            Local on-device records stored securely in IndexedDB.
          </p>
        </div>

        {diagnoses.length > 0 && (
          <button
            type="button"
            onClick={handleClearHistory}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-rust-600/30 text-xs font-semibold text-rust-600 hover:bg-rust-50 transition-colors w-fit"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear Local Ledger
          </button>
        )}
      </div>

      {/* Offline Queue Section (if pending) */}
      {offlineQueue.length > 0 && (
        <div className="rounded-2xl border border-amber-500/40 bg-amber-500/10 p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <WifiOff className="h-4 w-4 text-amber-700" />
              <h2 className="text-sm font-serif font-bold text-amber-900">
                Pending Offline Submissions ({offlineQueue.length})
              </h2>
            </div>
            <button
              type="button"
              onClick={handleSyncQueue}
              disabled={isSyncing}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-600 text-xs font-semibold text-white shadow-xs hover:bg-amber-700 disabled:opacity-60"
            >
              <RefreshCw
                className={`h-3 w-3 ${isSyncing ? "animate-spin" : ""}`}
              />
              Sync Now
            </button>
          </div>
          <p className="text-xs text-bark-600 mb-3">
            These foliar images were captured while offline and are waiting to
            be screened by the LeafScan V3 inference backend.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {offlineQueue.map((item) => (
              <div
                key={item.id}
                className="relative aspect-square rounded-lg overflow-hidden bg-bark-800 border border-bark-200"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={item.thumbnail}
                  alt="Queued scan"
                  className="h-full w-full object-cover"
                />
                <span className="absolute bottom-1.5 left-1.5 text-[9px] font-mono px-1 py-0.5 rounded bg-bark-900/80 text-white">
                  Queued
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-bark-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by crop, disease, or request ID..."
            className="w-full rounded-xl border border-bark-200 bg-white py-2.5 pl-9 pr-4 text-xs text-bark-900 placeholder:text-bark-400 focus:border-leaf-600 focus:outline-none focus:ring-1 focus:ring-leaf-600 shadow-xs"
          />
        </div>

        <div className="flex gap-2">
          {(["all", "confident", "uncertain"] as const).map((filter) => (
            <button
              key={filter}
              type="button"
              onClick={() => setStatusFilter(filter)}
              className={`rounded-xl px-3.5 py-2 text-xs font-semibold capitalize transition-all ${
                statusFilter === filter
                  ? "bg-leaf-600 text-white shadow-xs"
                  : "bg-white border border-bark-200 text-bark-700 hover:bg-bark-50"
              }`}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>

      {/* Diagnoses List */}
      {filteredDiagnoses.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-bark-300 p-12 text-center bg-white/40">
          <p className="text-sm font-medium text-bark-600">
            {searchQuery
              ? "No matching diagnostic records found."
              : "No foliar scans recorded in local history yet."}
          </p>
          <div className="mt-4">
            <Link
              href="/diagnose"
              className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-4 py-2 text-xs font-semibold text-white shadow-xs hover:bg-leaf-700"
            >
              Scan a Leaf
            </Link>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredDiagnoses.map((d) => {
            const isConfident = d.safetyStatus === "confident";
            const isLateBlightUncertain =
              d.safetyStatus === "late_blight_uncertain";

            return (
              <div
                key={d.requestId}
                className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 rounded-xl border border-bark-200 bg-white p-4 shadow-xs hover:border-leaf-500/50 transition-colors"
              >
                <div className="flex items-center gap-3.5">
                  {/* Thumbnail */}
                  <div className="h-14 w-14 shrink-0 rounded-lg overflow-hidden bg-bark-900 border border-bark-200">
                    {d.thumbnail ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={d.thumbnail}
                        alt=""
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="h-full w-full bg-bark-100 flex items-center justify-center text-[10px] text-bark-400">
                        N/A
                      </div>
                    )}
                  </div>

                  {/* Classification details */}
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      {isConfident ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase tracking-wider text-teal-800 bg-teal-500/20 px-2 py-0.5 rounded">
                          <ShieldCheck className="h-3 w-3 text-teal-700" />
                          Confident
                        </span>
                      ) : isLateBlightUncertain ? (
                        <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase tracking-wider text-rust-800 bg-rust-600/20 px-2 py-0.5 rounded">
                          <AlertOctagon className="h-3 w-3 text-rust-600" />
                          Late Blight Alert
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold uppercase tracking-wider text-amber-800 bg-amber-500/20 px-2 py-0.5 rounded">
                          <AlertTriangle className="h-3 w-3 text-amber-700" />
                          Uncertain
                        </span>
                      )}
                      <span className="text-[11px] font-mono text-bark-400">
                        {new Date(d.timestamp).toLocaleDateString()}
                      </span>
                    </div>

                    <h3 className="text-base font-serif font-bold text-bark-900">
                      {d.diagnosis}
                    </h3>

                    <div className="flex items-center gap-3 text-xs font-mono text-bark-500 mt-0.5">
                      <span>Crop: {d.plantSpecies || "Unknown"}</span>
                      {d.disease && <span>Pathology: {d.disease}</span>}
                    </div>
                  </div>
                </div>

                <div className="flex sm:flex-col items-center sm:items-end justify-between gap-2 pt-2 sm:pt-0 border-t sm:border-t-0 border-bark-100">
                  <span className="text-[11px] font-mono text-bark-400">
                    ID: {d.requestId.slice(0, 8)}...
                  </span>
                  <Link
                    href={`/diagnose/result?id=${encodeURIComponent(d.requestId)}`}
                    className="inline-flex items-center gap-1.5 text-xs font-semibold text-leaf-700 hover:text-leaf-800 px-3 py-1.5 rounded-lg bg-leaf-50 hover:bg-leaf-100 transition-colors"
                  >
                    View Report
                    <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
