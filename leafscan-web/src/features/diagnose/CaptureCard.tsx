"use client";

import {
  AlertCircle,
  Camera,
  CheckCircle,
  Image as ImageIcon,
  RefreshCw,
  Upload,
} from "lucide-react";
import Image from "next/image";
import type React from "react";
import { useRef, useState } from "react";

export interface CaptureCardProps {
  onFileSelect: (file: File) => void;
  previewUrl: string | null;
  onClear: () => void;
  isLoading: boolean;
  onDiagnose: () => void;
  error?: string | null;
  isOfflineQueued?: boolean;
}

export const CaptureCard: React.FC<CaptureCardProps> = ({
  onFileSelect,
  previewUrl,
  onClear,
  isLoading,
  onDiagnose,
  error,
  isOfflineQueued,
}) => {
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
    // reset input value so re-selecting same file works
    e.target.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <section
      aria-label="Leaf Capture and Intake"
      className="w-full rounded-2xl bg-white border border-bark-200/80 p-6 shadow-sm transition-all"
    >
      {/* Hidden file inputs */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        capture="environment"
        className="hidden"
        onChange={handleFileChange}
        data-testid="camera-input"
        aria-label="Camera capture input"
      />
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={handleFileChange}
        data-testid="gallery-input"
        aria-label="Gallery upload input"
      />

      {/* Main Upload / Preview Area */}
      {!previewUrl ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          className={`relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-colors min-h-[300px] ${
            isDragOver
              ? "border-leaf-600 bg-leaf-50/50"
              : "border-bark-200 hover:border-leaf-500 bg-bark-50/30"
          }`}
        >
          <div className="rounded-full bg-leaf-100 p-4 text-leaf-700 mb-4">
            <ImageIcon className="h-8 w-8" aria-hidden="true" />
          </div>

          <h3 className="text-lg font-serif font-bold text-bark-900">
            Submit Leaf Specimen
          </h3>
          <p className="mt-1 max-w-sm text-xs text-bark-500">
            Upload or capture a single potato or tomato leaf with visible
            pathology symptoms. Supported: JPEG, PNG, WEBP (Max 10 MB, 40 MP).
          </p>

          <div className="mt-6 flex flex-wrap gap-3 justify-center">
            <button
              type="button"
              onClick={() => cameraInputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-5 py-3 text-sm font-semibold text-white shadow-sm hover:bg-leaf-700 active:scale-95 transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-leaf-600 min-h-[44px]"
            >
              <Camera className="h-5 w-5" aria-hidden="true" />
              Capture with Camera
            </button>

            <button
              type="button"
              onClick={() => galleryInputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-xl border border-bark-300 bg-white px-5 py-3 text-sm font-semibold text-bark-900 shadow-xs hover:bg-bark-50 active:scale-95 transition-all focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-leaf-600 min-h-[44px]"
            >
              <Upload className="h-5 w-5 text-bark-600" aria-hidden="true" />
              Browse Device
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <div className="relative w-full max-w-md aspect-square rounded-xl overflow-hidden bg-bark-900 shadow-md border border-bark-200">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Leaf specimen ready for diagnostic evaluation"
              className="h-full w-full object-contain"
            />
            <button
              type="button"
              onClick={onClear}
              disabled={isLoading}
              className="absolute top-3 right-3 rounded-full bg-bark-900/80 text-white p-2 hover:bg-bark-900 focus-visible:outline-2 focus-visible:outline-white disabled:opacity-50 transition-colors"
              aria-label="Remove image and select another"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>

          <div className="mt-6 flex w-full max-w-md flex-col gap-3">
            <button
              type="button"
              onClick={onDiagnose}
              disabled={isLoading}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-leaf-600 px-6 py-3.5 text-base font-semibold text-white shadow-float hover:bg-leaf-700 active:scale-98 disabled:opacity-60 disabled:cursor-not-allowed transition-all min-h-[48px]"
            >
              {isLoading ? (
                <>
                  <RefreshCw
                    className="h-5 w-5 animate-spin"
                    aria-hidden="true"
                  />
                  Running Foliar Screening...
                </>
              ) : (
                "Execute Diagnosis"
              )}
            </button>

            <button
              type="button"
              onClick={onClear}
              disabled={isLoading}
              className="text-xs text-bark-500 hover:text-bark-800 transition-colors py-1 text-center"
            >
              Discard and choose another image
            </button>
          </div>
        </div>
      )}

      {/* Error Announcement */}
      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="mt-4 flex items-center gap-2.5 rounded-lg border border-rust-600/30 bg-rust-50 p-3 text-xs font-medium text-rust-800"
        >
          <AlertCircle className="h-4 w-4 shrink-0 text-rust-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Offline Queued Announcement */}
      {isOfflineQueued && (
        <div
          role="status"
          aria-live="polite"
          className="mt-4 flex items-center gap-2.5 rounded-lg border border-amber-500/30 bg-amber-50 p-3 text-xs font-medium text-amber-900"
        >
          <CheckCircle className="h-4 w-4 shrink-0 text-amber-600" />
          <span>
            You are currently offline. Submission has been queued in local
            storage and will sync automatically when connectivity returns.
          </span>
        </div>
      )}

      {/* Field Photography Guidelines */}
      <div className="mt-8 border-t border-bark-100 pt-5">
        <h4 className="text-xs font-mono font-bold uppercase tracking-wider text-bark-500 mb-3">
          Field Photography Best Practices
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-bark-600">
          <div className="rounded-lg bg-bark-50 p-2.5">
            <strong>Single Leaf:</strong> Frame one leaf against a plain
            background.
          </div>
          <div className="rounded-lg bg-bark-50 p-2.5">
            <strong>Natural Light:</strong> Avoid strong flashlight glare and
            heavy shadows.
          </div>
          <div className="rounded-lg bg-bark-50 p-2.5">
            <strong>Sharp Focus:</strong> Ensure veins and lesion margins are
            clear.
          </div>
          <div className="rounded-lg bg-bark-50 p-2.5">
            <strong>Target Crops:</strong> Optimized exclusively for Potato and
            Tomato.
          </div>
        </div>
      </div>
    </section>
  );
};
