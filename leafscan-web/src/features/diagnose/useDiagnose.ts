"use client";

import {
  queueOfflineSubmission,
  saveDiagnosis,
} from "@/features/history/historyStore";
import { diagnoseLeaf } from "@/lib/api/diagnoseClient";
import type { DiagnosisResponse } from "@/lib/api/schemas";
import { useDraftStore } from "@/stores/draftStore";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { optimizeImageForUpload, prevalidateImage } from "./prevalidate";

export function useDiagnose() {
  const router = useRouter();
  const { file, previewUrl, setDraft, clearDraft } = useDraftStore();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryAttempt, setRetryAttempt] = useState(0);
  const [retryMessage, setRetryMessage] = useState<string | undefined>();
  const [isOfflineQueued, setIsOfflineQueued] = useState(false);

  const handleSelectFile = async (selectedFile: File) => {
    setError(null);
    setIsOfflineQueued(false);

    const validation = await prevalidateImage(selectedFile);
    if (!validation.valid) {
      setError(validation.error || "Invalid file");
      return false;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setDraft(selectedFile, objectUrl);
    return true;
  };

  const handleDiagnose = async () => {
    if (!file) {
      setError("Please select or capture a leaf photograph first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setRetryAttempt(0);
    setRetryMessage(undefined);

    // If device is offline, queue submission
    if (typeof navigator !== "undefined" && !navigator.onLine) {
      try {
        const optimizedBlob = await optimizeImageForUpload(file);
        await queueOfflineSubmission(optimizedBlob, previewUrl || "");
        setIsLoading(false);
        setIsOfflineQueued(true);
        return;
      } catch (err: any) {
        setError(`Failed to queue offline diagnosis: ${err.message}`);
        setIsLoading(false);
        return;
      }
    }

    try {
      const optimizedBlob = await optimizeImageForUpload(file);
      const response: DiagnosisResponse = await diagnoseLeaf(optimizedBlob, {
        onRetry: (attempt, delay) => {
          setRetryAttempt(attempt);
          setRetryMessage(
            `Service busy. Retrying in ${(delay / 1000).toFixed(1)}s (Attempt ${attempt}/3)...`,
          );
        },
      });

      // Save to IndexedDB ledger
      const stored = await saveDiagnosis(response, previewUrl || undefined);
      setIsLoading(false);

      // Navigate to result
      const reqId = stored.requestId;
      router.push(`/diagnose/result?id=${encodeURIComponent(reqId)}`);
    } catch (err: any) {
      setIsLoading(false);
      setError(
        err.message ||
          "Diagnostic request failed. Please check network and retry.",
      );
    }
  };

  const handleReset = () => {
    clearDraft();
    setError(null);
    setIsOfflineQueued(false);
    setRetryAttempt(0);
  };

  return {
    file,
    previewUrl,
    isLoading,
    error,
    retryAttempt,
    retryMessage,
    isOfflineQueued,
    handleSelectFile,
    handleDiagnose,
    handleReset,
  };
}
