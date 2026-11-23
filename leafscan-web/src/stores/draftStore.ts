import { create } from "zustand";

export interface DraftState {
  file: File | null;
  previewUrl: string | null;
  thumbnailUrl: string | null;
  setDraft: (file: File, previewUrl: string, thumbnailUrl?: string) => void;
  clearDraft: () => void;
}

export const useDraftStore = create<DraftState>((set, get) => ({
  file: null,
  previewUrl: null,
  thumbnailUrl: null,
  setDraft: (file, previewUrl, thumbnailUrl) => {
    // Revoke previous URL if any to prevent memory leaks
    const current = get().previewUrl;
    if (current && current.startsWith("blob:") && current !== previewUrl) {
      if (
        typeof URL !== "undefined" &&
        typeof URL.revokeObjectURL === "function"
      ) {
        URL.revokeObjectURL(current);
      }
    }
    set({ file, previewUrl, thumbnailUrl: thumbnailUrl || previewUrl });
  },
  clearDraft: () => {
    const current = get().previewUrl;
    if (current && current.startsWith("blob:")) {
      if (
        typeof URL !== "undefined" &&
        typeof URL.revokeObjectURL === "function"
      ) {
        URL.revokeObjectURL(current);
      }
    }
    set({ file: null, previewUrl: null, thumbnailUrl: null });
  },
}));
