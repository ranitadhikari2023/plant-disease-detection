import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface UserPreferences {
  locale: "en" | "hi" | "es";
  cameraFacingMode: "environment" | "user";
  lowBandwidthMode: boolean;
  setLocale: (locale: "en" | "hi" | "es") => void;
  setCameraFacingMode: (mode: "environment" | "user") => void;
  setLowBandwidthMode: (enabled: boolean) => void;
}

export const usePrefsStore = create<UserPreferences>()(
  persist(
    (set) => ({
      locale: "en",
      cameraFacingMode: "environment",
      lowBandwidthMode: false,
      setLocale: (locale) => set({ locale }),
      setCameraFacingMode: (cameraFacingMode) => set({ cameraFacingMode }),
      setLowBandwidthMode: (lowBandwidthMode) => set({ lowBandwidthMode }),
    }),
    {
      name: "leafscan_preferences",
    },
  ),
);
