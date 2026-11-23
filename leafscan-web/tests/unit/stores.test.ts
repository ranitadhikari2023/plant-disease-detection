import { useDraftStore } from "@/stores/draftStore";
import { usePrefsStore } from "@/stores/prefsStore";
import { describe, expect, it } from "vitest";

describe("State Stores", () => {
  it("manages draftStore file and previewUrl lifecycles", () => {
    const dummyFile = new File(["test content"], "leaf.jpg", {
      type: "image/jpeg",
    });
    const { setDraft, clearDraft } = useDraftStore.getState();

    setDraft(
      dummyFile,
      "blob:http://localhost/test",
      "blob:http://localhost/thumb",
    );
    expect(useDraftStore.getState().file).toBe(dummyFile);
    expect(useDraftStore.getState().previewUrl).toBe(
      "blob:http://localhost/test",
    );

    clearDraft();
    expect(useDraftStore.getState().file).toBeNull();
    expect(useDraftStore.getState().previewUrl).toBeNull();
  });

  it("manages prefsStore preferences", () => {
    const { setLocale, setCameraFacingMode, setLowBandwidthMode } =
      usePrefsStore.getState();

    setLocale("hi");
    expect(usePrefsStore.getState().locale).toBe("hi");

    setCameraFacingMode("user");
    expect(usePrefsStore.getState().cameraFacingMode).toBe("user");

    setLowBandwidthMode(true);
    expect(usePrefsStore.getState().lowBandwidthMode).toBe(true);
  });
});
