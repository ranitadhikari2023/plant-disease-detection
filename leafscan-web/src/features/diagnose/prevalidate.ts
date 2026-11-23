export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10MB
export const MAX_PIXELS = 40_000_000; // 40 MP
export const MAX_LONGEST_EDGE = 1600; // 1600 px
export const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

export interface PrevalidationResult {
  valid: boolean;
  error?: string;
  width?: number;
  height?: number;
}

export async function prevalidateImage(
  file: File,
): Promise<PrevalidationResult> {
  if (!file) {
    return { valid: false, error: "No file provided" };
  }

  // 1. Check size limit
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      valid: false,
      error: `File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 10 MB limit.`,
    };
  }

  if (file.size === 0) {
    return { valid: false, error: "File is empty (0 bytes)." };
  }

  // 2. Check MIME type
  const mime = file.type.toLowerCase();
  const filename = file.name.toLowerCase();
  const isExtensionValid =
    filename.endsWith(".jpg") ||
    filename.endsWith(".jpeg") ||
    filename.endsWith(".png") ||
    filename.endsWith(".webp");

  if (!ALLOWED_MIME_TYPES.includes(mime) && !isExtensionValid) {
    return {
      valid: false,
      error:
        "Unsupported file format. Please upload a JPEG, PNG, or WEBP image.",
    };
  }

  // 3. Check pixel count via decoding
  try {
    let width = 0;
    let height = 0;

    if (typeof createImageBitmap === "function") {
      try {
        const bitmap = await createImageBitmap(file, {
          imageOrientation: "from-image",
        });
        width = bitmap.width;
        height = bitmap.height;
        bitmap.close();
      } catch {
        // Fallback below
      }
    } else if (
      typeof Image !== "undefined" &&
      typeof URL !== "undefined" &&
      typeof URL.createObjectURL === "function"
    ) {
      try {
        await Promise.race([
          new Promise<void>((resolve) => {
            const img = new Image();
            const url = URL.createObjectURL(file);
            img.onload = () => {
              width = img.naturalWidth || img.width;
              height = img.naturalHeight || img.height;
              URL.revokeObjectURL(url);
              resolve();
            };
            img.onerror = () => {
              URL.revokeObjectURL(url);
              resolve();
            };
            img.src = url;
          }),
          new Promise<void>((resolve) => setTimeout(resolve, 50)),
        ]);
      } catch {
        // Ignore fallback errors
      }
    }

    if (width > 0 && height > 0) {
      const totalPixels = width * height;
      if (totalPixels > MAX_PIXELS) {
        return {
          valid: false,
          error: `Image dimensions (${width}x${height} = ${(totalPixels / 1e6).toFixed(1)} MP) exceed the 40 MP maximum.`,
          width,
          height,
        };
      }
      return { valid: true, width, height };
    }

    return { valid: true };
  } catch (err: any) {
    return {
      valid: false,
      error:
        err.message ||
        "Failed to decode image raster. Please ensure the file is not corrupted.",
    };
  }
}

export async function optimizeImageForUpload(file: File): Promise<Blob> {
  // If running in an environment without DOM canvas, return file directly
  if (
    typeof document === "undefined" ||
    typeof HTMLCanvasElement === "undefined"
  ) {
    return file;
  }

  try {
    let width = 0;
    let height = 0;
    let imageSource: CanvasImageSource | null = null;
    let cleanup: (() => void) | null = null;

    if (typeof createImageBitmap === "function") {
      try {
        const bitmap = await createImageBitmap(file, {
          imageOrientation: "from-image",
        });
        width = bitmap.width;
        height = bitmap.height;
        imageSource = bitmap;
        cleanup = () => bitmap.close();
      } catch {
        // Fallback to Image
      }
    }

    if (!imageSource) {
      await new Promise<void>((resolve, reject) => {
        const img = new Image();
        const url = URL.createObjectURL(file);
        img.onload = () => {
          width = img.naturalWidth;
          height = img.naturalHeight;
          imageSource = img;
          cleanup = () => URL.revokeObjectURL(url);
          resolve();
        };
        img.onerror = (e) => {
          URL.revokeObjectURL(url);
          reject(e);
        };
        img.src = url;
      });
    }

    if (!imageSource || width === 0 || height === 0) {
      return file;
    }

    // Downscale if longest edge > 1600
    let targetWidth = width;
    let targetHeight = height;
    const longestEdge = Math.max(width, height);

    if (longestEdge > MAX_LONGEST_EDGE) {
      const scale = MAX_LONGEST_EDGE / longestEdge;
      targetWidth = Math.round(width * scale);
      targetHeight = Math.round(height * scale);
    }

    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      cleanup?.();
      return file;
    }

    ctx.drawImage(imageSource, 0, 0, targetWidth, targetHeight);
    cleanup?.();

    return new Promise<Blob>((resolve) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob);
          } else {
            resolve(file);
          }
        },
        "image/jpeg",
        0.85,
      );
    });
  } catch (err) {
    console.warn(
      "Client-side image optimization fallback to original file:",
      err,
    );
    return file;
  }
}
