export const MAX_GROUP_PHOTO_BYTES = 20 * 1024 * 1024;

const MIME_BY_EXTENSION = new Map([
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".png", "image/png"],
  [".webp", "image/webp"],
]);

export function inspectGroupPhoto(file) {
  if (!file) return { valid: false, code: "photo_required" };
  const name = String(file.name || "").replaceAll("\\", "/").split("/").at(-1) || "";
  const dot = name.lastIndexOf(".");
  const extension = dot < 0 ? "" : name.slice(dot).toLowerCase();
  const expectedMime = MIME_BY_EXTENSION.get(extension);
  if (!expectedMime || (file.type && file.type !== expectedMime)) {
    return { valid: false, code: "invalid_group_photo_type" };
  }
  if (!Number.isFinite(file.size) || file.size <= 0) {
    return { valid: false, code: "empty_group_photo" };
  }
  if (file.size > MAX_GROUP_PHOTO_BYTES) {
    return { valid: false, code: "group_photo_too_large" };
  }
  return { valid: true, code: null };
}

export function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "Unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
