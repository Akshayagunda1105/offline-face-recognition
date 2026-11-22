export const MAX_GALLERY_IMAGES = 50;
export const SUPPORTED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);

function basename(filename) {
  return String(filename).replaceAll("\\", "/").split("/").at(-1) || "";
}

export function identityLabelFromFilename(filename) {
  const name = basename(filename);
  const extensionIndex = name.lastIndexOf(".");
  if (extensionIndex <= 0) return "";
  return name.slice(0, extensionIndex).normalize("NFC").trim();
}

function identityKey(label) {
  // JavaScript has no direct casefold() equivalent; these mappings cover
  // Unicode folds that lowercase alone would otherwise miss.
  return label.normalize("NFKC").trim().toLocaleLowerCase("und")
    .replaceAll("ß", "ss")
    .replaceAll("ς", "σ");
}

export function inspectGalleryFiles(fileList) {
  const selected = Array.from(fileList || []);
  const files = [];
  const unsupportedFiles = [];
  const invalidFilenames = [];
  const firstByIdentity = new Map();
  const duplicatesByIdentity = new Map();

  for (const file of selected) {
    const name = basename(file.name);
    const extensionIndex = name.lastIndexOf(".");
    const extension = extensionIndex < 0 ? "" : name.slice(extensionIndex).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.has(extension)) {
      unsupportedFiles.push(file.name);
      continue;
    }

    files.push(file);
    const label = identityLabelFromFilename(file.name);
    const key = identityKey(label);
    if (!label || !key) {
      invalidFilenames.push(file.name);
      continue;
    }

    const previous = firstByIdentity.get(key);
    if (previous) {
      if (!duplicatesByIdentity.has(key)) {
        duplicatesByIdentity.set(key, { label: previous.label, filenames: [previous.filename] });
      }
      duplicatesByIdentity.get(key).filenames.push(file.name);
    } else {
      firstByIdentity.set(key, { label, filename: file.name });
    }
  }

  return {
    selectedCount: selected.length,
    files,
    unsupportedFiles,
    invalidFilenames,
    duplicateIdentities: Array.from(duplicatesByIdentity.values()),
    exceedsLimit: files.length > MAX_GALLERY_IMAGES,
  };
}
