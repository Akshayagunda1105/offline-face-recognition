import assert from "node:assert/strict";
import test from "node:test";

import { inspectGalleryFiles, identityLabelFromFilename, MAX_GALLERY_IMAGES } from "../src/gallery/files.js";

const file = (name) => ({ name });

test("accepts supported image extensions and derives labels from basenames", () => {
  const result = inspectGalleryFiles([
    file("Ada.jpg"), file("Grace.JPEG"), file("Lin.png"), file("Katherine.webp"),
  ]);
  assert.equal(result.files.length, 4);
  assert.deepEqual(result.unsupportedFiles, []);
  assert.deepEqual(result.invalidFilenames, []);
  assert.deepEqual(result.duplicateIdentities, []);
  assert.equal(identityLabelFromFilename("folder\\Ada Lovelace.jpg"), "Ada Lovelace");
});

test("reports an empty folder selection", () => {
  const result = inspectGalleryFiles([]);
  assert.equal(result.selectedCount, 0);
  assert.equal(result.files.length, 0);
});

test("excludes unsupported file types for an actionable UI message", () => {
  const result = inspectGalleryFiles([file("Photo.gif"), file("notes.txt"), file("Person.png")]);
  assert.deepEqual(result.files.map((item) => item.name), ["Person.png"]);
  assert.deepEqual(result.unsupportedFiles, ["Photo.gif", "notes.txt"]);
});

test("detects duplicate labels case-insensitively and after compatibility normalization", () => {
  const result = inspectGalleryFiles([
    file("Akshaya.jpg"), file("akshaya.PNG"),
    file("Ａda.webp"), file("Ada.jpeg"),
    file("Straße.jpg"), file("STRASSE.png"),
  ]);
  assert.deepEqual(result.duplicateIdentities.map((entry) => entry.filenames), [
    ["Akshaya.jpg", "akshaya.PNG"],
    ["Ａda.webp", "Ada.jpeg"],
    ["Straße.jpg", "STRASSE.png"],
  ]);
});

test("enforces the 50-image boundary", () => {
  const atLimit = inspectGalleryFiles(Array.from({ length: MAX_GALLERY_IMAGES }, (_, index) => file(`Person-${index}.jpg`)));
  const overLimit = inspectGalleryFiles(Array.from({ length: MAX_GALLERY_IMAGES + 1 }, (_, index) => file(`Person-${index}.jpg`)));
  assert.equal(atLimit.exceedsLimit, false);
  assert.equal(overLimit.exceedsLimit, true);
});

test("rejects supported extensions without a usable filename stem", () => {
  const result = inspectGalleryFiles([file(".jpg")]);
  assert.deepEqual(result.invalidFilenames, [".jpg"]);
});
