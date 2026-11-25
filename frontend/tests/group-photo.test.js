import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "vite";
import { renderToStaticMarkup } from "react-dom/server";

import { MAX_GROUP_PHOTO_BYTES, inspectGroupPhoto } from "../src/group-photo/file.js";
import { knownUnknownFixture, zeroFacesFixture } from "./fixtures/recognition-responses.js";

let vite;
let api;
let decodeRecognitionResponse;
let RecognitionResults;
let friendlyError;
let GroupPhotoWorkflow;
let ExperimentalNotice;
let galleryFileFailures;
let GalleryFileFailure;

before(async () => {
  vite = await createServer({ server: { middlewareMode: true }, appType: "custom" });
  api = await vite.ssrLoadModule("/src/api/client.js");
  ({ decodeRecognitionResponse } = await vite.ssrLoadModule("/src/recognition/response.js"));
  ({ default: RecognitionResults } = await vite.ssrLoadModule("/src/recognition/RecognitionResults.jsx"));
  ({ friendlyError } = await vite.ssrLoadModule("/src/api/errors.js"));
  ({ default: GroupPhotoWorkflow } = await vite.ssrLoadModule("/src/recognition/GroupPhotoWorkflow.jsx"));
  ({ default: ExperimentalNotice } = await vite.ssrLoadModule("/src/recognition/ExperimentalNotice.jsx"));
  ({ galleryFileFailures } = await vite.ssrLoadModule("/src/api/errors.js"));
  ({ default: GalleryFileFailure } = await vite.ssrLoadModule("/src/gallery/GalleryFileFailure.jsx"));
});

after(async () => { await vite?.close(); });

test("validates supported group-photo types and enforces the exact 20 MiB limit", () => {
  for (const [name, type] of [["a.jpg", "image/jpeg"], ["b.jpeg", "image/jpeg"], ["c.png", "image/png"], ["d.webp", "image/webp"]]) {
    assert.deepEqual(inspectGroupPhoto({ name, type, size: 100 }), { valid: true, code: null });
  }
  assert.equal(inspectGroupPhoto({ name: "max.jpg", type: "image/jpeg", size: MAX_GROUP_PHOTO_BYTES }).valid, true);
  assert.equal(inspectGroupPhoto({ name: "bad.gif", type: "image/gif", size: 100 }).code, "invalid_group_photo_type");
  assert.equal(inspectGroupPhoto({ name: "bad.jpg", type: "image/png", size: 100 }).code, "invalid_group_photo_type");
  assert.equal(inspectGroupPhoto({ name: "large.jpg", type: "image/jpeg", size: MAX_GROUP_PHOTO_BYTES + 1 }).code, "group_photo_too_large");
  assert.equal(inspectGroupPhoto({ name: "empty.jpg", type: "image/jpeg", size: 0 }).code, "empty_group_photo");
});

test("recognition API posts photo multipart with the session Bearer token and browser-managed boundary", async (t) => {
  let captured;
  t.mock.method(globalThis, "fetch", async (url, init) => {
    captured = { url, init };
    return Response.json(knownUnknownFixture);
  });
  const file = new File(["photo bytes"], "Group.jpeg", { type: "image/jpeg" });
  const response = await api.galleryApi.recognizeGroupPhoto("test-session-secret", file);
  assert.equal(response.face_count, 2);
  assert.equal(captured.url, `${api.API_BASE_URL}/api/group-photos/recognize`);
  assert.equal(captured.init.method, "POST");
  assert.equal(captured.init.headers.Authorization, "Bearer test-session-secret");
  assert.equal(captured.init.headers["Content-Type"], undefined);
  assert.equal(captured.init.body.get("photo").name, "Group.jpeg");
});

test("surfaces readiness, expired-session, invalid-image, upload-size, and busy errors", async (t) => {
  const cases = [
    [503, "recognition_not_ready", null],
    [404, "gallery_session_unavailable", null],
    [422, "invalid_group_photo", null],
    [413, "group_photo_too_large", null],
    [503, "processing_busy", "2"],
    [413, "annotated_response_too_large", null],
  ];
  for (const [status, code, retryAfter] of cases) {
    t.mock.method(globalThis, "fetch", async () => Response.json(
      { detail: { code, message: `fixture ${code}` } },
      { status, headers: retryAfter ? { "Retry-After": retryAfter } : {} },
    ));
    await assert.rejects(
      api.galleryApi.recognizeGroupPhoto("session", new File(["x"], "group.jpg", { type: "image/jpeg" })),
      (error) => error instanceof api.ApiError && error.status === status && error.code === code &&
        error.retryAfterSeconds === (retryAfter ? Number(retryAfter) : null),
    );
    t.mock.restoreAll();
  }
});

test("maps backend failures to user-facing messages without exposing policy scores", async () => {
  const readiness = friendlyError(new api.ApiError("readiness", { status: 503, code: "recognition_not_ready" }));
  const expired = friendlyError(new api.ApiError("expired", { status: 404, code: "gallery_session_unavailable" }));
  const invalid = friendlyError(new api.ApiError("invalid", { status: 422, code: "invalid_group_photo" }));
  const large = friendlyError(new api.ApiError("large", { status: 413, code: "group_photo_too_large" }));
  const busy = friendlyError(new api.ApiError("busy", { status: 503, code: "processing_busy", retryAfterSeconds: 2 }));
  assert.match(readiness, /unavailable.*uncalibrated/iu);
  assert.match(expired, /expired.*backend restart/iu);
  assert.match(invalid, /corrupted|extension/iu);
  assert.match(large, /20 MiB/iu);
  assert.match(busy, /Retry after 2 seconds/iu);
  for (const message of [readiness, expired, invalid, large, busy]) {
    assert.doesNotMatch(message, /similarity|confidence|margin/iu);
  }
});

test("decodes and renders Known and Unknown rows in backend order without scores", () => {
  const result = decodeRecognitionResponse(knownUnknownFixture);
  const markup = renderToStaticMarkup(RecognitionResults({ result, imageUrl: "blob:test-image" }));
  assert.match(markup, /Fixture Person/);
  assert.match(markup, />Unknown</);
  assert.ok(markup.indexOf("Fixture Person") < markup.indexOf("Unknown"));
  assert.doesNotMatch(markup, /similarity|confidence|margin|0\.9/iu);
  assert.equal(result.annotatedBlob.type, "image/jpeg");
});

test("renders the explicit zero-face state", () => {
  const result = decodeRecognitionResponse(zeroFacesFixture);
  const markup = renderToStaticMarkup(RecognitionResults({ result, imageUrl: "blob:test-image" }));
  assert.match(markup, /No faces detected in this photo\./);
  assert.match(markup, /<strong>0<\/strong> faces/);
});

test("experimental status is warned about and gates the recognition action", () => {
  const warning = renderToStaticMarkup(ExperimentalNotice({
    check: { state: "loaded", data: { experimental: true, calibrated: false } },
  }));
  assert.match(warning, /Experimental recognition — not calibrated/u);
  assert.match(warning, /local evaluation only/u);

  const blocked = renderToStaticMarkup(GroupPhotoWorkflow({
    selection: { valid: true, file: { name: "group.jpg", size: 10 } },
    hasGallery: true,
    recognitionAvailable: false,
    recognitionUnavailableMessage: "Experimental recognition is disabled in backend configuration.",
    busy: null,
  }));
  assert.match(blocked, /disabled in backend configuration/u);
  assert.match(blocked, /<button class="primary-button recognize-button" disabled="">Recognize group photo/u);

  const available = renderToStaticMarkup(GroupPhotoWorkflow({
    selection: { valid: true, file: { name: "group.jpg", size: 10 } },
    hasGallery: true,
    recognitionAvailable: true,
    recognitionUnavailableMessage: "Experimental mode enabled.",
    busy: null,
  }));
  assert.match(available, /Recognize group photo/u);
  assert.doesNotMatch(available, /recognitionUnavailableMessage/u);
});

test("gallery upload errors preserve and render every failed filename, reason, and total", async (t) => {
  t.mock.method(globalThis, "fetch", async () => Response.json({
    detail: {
      code: "gallery_validation_failed",
      message: "3 reference image(s) failed validation",
      invalid_image_count: 3,
      errors: [
        { filename: "Family.png", code: "multiple_faces", message: "More than one face was detected" },
        { filename: "NoFace.jpg", code: "zero_faces", message: "No face was detected" },
        { filename: "Broken.webp", code: "invalid_reference_image", message: "Image could not be decoded" },
      ],
    },
  }, { status: 422 }));

  let caught;
  await assert.rejects(
    api.galleryApi.create([
      new File(["a"], "Family.png", { type: "image/png" }),
      new File(["b"], "NoFace.jpg", { type: "image/jpeg" }),
      new File(["c"], "Broken.webp", { type: "image/webp" }),
    ]),
    (error) => {
      caught = error;
      return error instanceof api.ApiError && error.code === "gallery_validation_failed" &&
        error.invalidImageCount === 3 && error.failures.length === 3;
    },
  );
  const report = galleryFileFailures(caught);
  assert.deepEqual(report, {
    invalidImageCount: 3,
    failures: [
      { filename: "Family.png", code: "multiple_faces", reason: "More than one face was detected.", fix: "Choose a reference photo containing exactly one face." },
      { filename: "NoFace.jpg", code: "zero_faces", reason: "No face was detected in this image.", fix: "Choose a clear reference photo in which one face is visible." },
      { filename: "Broken.webp", code: "invalid_reference_image", reason: "The image is corrupted, unsupported, or its contents do not match its file extension.", fix: "Use a valid single-frame JPG, JPEG, PNG, or WebP image with the matching extension." },
    ],
  });
  const markup = renderToStaticMarkup(GalleryFileFailure({ report }));
  assert.match(markup, /3 invalid images/u);
  for (const filename of ["Family.png", "NoFace.jpg", "Broken.webp"]) assert.match(markup, new RegExp(filename.replaceAll(".", "\\."), "u"));
  assert.match(markup, /Reason: More than one face was detected/u);
  assert.match(markup, /Reason: No face was detected in this image/u);
  assert.match(markup, /Reason: The image is corrupted/u);

  for (const [code, filename] of [
    ["zero_faces", "NoFace.jpg"],
    ["invalid_reference_image", "Broken.webp"],
    ["reference_image_too_large", "Large.jpg"],
    ["duplicate_identity", "alex.png"],
    ["invalid_identity", ".jpg"],
  ]) {
    const single = galleryFileFailures(new api.ApiError("validation", { code, filename }));
    assert.equal(single.failures[0].filename, filename);
    assert.ok(single.failures[0].fix.length > 0);
  }
  t.mock.restoreAll();
});

test("rejects malformed success payloads instead of displaying incomplete results", () => {
  assert.throws(() => decodeRecognitionResponse({ ...knownUnknownFixture, face_count: 1 }), /unexpected recognition response/iu);
  assert.throws(() => decodeRecognitionResponse({ ...knownUnknownFixture, annotated_image_base64: "%%%" }), /unexpected recognition response/iu);
});
