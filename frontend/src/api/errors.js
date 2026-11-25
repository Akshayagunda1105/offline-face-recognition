import { ApiError } from "./client.js";
import { MAX_GALLERY_IMAGES } from "../gallery/files.js";
import { MAX_GROUP_PHOTO_BYTES } from "../group-photo/file.js";

const API_ERROR_MESSAGES = {
  network_error: "Could not reach the local backend. Check that it is running.",
  reference_images_required: "Select at least one supported reference image.",
  empty_gallery: "Select at least one reference image.",
  too_many_identities: `A gallery can contain at most ${MAX_GALLERY_IMAGES} reference images.`,
  invalid_identity: "A filename does not provide a valid identity label.",
  duplicate_identity: "More than one file has the same identity label. Use one image and a unique filename per person.",
  invalid_reference_image: "A reference image is invalid, unsupported, corrupted, or its extension does not match its contents.",
  zero_faces: "A reference image contains no face. Each reference photo must contain exactly one face.",
  multiple_faces: "A reference image contains multiple faces. Each reference photo must contain exactly one face.",
  invalid_face_detection: "A reference face could not be validated. Review that image and try again.",
  embedding_alignment_failure: "A reference face could not be aligned. Review that image and try again.",
  invalid_embedding: "A reference face could not be converted into a valid representation.",
  face_processing_failed: "Face processing failed. The backend did not return a partial result.",
  gallery_too_large: "The selected references exceed the backend upload limit.",
  gallery_session_capacity: "The backend has reached its active-gallery limit. Delete an unused gallery or wait for one to expire.",
  processing_busy: "Another local image job is running. Try again shortly.",
  models_not_ready: "The local face models are unavailable. Check backend model readiness.",
  gallery_session_required: "The gallery session is missing. Create a reference gallery again.",
  gallery_session_unavailable: "The gallery expired or was cleared by a backend restart. Select the reference folder again.",
  gallery_store_closed: "The backend gallery service is unavailable. Try again after it restarts.",
  recognition_not_ready: "Experimental recognition is unavailable. Enable explicit experimental mode and configure both thresholds; decisions remain uncalibrated.",
  photo_required: "Choose a group photo before submitting.",
  invalid_group_photo: "The photo is corrupted, unsupported, too large in dimensions, or its filename extension does not match its contents.",
  group_photo_too_large: `The group photo exceeds the ${MAX_GROUP_PHOTO_BYTES / (1024 * 1024)} MiB upload limit.`,
  annotated_response_too_large: "The annotated image response exceeds the backend size limit.",
  invalid_face_detections: "The backend returned invalid face detections. No partial result was displayed.",
  face_embedding_misaligned: "The backend could not align all detected faces with their embeddings.",
  original_coordinates_missing: "The backend could not map all faces to the original photo coordinates.",
  matching_incomplete: "The backend did not return a result for every detected face.",
  matching_misaligned: "The backend returned face results in an inconsistent order.",
  annotation_failed: "The backend could not create the annotated image.",
};

const GALLERY_FILE_FAILURES = {
  reference_image_too_large: {
    reason: "This reference image exceeds the individual upload size limit.",
    fix: "Choose a smaller file and try again.",
  },
  multiple_faces: {
    reason: "More than one face was detected.",
    fix: "Choose a reference photo containing exactly one face.",
  },
  zero_faces: {
    reason: "No face was detected in this image.",
    fix: "Choose a clear reference photo in which one face is visible.",
  },
  invalid_reference_image: {
    reason: "The image is corrupted, unsupported, or its contents do not match its file extension.",
    fix: "Use a valid single-frame JPG, JPEG, PNG, or WebP image with the matching extension.",
  },
  duplicate_identity: {
    reason: "Another uploaded image uses the same identity filename.",
    fix: "Keep one reference image per identity and give each person a unique filename stem.",
  },
  invalid_identity: {
    reason: "This filename cannot be used as an identity label.",
    fix: "Rename the image with a non-empty, readable identity name before its extension.",
  },
};

export function galleryFileFailures(error) {
  if (!(error instanceof ApiError)) return null;
  const rawFailures = error.failures?.length
    ? error.failures
    : error.filename
      ? [{ filename: error.filename, code: error.code, message: error.message }]
      : [];
  if (!rawFailures.length) return null;
  const failures = rawFailures.map((failure) => {
    const guidance = GALLERY_FILE_FAILURES[failure.code];
    return {
      filename: failure.filename,
      code: failure.code,
      reason: guidance?.reason || failure.message || "This reference image failed validation.",
      fix: guidance?.fix || "Review this image and select a valid reference photo, then try again.",
    };
  });
  const uniqueFileCount = new Set(failures.map((failure) => failure.filename)).size;
  return {
    invalidImageCount: error.invalidImageCount ?? uniqueFileCount,
    failures,
  };
}

export function friendlyError(error) {
  if (!(error instanceof ApiError)) return error?.message || "The backend returned an unexpected response. Please try again.";
  let message = API_ERROR_MESSAGES[error.code] || error.message || "The backend request failed. Please try again.";
  if (error.code === "processing_busy" && error.retryAfterSeconds !== null) {
    message += ` Retry after ${error.retryAfterSeconds} second${error.retryAfterSeconds === 1 ? "" : "s"}.`;
  }
  return message;
}
