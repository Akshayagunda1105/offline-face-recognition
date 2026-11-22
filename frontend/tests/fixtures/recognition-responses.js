// Test-only payloads. These are fabricated UI fixtures, never model predictions.
export const knownUnknownFixture = Object.freeze({
  face_count: 2,
  annotated_image_media_type: "image/jpeg",
  annotated_image_base64: "aGVsbG8=",
  identifications: [
    { detection_index: 0, bounding_box: { x: 1, y: 2, width: 30, height: 40 }, status: "Known", identity_label: "Fixture Person" },
    { detection_index: 1, bounding_box: { x: 50, y: 3, width: 28, height: 38 }, status: "Unknown", identity_label: null },
  ],
});

export const zeroFacesFixture = Object.freeze({
  face_count: 0,
  annotated_image_media_type: "image/jpeg",
  annotated_image_base64: "aGVsbG8=",
  identifications: [],
});
