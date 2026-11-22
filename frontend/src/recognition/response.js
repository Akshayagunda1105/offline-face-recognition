export class InvalidRecognitionResponseError extends Error {
  constructor(message) {
    super(message);
    this.name = "InvalidRecognitionResponseError";
  }
}

function invalidResponse() {
  return new InvalidRecognitionResponseError("The backend returned an unexpected recognition response.");
}

export function decodeRecognitionResponse(payload) {
  if (!payload || typeof payload !== "object" || payload.annotated_image_media_type !== "image/jpeg") {
    throw invalidResponse();
  }
  if (!Number.isInteger(payload.face_count) || payload.face_count < 0 || !Array.isArray(payload.identifications) ||
      payload.face_count !== payload.identifications.length || typeof payload.annotated_image_base64 !== "string" ||
      payload.annotated_image_base64.length === 0) {
    throw invalidResponse();
  }

  const identifications = payload.identifications.map((item) => {
    if (!Number.isInteger(item?.detection_index) || item.detection_index < 0 ||
        !["Known", "Unknown"].includes(item.status) ||
        (item.status === "Known" && (typeof item.identity_label !== "string" || !item.identity_label)) ||
        (item.status === "Unknown" && item.identity_label !== null)) {
      throw invalidResponse();
    }
    return {
      detection_index: item.detection_index,
      status: item.status,
      identity_label: item.identity_label,
    };
  });

  let binary;
  try {
    binary = globalThis.atob(payload.annotated_image_base64);
  } catch {
    throw invalidResponse();
  }
  if (binary.length === 0) throw invalidResponse();
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);

  return {
    faceCount: payload.face_count,
    annotatedBlob: new Blob([bytes], { type: payload.annotated_image_media_type }),
    identifications,
  };
}
