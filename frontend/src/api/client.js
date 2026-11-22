const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = (configuredBaseUrl || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, { status = null, code = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function requestJson(path, { method = "GET", sessionId, body, allowUnavailable = false } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: {
        Accept: "application/json",
        ...(sessionId ? { Authorization: `Bearer ${sessionId}` } : {}),
      },
      ...(body ? { body } : {}),
      cache: "no-store",
    });
  } catch {
    throw new ApiError("Could not reach the local backend. Check that it is running.", { code: "network_error" });
  }

  if (response.status === 204) return null;

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("The backend returned an unreadable response.", { status: response.status });
  }

  if (!response.ok && !(allowUnavailable && response.status === 503)) {
    const detail = data?.detail;
    throw new ApiError(
      detail?.message || detail?.code || `The backend request failed (${response.status}).`,
      { status: response.status, code: detail?.code || null },
    );
  }

  // Readiness endpoints intentionally use HTTP 503 when unavailable. Keep
  // their response bodies so the UI can distinguish not-ready from offline.
  return { data, ok: response.ok, status: response.status };
}

export const healthApi = {
  liveness: () => requestJson("/health/live", { allowUnavailable: true }),
  models: () => requestJson("/health/models", { allowUnavailable: true }),
  recognition: () => requestJson("/health/recognition", { allowUnavailable: true }),
};

export const galleryApi = {
  async create(files) {
    const form = new FormData();
    for (const file of files) {
      form.append("reference_images", file, file.name);
    }
    // Do not set Content-Type: fetch must add the multipart boundary.
    const result = await requestJson("/api/galleries", { method: "POST", body: form });
    return result.data;
  },

  async getCurrent(sessionId) {
    const result = await requestJson("/api/galleries/current", { sessionId });
    return result.data;
  },

  deleteCurrent(sessionId) {
    return requestJson("/api/galleries/current", { method: "DELETE", sessionId });
  },
};
