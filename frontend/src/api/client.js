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

async function getHealth(path) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("Could not reach the local backend. Check that it is running.");
  }

  let data;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("The backend returned an unreadable response.", { status: response.status });
  }

  // Readiness endpoints intentionally use HTTP 503 when unavailable. Keep
  // their response bodies so the UI can distinguish not-ready from offline.
  return { data, ok: response.ok, status: response.status };
}

export const healthApi = {
  liveness: () => getHealth("/health/live"),
  models: () => getHealth("/health/models"),
  recognition: () => getHealth("/health/recognition"),
};
