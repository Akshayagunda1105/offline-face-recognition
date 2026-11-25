# Offline Face Recognition Architecture

> **Status:** The backend and frontend workflows described here are implemented. Recognition remains explicitly experimental and uncalibrated. YuNet's detection threshold is provisional; matching thresholds are configurable experimental policy, not validated performance guarantees.

## Diagram

[![Offline face-recognition architecture](docs/architecture/preview.png)](.archify/architecture-offline-face-recognition-20261121-0124/offline-face-recognition.html)

The diagram is a component-level view of the modular monolith. The sections below describe implementation details that are not expanded in the diagram, including aggregated validation errors and global one-to-one matching.

## Runtime components

| Component | Responsibility and current behavior |
|---|---|
| **React + Vite frontend** | Provides reference-folder selection, gallery status/replacement/deletion, group-photo preview/upload, health status, annotated-result rendering, and a separate Known/Unknown list. Keeps the session token in React memory and displays the uncalibrated-experiment warning when experimental recognition is available. |
| **FastAPI API** | Validates multipart requests, enforces upload limits, authenticates gallery operations with a bearer session token, maps typed failures to HTTP responses, and coordinates the processing services. |
| **Settings** | Validates environment configuration. Loads `backend/.env` by package-anchored path only when no explicit environment mapping is supplied; existing process variables win. Matching threshold and ambiguity margin must both be configured or both absent. Experimental opt-in defaults false. |
| **ModelRuntime** | Loads one local YuNet ONNX model and one local SFace ONNX model during FastAPI lifespan. Uses OpenCV DNN CPU backend/target and disables OpenCL. It does not download models. Calibrated `recognition_ready` remains false; experimental availability requires ready models, explicit opt-in, and both thresholds. |
| **Image decoder** | Pillow verifies JPG/JPEG, PNG, or WebP content against the extension, rejects multi-frame/corrupt files and resource-limit violations, applies EXIF orientation, and returns a BGR array. It does not intentionally persist images. |
| **GalleryBuilder** | Sanitizes filename basenames and derives display labels from stems; normalized keys detect case/Unicode-equivalent duplicates. Validates every selected image, collecting per-file failures. Rejects zero/multiple faces and invalid detection or embedding results. It creates no session unless every image succeeds. |
| **FaceAnalysisEngine / YuNetDetector** | Conditionally resizes only when the longest dimension exceeds the configured maximum (default 1200 px), preserving aspect ratio without upscaling. YuNet detects on this working image using provisional score threshold 0.85, NMS 0.30, top_k 5000. Raw detector rows/landmarks remain in working-image coordinates for SFace. Bounding boxes are separately mapped to original-image coordinates using actual rounded X/Y resize scales. |
| **FaceAlignment / SFaceEmbedder** | Calls SFace alignment with the same working image and the corresponding unchanged YuNet row. Produces finite, L2-normalized `float32` vectors of shape `(128,)`. Detection and embedding failures are surfaced with context; faces are not silently skipped. |
| **GallerySessionStore / GalleryService** | Holds a random session ID, identity-to-embedding mappings, creation and expiry times in process memory. Defaults: 30-minute TTL, 30-second cleanup interval, maximum five active sessions. Session access is isolated; deletion/expiry/shutdown clear references and arrays best-effort. No database or persistent embedding cache is used. |
| **FaceMatcher** | Computes cosine similarity for each query against every gallery identity and retains identity-score associations. Single-face policy requires both configured similarity and top-two margin conditions; a score tie is Unknown even when configured margin is zero. A one-identity gallery cannot establish a top-two margin and is rejected. |
| **Batch assignment** | Resolves all faces in one group photo together. It enforces at most one face per identity and at most one identity per face, favors the maximum number of policy-valid assignments, then the greatest total cosine similarity. It preserves the strongest individually valid top claim for contested identities; other faces can receive a valid alternative only when the policy remains satisfied against identities left unassigned. Equal claim ties and equally optimal global assignments are not broken by detection order: affected faces remain Unknown. The input detection indices are preserved. |
| **Unknown rejection** | Uses the same global/configured similarity threshold and ambiguity margin for every gallery; no identity-specific rules are present. Low scores, insufficient margin, ties, missing alternatives, and unresolved assignment conflicts produce `Unknown`. Similarities are measurements, not probabilities. Internal rejection reasons/scores are not returned by the user-facing result. |
| **ImageAnnotator** | Draws onto a copy of the original-resolution BGR image. Known identity boxes and label backgrounds are green; Unknown boxes and label backgrounds are red. Coordinates are clipped to image bounds; the uploaded source array is not modified. |

## Gallery creation flow

1. The browser's directory picker supplies files. The frontend accepts JPG/JPEG/PNG/WebP, preserves filenames, rejects empty selection, unsupported formats, case-insensitive duplicate stems, and more than 50 selected images before upload.
2. React sends every file in repeated multipart `reference_images` fields to `POST /api/galleries`. It does not set multipart `Content-Type` manually.
3. FastAPI enforces individual and aggregate byte limits and promptly closes upload objects. The builder first validates identity names and decodes/checks each image. Basic validation failures are gathered before expensive model inference.
4. For every remaining valid file, the engine must produce exactly one valid detection and one normalized embedding. Zero faces, multiple faces, invalid detector rows, decode/embedding failures, invalid identities, and duplicate identities are reported with filenames in a structured `errors` array.
5. Gallery construction is transactional: if any selected image fails, the request returns validation errors and no new session is created. A previous frontend gallery remains active on replacement failure. On successful replacement, the frontend adopts the new session and then requests deletion of the old one.
6. The successful response supplies a session ID, timestamps, identity count, and labels. The token is retained only in frontend memory; the backend store retains embeddings only for the session TTL or until deletion/shutdown.

## Group-photo processing and matching flow

1. The frontend submits one JPG/JPEG/PNG/WebP photo as multipart `photo` to `POST /api/group-photos/recognize`, with `Authorization: Bearer <session_id>`. The initial maximum upload size is 20 MiB; processing concurrency defaults to one job.
2. The route first checks experimental availability. The backend returns `recognition_not_ready` (HTTP 503) unless models are ready, both thresholds are valid, and explicit experimental opt-in is on. This does not change calibrated `recognition_ready`, which remains false.
3. The decoder validates the image. The engine prepares the same conditional-resize working image for YuNet detection and SFace alignment, extracts one embedding for every valid detection, and keeps each YuNet detection index attached to that result. Any invalid row, face failure, or mapping mismatch fails the whole request; no partial identification list is returned.
4. The gallery service ranks every detected face against the selected session's normalized embeddings. The matcher performs the global one-to-one assignment and re-evaluates margins among still-unassigned identities. Neither detector ordering nor an arbitrary tie-break assigns an identity.
5. The processor maps Known decisions to filename-derived labels and all rejections to `Unknown`. It annotates a copy of the original-resolution image, preserving the original input.
6. The API returns `face_count`, a JPEG `annotated_image_base64` plus media type, and an ordered `identifications` array containing detection index, original-coordinate bounding box, Known/Unknown status and optional identity label. Similarity scores, margins, calibration claims, and internal rejection reasons are not included.
7. The frontend decodes the annotated response into a Blob URL, displays the image and separate list, and revokes prior object URLs when replacing/clearing results or unmounting. Zero faces produce an explicit no-faces state. On an expired/deleted session it clears stale gallery/results and asks the user to create a gallery again.

## API surface

| Method and path | Authentication | Behavior |
|---|---|---|
| `GET /health/live` | None | Liveness status. |
| `GET /health/models` | None | YuNet/SFace loading status; returns 503 if local models are unavailable. |
| `GET /health/recognition` | None | Reports `models_ready`, `recognition_ready` (always false), `experimental_recognition_available`, `experimental`, `calibrated` (always false), and deterministic `reason`. A successful status report is HTTP 200 even when recognition is unavailable. |
| `POST /api/galleries` | None | Multipart repeated `reference_images`; returns 201 with session ID, times, identity count and labels, or structured per-file validation failures. |
| `GET /api/galleries/current` | Bearer gallery session | Returns current gallery metadata and labels; no embeddings. |
| `DELETE /api/galleries/current` | Bearer gallery session | Deletes the session and releases in-memory embeddings best-effort; returns 204. |
| `POST /api/group-photos/recognize` | Bearer gallery session | Multipart `photo`; requires experimental availability and returns an annotated image plus ordered identifications. |

FastAPI's generated OpenAPI schema describes the health and application routes. Errors use structured codes, including model/readiness, image validation, gallery/session, concurrency, and processing failures. Uploads use Starlette's temporary spooling; the application does not intentionally save source photos as files. The annotated image is bounded by the configured response-size limit.

## Configuration and offline boundary

The model paths default to repository-local ONNX assets and are resolved independently of the working directory. The backend performs no runtime model download. The UI bundles its React/Vite assets locally and defaults to the loopback backend. `VITE_API_BASE_URL` is configurable, so offline deployment must keep it local. The code has not been certified as a packaged application on a clean machine with networking physically disabled.

`backend/.env.example` documents optional local evaluation settings. Fresh/default settings keep experimental recognition disabled and both matching thresholds unset. The supplied example threshold pair `0.40`/`0.10` is explicitly experimental, not calibrated. The private `backend/.env` is Git-ignored; process variables override file values. There is no separate calibrated mode in the current implementation.

## Trade-offs and known limits

- **YuNet + SFace:** Uses OpenCV's local DNN and face APIs with pinned ONNX files and CPU execution. This keeps the runtime comparatively compact and avoids external inference services. It gives up the possibility that another recognizer or detector performs better on difficult poses, small faces, or varied capture conditions; model selection has not been established as universally optimal.
- **One reference per identity:** Keeps gallery creation simple and supports arbitrary user-selected folders without identity-specific training. It cannot represent within-person appearance variation as well as multiple independent references.
- **Cosine plus margin rejection:** Is transparent and inexpensive for normalized embeddings. Experimental values may reject genuine people or accept unknown ones; no calibrated performance guarantee is made.
- **Global one-to-one assignment:** Prevents the same gallery identity from being assigned to multiple faces in one photo and resolves conflicts across the whole batch. It adds combinatorial work and may conservatively leave faces Unknown when equally good assignments exist.
- **In-memory sessions:** Avoids a database and persistent biometric cache. Sessions are process-local and disappear on restart; they are not shared across workers/processes.
- **Per-request failures:** The backend fails the complete group-photo request if face processing is incomplete or inconsistent, rather than returning a partial result that could misalign detections and identities.

Implementation/test details are in the backend and frontend source trees; experiment records and limitations are in [RESEARCH.md](RESEARCH.md). This architecture document does not claim calibrated accuracy.
