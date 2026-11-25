# Offline Face Recognition

A local application for creating a reference gallery from a user-selected folder and producing candidate identity labels for faces in a group photo. Reference filenames provide identity labels. The application is designed for CPU-first Windows use and performs inference with local ONNX files.

## Current status

The FastAPI backend and React frontend workflows are implemented. YuNet/SFace feasibility experiments and real-person validation work have been run, but the recognition policy is **experimental and not calibrated**. Similarity scores are not probabilities, and this project does not claim general recognition accuracy.

Recognition is unavailable by default. When local models are ready, both matching values are configured, and explicit experimental opt-in is enabled, the application can return experimental Known/Unknown results. The health response continues to report `recognition_ready: false` and `calibrated: false`.

The current real-person validation collection is reported by the project owner as 34 reference identities and 29 group photographs. The checked-in research record does not contain complete, independently verified ground truth and aggregate evaluation metrics for that current collection. Earlier local reports concern a smaller 21-reference/six-photo experiment; they are not evidence of current or general accuracy. No `EVALUATION.md` exists yet.

## Features

- Select a folder of reference images in the browser. Filenames become identity labels; identities are not hard-coded.
- Validate all selected references and show aggregated file-specific failures. Gallery creation is transactional: an invalid image prevents a partial gallery from being created.
- Keep normalized SFace embeddings in process memory, scoped to an expiring gallery session.
- Upload a group photo, detect faces, produce embeddings, and compare them to the active gallery.
- Apply configurable cosine-similarity and top-two margin rules for experimental Unknown rejection.
- Resolve identity conflicts across a whole group photo with a deterministic one-to-one assignment: one identity can be assigned to at most one face. Tied global alternatives remain Unknown rather than being selected by detection order.
- Return a separate identification list and an annotated copy. Known faces use green boxes and label backgrounds; Unknown faces use red. Similarity scores and internal rejection reasons are not shown in the UI.
- Inspect model and experimental recognition availability through local health endpoints.

## Technology

- Python 3.12, FastAPI, Pydantic, Uvicorn, `python-dotenv`, NumPy, Pillow, and `opencv-python-headless`.
- OpenCV YuNet for detection and OpenCV SFace for face alignment and 128-dimensional embeddings.
- React 19 and Vite 7 with local CSS; no UI component framework is required.
- Pytest and HTTPX for backend tests; Node's built-in test runner for frontend tests.

## Requirements and installation

Use Windows with Python 3.12 and Node.js/npm compatible with Vite 7. The pinned model files and license/attribution notices are in `models/`. Install dependencies before going offline; inference does not download packages or models.

From PowerShell at the repository root:

```powershell
py -3.12 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -e ".\backend[test]"
```

Install frontend dependencies separately:

```powershell
Set-Location frontend
npm install
Copy-Item .env.example .env
```

Dependency installation may require a package cache or network access. Runtime model inference and the local UI are designed not to require Internet access.

## Configuration

The backend reads `backend/.env` automatically, anchored to the backend package location. It is ignored by Git. Process environment variables take precedence over values in that file. Passing an explicit environment mapping to `Settings.from_environment()` bypasses dotenv loading.

Start from the documented template:

```powershell
Copy-Item backend/.env.example backend/.env
```

The example keeps explicit opt-in disabled. To run the current experimental policy locally, set all three values in `backend/.env` (or the process environment):

```dotenv
OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD=0.40
OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN=0.10
OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED=true
```

`0.40` and `0.10` are experimental values, not calibrated or production-validated. Both must be supplied together. Removing the opt-in or setting it to `false` disables experimental recognition. Do not commit `backend/.env` or private data.

Other settings have validated defaults in `backend/src/offline_face_recognition/core/config.py`, including image/upload limits, session TTL/capacity, concurrency, detection threshold, and the 1200-pixel maximum detection dimension.

## Run the application

Start the backend from the repository root:

```powershell
backend/.venv/Scripts/python.exe -m uvicorn offline_face_recognition.main:app --app-dir backend/src --host 127.0.0.1 --port 8000
```

In another PowerShell terminal, start the frontend:

```powershell
Set-Location frontend
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`). The API base URL defaults to `http://127.0.0.1:8000`; `VITE_API_BASE_URL` can override it for local development. Keep it pointed at a local backend for offline use. Health endpoints are `/health/live`, `/health/models`, and `/health/recognition`.

## Use

1. In **Reference gallery**, select a folder containing one supported image per identity. Each image must contain exactly one detectable, valid face. Review filenames and validation feedback, then create the gallery.
2. The backend creates an in-memory, session-scoped gallery only after all references pass. The browser keeps the session token in React memory. Replacing a gallery creates the replacement first; if that fails, the previous gallery remains active.
3. Select one group photo and choose **Recognize**. The action is enabled only when the backend reports experimental recognition available. The UI displays a persistent **Experimental recognition — not calibrated** warning in that mode.
4. Review the annotated copy and separate identity list. Output order follows face detection order. Known and Unknown labels are shown without similarity scores. These are experimental decisions, not validated identifications.

The API also supports gallery inspection and explicit deletion. Gallery sessions expire; after backend restart, in-memory sessions are gone and must be recreated.

## Image formats and limits

Only JPG/JPEG, PNG, and WebP are accepted. HEIC and GIF are not supported. Pillow validates encoded format against filename extension, applies EXIF orientation, and decodes to OpenCV-compatible BGR. Animated/multi-frame files, corrupted files, invalid dimensions, and over-limit images are rejected.

Defaults include up to 50 reference identities, 20 MiB per uploaded image, 100 MiB total gallery upload, 12,000 pixels per image dimension, and 40 million pixels per image. Group-photo processing is limited to one concurrent job by default. Detection downsizes only images whose longest side exceeds 1200 pixels; it preserves aspect ratio and does not upscale. Configure limits through the `OFFLINE_FACE_*` settings in backend config.

Each reference must yield exactly one valid face. Group faces that cannot be processed fail the complete request rather than being silently omitted. Recognition requires at least two gallery identities to establish the top-two margin; ties are rejected. One-to-one assignment can leave otherwise plausible faces Unknown when identity conflicts or global assignment ties cannot be resolved safely.

## Offline operation and data handling

The application defaults to a loopback API and uses local frontend assets and locally stored ONNX models. Model loading uses OpenCV's CPU backend/target and disables OpenCL; no runtime model downloads or cloud recognition calls are part of the implementation. A fully packaged, clean-machine offline deployment has not been claimed as independently validated.

Reference and group images are processed locally and are not intentionally stored as application files. Gallery embeddings remain in process memory and are cleared best-effort on deletion, expiry, and shutdown. Browser session tokens are memory-only. Do not put personal photos, embeddings, outputs, or `.env` files in Git.

## Tests

From the repository root:

```powershell
backend/.venv/Scripts/python.exe -m pytest -c backend/pyproject.toml backend/tests
Set-Location frontend
npm test
npm run build
```

Some optional integration tests depend on local synthetic data and skip when it is absent. The real-person validation data and experiment outputs are local and Git-ignored; they are not test fixtures to publish.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current component design and [RESEARCH.md](RESEARCH.md) for recorded research and experiment limitations.
