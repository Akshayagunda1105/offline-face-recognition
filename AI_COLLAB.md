# AI Collaboration Record

This record describes the AI-assisted work honestly and separates verifiable Codex history from ChatGPT contributions described by the project owner. It is a summary, not a reconstructed transcript. The application itself performs face detection and recognition locally with the bundled OpenCV models; it makes no cloud AI or recognition API calls at runtime.

## Collaborations

### Codex

Codex was used for repository setup and inspection, technical research, architecture planning and Archify work, model provenance and licence checks, implementation, debugging, testing, and documentation review. The decisions below are supported by the available Codex conversation and repository evidence. They do not imply that every AI suggestion was accepted.

### ChatGPT

The project owner describes ChatGPT as an independent technical mentor/reviewer used to assess Codex suggestions and review technical decisions. The ChatGPT conversation contents were not available in the working repository or readable from the provided share page during this update. Accordingly, this document does not assign particular suggestions or quoted language to ChatGPT. The owner-provided conversation link is included below so the full history can be reviewed separately.

## Key decisions

### 1. Offline, CPU-first modular monolith

- **AI suggested:** A local modular monolith with separate UI, API, face-analysis, gallery, matching, and annotation responsibilities, using local models and no runtime cloud inference.
- **I decided:** Keep the application on React/Vite and FastAPI, run locally on Windows/CPU, and load local ONNX models without runtime downloads.
- **Why:** The evaluator must be able to run it offline on unseen data, and the components need independent responsibilities without distributed-service overhead.
- **Evidence:** [ARCHITECTURE.md](ARCHITECTURE.md); `backend/src/offline_face_recognition/main.py`; `backend/src/offline_face_recognition/models/runtime.py`; `frontend/vite.config.js`.

### 2. YuNet and SFace model selection

- **AI suggested:** OpenCV YuNet for detection and OpenCV SFace for embeddings as a compact, locally runnable pipeline.
- **I decided:** Use these models, pin the OpenCV Zoo source revisions, keep the ONNX assets in the repository with their applicable notices, and treat YuNet's `0.85` detection score as provisional.
- **Why:** The installed OpenCV APIs load the local models and support CPU inference; the small experiments justified feasibility only, not comparative superiority or general accuracy.
- **Evidence:** [RESEARCH.md](RESEARCH.md); `models/ATTRIBUTION.txt`; `models/LICENSE`; `models/sface/ATTRIBUTION.txt`; `models/sface/LICENSE`; `backend/src/offline_face_recognition/models/runtime.py`.

### 3. Dynamic filename-based reference gallery

- **AI suggested:** Build a session gallery dynamically from the user's selected folder and derive labels from filenames instead of hard-coding identities or training per identity.
- **I decided:** Keep one reference image per identity, accept JPG/JPEG/PNG/WebP, validate the full selection, and create a gallery only when every reference succeeds.
- **Why:** The judge must be able to replace the dataset without code changes; transactional validation avoids a partial or mislabelled gallery.
- **Evidence:** `backend/src/offline_face_recognition/gallery/identity.py`; `backend/src/offline_face_recognition/gallery/builder.py`; `backend/tests/unit/test_gallery_builder.py`; `frontend/src/` gallery workflow.

### 4. Session-scoped in-memory embeddings

- **AI suggested:** Keep biometric embeddings in process memory and scope them to an expiring gallery session rather than introducing a database or persistent cache.
- **I decided:** Use unpredictable session IDs, a configurable TTL and capacity, explicit deletion, and best-effort clearing on deletion, expiry, and shutdown.
- **Why:** This limits retention and is sufficient for the current gallery size; a process restart can require the user to rebuild the gallery.
- **Evidence:** `backend/src/offline_face_recognition/gallery/session_store.py`; `backend/tests/unit/test_session_store.py`.

### 5. Configurable cosine rejection policy, explicitly experimental

- **AI suggested:** Compare normalized SFace vectors using cosine similarity and reject weak or ambiguous candidates using an absolute score and top-two margin.
- **I decided:** Keep the `0.40` similarity and `0.10` margin as configurable experimental values, require both together and an explicit opt-in, default the opt-in to false, and keep calibrated readiness false.
- **Why:** Experimental execution is useful for evaluation, but the available data does not justify claiming calibration. Similarity is not a probability.
- **Evidence:** `backend/src/offline_face_recognition/core/config.py`; `backend/src/offline_face_recognition/models/runtime.py`; `backend/src/offline_face_recognition/matching/matcher.py`; `backend/.env.example`; `backend/tests/unit/test_config.py`; `backend/tests/unit/test_model_runtime.py`.

### 6. One-to-one batch identity assignment

- **AI suggested:** Resolve identity conflicts across the entire group photo instead of independently allowing several faces to receive the same identity.
- **I decided:** Use a global assignment that maximizes valid assignment count and then score, allows policy-valid alternatives, and leaves tied claims or tied global outcomes Unknown without using detection order as a tie-break.
- **Why:** A person's identity should appear at most once in a group photo, and arbitrary order-based conflict resolution would make results unstable.
- **Evidence:** `backend/src/offline_face_recognition/matching/matcher.py`; `backend/tests/integration/test_batch_gallery_matching.py`.

### 7. Conditional 1200-pixel detection resize

- **AI suggested:** For large images, resize before YuNet while preserving aspect ratio, then map boxes to original-image coordinates; use the same working image and raw detection row for SFace alignment.
- **I decided:** Apply the resize only when the longest side exceeds 1200 pixels, never upscale, preserve original detection rows and indices, and annotate a separate original-resolution copy.
- **Why:** Diagnostics showed improved detection feasibility for some large references; the change must preserve coordinate and alignment consistency and does not claim improved recognition accuracy.
- **Evidence:** `backend/src/offline_face_recognition/faces/resizing.py`; `backend/src/offline_face_recognition/faces/engine.py`; `backend/tests/unit/test_resizing.py`; `backend/tests/integration/test_face_engine.py`.

### 8. Aggregate validation and explicit test boundaries

- **AI suggested:** Report invalid references with file-specific reasons, validate basic errors before expensive inference, and test policy behavior separately from real-person evaluation.
- **I decided:** Collect all safe per-image validation failures, reject gallery creation transactionally, retain the previous gallery on failed replacement, and keep local photos, embeddings, and experimental outputs out of Git. Do not treat model predictions as ground truth.
- **Why:** The user needs actionable feedback without partially replacing a working gallery, and test/evaluation evidence must not leak private biometric data or overstate performance.
- **Evidence:** `backend/src/offline_face_recognition/gallery/builder.py`; `backend/src/offline_face_recognition/api/galleries.py`; `backend/tests/unit/test_gallery_builder.py`; `frontend/src/` gallery error rendering; `RESEARCH.md` limitations.

## Genuine excerpts from the available Codex conversation

These are user-authored instructions, copied from the Codex conversation to show the decisions made. They are not presented as AI-authored statements.

> **Codex conversation — user instruction:** “the app must run fully offline on my laptop, no internet at runtime, no cloud AI APIs inside the solution.”

> **Codex conversation — user decision:** “Keep gallery embeddings in process memory, scoped to a session.”

> **Codex conversation — user refinement:** “If the highest and second-highest similarity scores are equal, reject the result as ambiguous, even when the configured ambiguity margin is zero.”

No ChatGPT quotation is included because the linked ChatGPT conversation could not be inspected here. Add a verbatim excerpt only after checking that source directly; do not paraphrase one as a quotation.

## Complete conversation histories

- **Codex:** [[CODEX_SHARE_LINK]](https://chatgpt.com/s/cx_6b071beb23648191a0c5d4cb1b841402)
- **ChatGPT technical mentor/reviewer:** [[CHATGPT_SHARE_LINK]](https://chatgpt.com/share/6b071bfd-1b5c-83ee-a354-7d9cd9cb388d)

These links were supplied by the project owner. Their contents are not reproduced or reconstructed in this file. Preserve the complete histories separately as required by the assignment.

## AI use and evidence limits

AI was used during development, as summarized above. At runtime, the application loads the local YuNet and SFace ONNX files and processes images on the local machine; no cloud AI or recognition API is part of the runtime pipeline.

The project instructions supplied in this workspace also state that AI tools may be used only to create practice data. That restriction appears to conflict with the development assistance recorded here. The original assignment brief should be checked before submission; this record does not resolve or conceal the conflict.

The owner reports the current validation collection contains 34 reference identities and 29 group photographs. This document does not claim complete ground truth, threshold calibration, or aggregate accuracy for that collection. See [RESEARCH.md](RESEARCH.md) for the limits on recorded experiments. No photographs, embeddings, secrets, or person-level validation outputs are included here.
