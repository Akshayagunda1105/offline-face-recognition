# Offline Face Recognition

## Problem

An office needs to identify people in a group photo using a user-selected folder of reference photos, where each reference filename provides the person's identity label. People who do not match the selected reference folder should be reported as unknown.

## Intended application

The proposed application will let a user select a reference-photo folder and upload one group photo. It is intended to detect and align faces, compare their embeddings with a session gallery built from the references, label known, unknown, or uncertain faces on the photo, and show a separate results list. A different reference folder should work without identity-specific code changes or retraining.

## Proposed technologies

- **Python 3.12**, **FastAPI**, and **Pydantic** for a local backend and request validation.
- **React + Vite** and **Tailwind CSS** for the frontend; the UI component library is still to be decided.
- **OpenCV YuNet** for face detection and **OpenCV SFace** for embeddings, using locally stored ONNX models. Initial synthetic feasibility experiments have been completed; broader validation remains pending.
- **NumPy** for normalized embeddings and cosine similarity; **Pillow** where useful for image handling.
- **Pytest + HTTPX** for backend and API testing.

The approved architecture keeps reference embeddings in process memory, scoped to a gallery session, with no database server or persistent embedding cache initially. Reconsider persistence only if experiments demonstrate a clear need and it is explicitly approved.

## Offline runtime

The intended application must run fully offline after setup. Frontend assets and ONNX model files must be loaded locally; runtime cloud APIs, external recognition services, model downloads, and other external network requests are out of scope. Offline setup and execution still need to be implemented and verified on Windows.

## Current status

The proposed architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md), and experimental methods and results are recorded in [RESEARCH.md](RESEARCH.md). Initial synthetic feasibility experiments covered YuNet detection and threshold sensitivity, SFace model loading and embedding extraction, an 8 × 5 reference-to-group similarity matrix, and a 5 × 5 reference consistency matrix. A YuNet detection score threshold of 0.85 is provisional for subsequent experiments.

The application is not implemented. These small synthetic experiments do not establish general recognition accuracy or a final open-set matching threshold. Real-person validation, matching-threshold calibration, and clean-machine offline application testing remain pending.

## Planned development and validation

1. Validate detection and embedding behavior on representative real-person data with appropriate consent; evaluate false matches and calibrate open-set and ambiguity thresholds.
2. Implement and test the modular backend, including reference validation, session gallery creation, matching, unknown rejection, and annotation.
3. Build the React folder-selection, upload, and results-review flow, preserving reference filenames.
4. Test invalid images, zero or multiple faces, unknown identities, ambiguous matches, and representative group-photo conditions.
5. Package local frontend assets and model files, then verify the complete app on a clean Windows environment with internet access disabled.

Application stages remain planned; the synthetic model experiments are the completed work described in [RESEARCH.md](RESEARCH.md).
