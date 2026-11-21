# Offline Face Recognition

## Problem

An office needs to identify people in a group photo using a user-selected folder of reference photos, where each reference filename provides the person's identity label. People who do not match the selected reference folder should be reported as unknown.

## Intended application

The proposed application will let a user select a reference-photo folder and upload one group photo. It is intended to detect and align faces, compare their embeddings with a session gallery built from the references, label known, unknown, or uncertain faces on the photo, and show a separate results list. A different reference folder should work without identity-specific code changes or retraining.

## Proposed technologies

- **Python 3.12**, **FastAPI**, and **Pydantic** for a local backend and request validation.
- **React + Vite** and **Tailwind CSS** for the frontend; the UI component library is still to be decided.
- **OpenCV YuNet** for face detection and **OpenCV SFace** for embeddings, using ONNX model files stored locally. YuNet and SFace are candidates pending experimental validation.
- **NumPy** for normalized embeddings and cosine similarity; **Pillow** where useful for image handling.
- **Pytest + HTTPX** for backend and API testing.

The approved architecture keeps reference embeddings in process memory, scoped to a gallery session, with no database server or persistent embedding cache initially. Reconsider persistence only if experiments demonstrate a clear need and it is explicitly approved.

## Offline runtime

The intended application must run fully offline after setup. Frontend assets and ONNX model files must be loaded locally; runtime cloud APIs, external recognition services, model downloads, and other external network requests are out of scope. Offline setup and execution still need to be implemented and verified on Windows.

## Current status

Architecture is documented in [ARCHITECTURE.md](ARCHITECTURE.md). Application implementation and recognition experiments have not yet started. No application behavior, model accuracy, or offline runtime has been validated.

## Planned development and validation

1. Research model licenses, Windows CPU compatibility, image handling, and suitable practice data; run controlled experiments for detection, embeddings, and matching thresholds.
2. Implement and test the modular backend, including reference validation, session gallery creation, matching, unknown rejection, and annotation.
3. Build the React folder-selection, upload, and results-review flow, preserving reference filenames.
4. Test invalid images, zero or multiple faces, unknown identities, ambiguous matches, and representative group-photo conditions.
5. Package local frontend assets and model files, then verify the complete app on a clean Windows environment with internet access disabled.

These are planned stages; none should be read as completed work.
