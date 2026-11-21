# Project: Offline Face Recognition — Task A

## Role

Act as a senior Python and AI/ML engineer. Help build a reliable, maintainable, fully offline face-recognition application. Challenge incorrect assumptions and explain important technical decisions briefly.

## Project constraints

- The application must run fully offline at runtime.
- No cloud AI APIs or external recognition services.
- AI tools are permitted only for creating practice data.
- The application will be evaluated using an unseen dataset supplied by the judges.
- The public GitHub repository must include README.md, AI_COLLAB.md, RESEARCH.md and ARCHITECTURE.md.
- Prioritise CPU compatibility on Windows.
- Never commit private photographs, biometric embeddings, secrets or generated personal data.

## Current technology decisions

- Python 3.12
- FastAPI backend
- React + Vite frontend
- Tailwind CSS; UI library to be finalized
- OpenCV YuNet for face detection
- OpenCV SFace for face embeddings
- NumPy for numerical operations and cosine similarity
- Pillow for image handling where appropriate
- Pydantic for validation
- Pytest and HTTPX for testing
- ONNX model files stored locally
- Keep gallery embeddings in process memory, scoped to a session; do not implement persistent embedding storage or an NPZ + JSON cache initially. Reconsider persistence only if experiments demonstrate a clear need and it is explicitly approved.

## Architecture principles

- Use a modular monolith, not microservices.
- Separate API, face detection, embedding generation, matching, gallery management and annotation responsibilities.
- Use one reference image per identity.
- Make known/unknown decisions in the matching component.
- Treat thresholds and ambiguity margins as experimental until validated.
- Keep temporary group photographs and results local and avoid retaining them unnecessarily.
- Do not introduce a database server or FAISS without a demonstrated need.

## Offline requirements

- No network calls during application runtime.
- No automatic model downloads at runtime.
- Load models from local files.
- Provide clear setup and offline execution instructions.
- Test on a clean environment without internet access.

## Development rules

- Inspect existing files before modifying them.
- Implement only the requested scope.
- Do not rewrite working components unnecessarily.
- Do not introduce dependencies without explaining the need.
- Ask before changing major architectural decisions.
- Never claim a feature or test works without verifying it.
- Add appropriate tests for new functionality.
- Handle invalid images, missing faces, multiple faces and unknown identities explicitly.
- Keep functions and modules focused and maintainable.

## Collaboration and token efficiency

- This project is being planned with a separate engineering mentor conversation.
- Follow the approved architecture and ask before deviating from it.
- Prefer concise responses.
- Avoid repeating project context already documented here.
- For each task, report changed files, tests executed, results and unresolved issues.
- Do not generate large amounts of code beyond the requested scope.
- Do not begin implementation until explicitly instructed.

## Current status

- Workspace contains project documentation, Archify architecture artifacts, and setup files.
- Milestone 1 backend foundation is implemented; remaining application implementation and experiments are pending.
- YuNet and SFace model setup and synthetic feasibility experiments are documented; recognition thresholds remain uncalibrated.
