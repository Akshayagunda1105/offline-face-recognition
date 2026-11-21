# AI Collaboration Record

## Scope and status

Codex has supported project setup and repository inspection, web-based technical research, architecture decisions, Archify diagram generation and refinement, and execution and documentation of local synthetic feasibility experiments. Application implementation has not started. Experiment outcomes below are limited to the supplied small synthetic dataset; they are not general accuracy claims.

## Decisions recorded

1. **Offline and CPU-first:** The application is intended to run locally on Windows without cloud recognition APIs, runtime network requests, or model downloads. AI tools may be used only to create practice data, not as part of the application.
2. **Proposed stack:** Python 3.12, FastAPI, React + Vite, Tailwind CSS, NumPy, Pydantic, Pillow where useful, and Pytest + HTTPX. The UI component library remains undecided.
3. **Recognition candidates:** OpenCV Zoo YuNet for detection and SFace for embeddings were selected. Both pinned model files are local assets with their respective MIT and Apache 2.0 notices. YuNet score threshold `0.85` is provisional based on synthetic threshold sensitivity; it is not a final matching threshold.
4. **Folder-independent gallery:** A user-selected reference folder supplies the identities through filenames. Replacing the folder should rebuild the gallery without identity-specific code changes or retraining. One reference image per identity is the current design.
5. **Session-only embedding storage:** Reference embeddings are proposed to remain in process memory, scoped to a gallery session. Persistent embedding storage and an NPZ+JSON cache are deferred; reconsider them only if experiments show a clear need and the change is explicitly approved.
6. **Separated recognition responsibilities:** The proposed modular monolith separates request handling, gallery building, detection, alignment, embedding extraction, matching, unknown rejection, and annotation. SFace feature vectors are explicitly L2-normalized before comparison; cosine similarity ranks candidates. Known/unknown/uncertain thresholds and ambiguity handling remain uncalibrated.
7. **Privacy and evidence:** Personal photos, biometric embeddings, secrets, and generated personal data must not be committed. Features and model performance will be described as implemented or validated only after evidence exists.

These decisions are summarized from the actual conversation and are not a claim that the proposed components have been implemented.

## Verified conversation excerpt

The user-authored wording below is verbatim from the available conversation record. Other excerpts previously included here could not be verified against an accessible transcript and have been removed rather than reconstructed.

> User: “Keep gallery embeddings in process memory, scoped to a session. Do not implement persistent embedding storage or an NPZ+JSON cache initially. Reconsider persistence only if experiments demonstrate a clear need and we explicitly approve the change.”

## Completed and pending work

- **Completed:** project setup and repository inspection; offline-model research; approved architecture and Archify diagram; pinned YuNet and SFace model acquisition and verification; YuNet detection and threshold-sensitivity experiments; SFace load/embedding smoke test; 8 × 5 group-to-reference similarity experiment; and 5 × 5 reference-gallery consistency experiment. Methods and aggregate results are in [RESEARCH.md](RESEARCH.md). The raw embedding vectors and generated/annotated images remain local and ignored.
- **Not started:** application implementation, real-person validation, open-set matching-threshold calibration, and clean-machine offline application validation. The synthetic experiments do not demonstrate general recognition accuracy.
- **Transcript evidence:** The assignment brief requires 2–3 pasted chat excerpts and the full AI chat history. Only the user-authored excerpt above could be verified verbatim from the available conversation record. Add further excerpts and the full history only from an actual transcript export or share link; no missing conversation has been reconstructed.
