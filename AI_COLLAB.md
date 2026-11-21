# AI Collaboration Record

## Scope and status

Codex has supported project setup and repository inspection, web-based technical research, architecture discussion, and Archify diagram generation and refinement. Work so far is planning and documentation only. Application implementation and recognition experiments have not started; no project benchmark results or runtime behavior are claimed here.

## Decisions recorded

1. **Offline and CPU-first:** The application is intended to run locally on Windows without cloud recognition APIs, runtime network requests, or model downloads. AI tools may be used only to create practice data, not as part of the application.
2. **Proposed stack:** Python 3.12, FastAPI, React + Vite, Tailwind CSS, NumPy, Pydantic, Pillow where useful, and Pytest + HTTPX. The UI component library remains undecided.
3. **Recognition candidates:** OpenCV YuNet for detection and OpenCV SFace for embeddings were selected as candidates. Codex researched and compared them with InsightFace/ArcFace, dlib/`face_recognition`, and DeepFace, including installation, offline use, benchmark claims, and model licensing. Published figures were treated as protocol-specific context, not as results for this project.
4. **Folder-independent gallery:** A user-selected reference folder supplies the identities through filenames. Replacing the folder should rebuild the gallery without identity-specific code changes or retraining. One reference image per identity is the current design.
5. **Session-only embedding storage:** Reference embeddings are proposed to remain in process memory, scoped to a gallery session. Persistent embedding storage and an NPZ+JSON cache are deferred; reconsider them only if experiments show a clear need and the change is explicitly approved.
6. **Separated recognition responsibilities:** The proposed modular monolith separates request handling, gallery building, detection, alignment, embedding extraction, matching, unknown rejection, and annotation. Matching uses cosine similarity; known/unknown/uncertain thresholds and ambiguity handling remain experimental.
7. **Privacy and evidence:** Personal photos, biometric embeddings, secrets, and generated personal data must not be committed. Features and model performance will be described as implemented or validated only after evidence exists.

These decisions are summarized from the actual conversation and are not a claim that the proposed components have been implemented.

## Conversation excerpts

**Excerpt 1 — model choice**

> Codex: “For this offline, Windows-first project, I recommend YuNet for detection and SFace for embeddings, with matching implemented locally using NumPy cosine similarity and a threshold you validate on representative data.”
>
> User: “The proposed technology stack looks reasonable.”

**Excerpt 2 — embedding persistence**

> User: “Keep gallery embeddings in process memory, scoped to a session. Do not implement persistent embedding storage or an NPZ+JSON cache initially. Reconsider persistence only if experiments demonstrate a clear need and we explicitly approve the change.”
>
> Codex: “Updated only `AGENTS.md`. Replaced the NPZ+JSON storage instruction with: ‘Keep gallery embeddings in process memory, scoped to a session; do not implement persistent embedding storage or an NPZ + JSON cache initially. Reconsider persistence only if experiments demonstrate a clear need and it is explicitly approved.’ No other files were modified.”

## Completed and pending work

- **Completed:** inspected the project instructions and documentation; researched offline recognition options; discussed and recorded the proposed architecture; generated and refined the Archify diagram and `ARCHITECTURE.md`; created `README.md` and `.gitignore`; updated the embedding-storage instruction in `AGENTS.md`.
- **Not started:** application implementation, practice-data creation, controlled model experiments, threshold calibration, and clean-machine offline runtime validation. YuNet and SFace remain unvalidated candidates.
- **Transcript evidence:** This file is a concise summary with two genuine excerpts, not a full conversation export. The assignment brief requests the full AI chat history; an actual export or share link still needs to be added before submission. No prompts, responses, or experiment results have been fabricated to fill that gap.

## Documentation consistency items

At the time this file was created, `README.md` still described the old NPZ+JSON choice as appearing in `AGENTS.md`, although `AGENTS.md` had since been updated to session-only in-memory storage. `AGENTS.md` also still says the workspace was confirmed empty. Both existing files were left unchanged under the instruction to create only this file.
