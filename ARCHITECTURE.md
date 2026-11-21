# Architecture: Proposed Offline Face Recognition

> **Status: proposed, not implemented.** Initial YuNet and SFace feasibility experiments have been completed on a small synthetic dataset. Broader real-person validation, open-set threshold calibration, and application implementation remain pending; these experiments do not establish general recognition accuracy or working application features.

## Diagram

[![Proposed offline face-recognition architecture (light theme)](docs/architecture/preview.png)](.archify/architecture-offline-face-recognition-20261121-0124/offline-face-recognition.html)

Open the image above to view the **interactive standalone Archify diagram**. The proposed Windows CPU runtime loads the built frontend assets and ONNX model files locally; it makes no external network requests at runtime, including cloud API calls or model downloads.

## Components and responsibilities

| Component | Responsibility |
|---|---|
| **React frontend** | Lets the user select a reference folder and group photo, transfers reference image files with their original filenames, and displays both the annotated photo and a separate identification list. |
| **FastAPI backend** | Validates requests and coordinates gallery creation and group-photo recognition. Recognition decisions remain in the engine components rather than the UI. |
| **Gallery Builder** | Validates reference images, derives each identity label from its filename, and creates a gallery for the current session. Invalid or unreadable images and references containing zero or multiple faces are rejected and reported. |
| **YuNet (selected candidate)** | Detects faces and landmarks in reference images and the group photo. Initial synthetic detection and threshold-sensitivity tests are complete; `0.85` is a provisional detection score threshold. Broader validation remains pending. |
| **Face Alignment** | Uses detected landmarks to align reference and group-photo face crops before embedding extraction. |
| **SFace (selected candidate)** | Extracts a 128-dimensional `float32` feature from each aligned crop. The embedding-generation step explicitly L2-normalizes this output before storage or comparison. Initial synthetic extraction and similarity experiments are complete; broader validation remains pending. |
| **In-memory Session Store** | Keeps a gallery-session record in process memory: a session ID, identity-to-normalized-embedding mappings, creation time, and expiry. It is not a database server. |
| **Matcher** | Ranks candidate identities for each group-photo face using cosine similarity against the selected session’s reference embeddings. |
| **Unknown Rejection** | Will make the final decision using an acceptance threshold and ambiguity rule after calibration: **known** when the top candidate clears the threshold and is sufficiently separated from the runner-up; **unknown** when no candidate clears the threshold; **uncertain** when the top score or candidate separation falls in the ambiguity region. Threshold and ambiguity calibration remain pending. |
| **Image Annotator** | Draws bounding boxes and the resulting identity/status labels on a copy of the group photo. |

## Data flow

1. The user selects a reference folder in React. The browser supplies the image files and filenames; React uploads them to FastAPI.
2. FastAPI validates the request and asks the Gallery Builder to create a gallery. The Gallery Builder obtains each identity label from its filename and rejects/reports references with zero or multiple faces, so another folder can be used without identity-specific code or retraining.
3. For each valid reference, YuNet detects a face and landmarks, Face Alignment aligns its crop, SFace extracts a feature, and NumPy L2-normalizes it. The In-memory Session Store holds a session ID, identity-to-normalized-embedding mappings, creation time, and expiry.
4. The user uploads a group photo. FastAPI routes it through YuNet, Face Alignment, SFace, and L2 normalization to detect and embed each usable group-photo face.
5. The Matcher compares each query embedding against the embeddings from that gallery session and ranks candidates by cosine similarity.
6. Unknown Rejection applies an acceptance threshold and ambiguity rule: known for a qualifying, sufficiently separated top candidate; unknown when no candidate qualifies; uncertain when the score or separation is ambiguous. No matching threshold or ambiguity margin has been selected. YuNet's `0.85` detection threshold is provisional and is separate from these matching decisions.
7. The Image Annotator draws the face boxes and outcomes. FastAPI returns the annotated image and a separate results list to React for display.

No identity-specific model training is part of this flow. Selecting a different reference folder rebuilds the session gallery from the supplied filenames and images.

## Trade-offs

- **Chosen:** a modular monolith with separate vision and matching responsibilities. It keeps local setup and debugging manageable while allowing the recognition engine to be tested independently of React and FastAPI.
- **Chosen:** CPU-first local inference with local model files and frontend assets. This supports offline runtime and avoids cloud services and runtime downloads.
- **Chosen:** in-memory embeddings scoped to a gallery session, with no database server or persistent cache initially. This limits biometric retention and suits the small initial gallery; it also means a process restart requires rebuilding the gallery.
- **Given up:** persistent caching and approximate-nearest-neighbor infrastructure such as FAISS. They add storage and operational complexity without a demonstrated need for the initial gallery size.
- **Evidence so far:** local CPU model loading and synthetic detection/embedding/similarity feasibility runs completed; the observed timing and similarity results are summarized in [RESEARCH.md](RESEARCH.md). **Unresolved:** performance and behavior on representative real-person data, face-size and group-photo variation, false-match evaluation, and calibrated known/unknown/uncertain thresholds. The application remains unimplemented and offline application execution has not been verified.

## Privacy notes

Reference embeddings are biometric data. The proposed design keeps them in process memory, scopes them to a gallery session, and avoids persisting reference photos or embeddings. Implementation should also avoid logging images, filenames, face crops, and embeddings; bind the service to localhost; release session memory when no longer needed; and keep personal photos and embeddings out of the public repository. These are design requirements, not implemented controls.
