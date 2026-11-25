# Research and Experimental Results

## Scope and status

This document preserves the recorded synthetic feasibility measurements and model provenance, then summarizes the implementation choices now present in the repository. Measurements below apply only to the stated images, settings, and laptop. They are not general recognition accuracy, demographic performance, or threshold-calibration results.

The project now has a local FastAPI backend and React frontend, including gallery creation, image analysis, experimental matching, unknown rejection, batch assignment, annotation, and tests. The recognition policy remains explicitly experimental. Calibrated readiness is always false; `0.40` similarity and `0.10` margin are configurable sample values, not validated operating thresholds.

Photographs, annotated images, raw embedding vectors, and detailed person-level reports are kept out of this public document and repository. Local experiment outputs are under ignored `outputs/`.

## Environment and hardware

| Item | Recorded configuration |
|---|---|
| OS | Windows 11, AMD64 |
| Python | 3.12.6, project virtual environment |
| OpenCV | `opencv-python-headless` 5.1.0.95 (`cv2` 5.1.0) |
| NumPy | 2.5.5 |
| Processor information available to the experiment | Intel64 Family 6, Model 186, Stepping 3; 12 logical processors reported by Windows |
| Inference configuration | OpenCV DNN backend; CPU target requested; OpenCL disabled; CUDA-enabled device count 0 |

OpenCV printed `Targets are not supported by the new graph engine for now` when YuNet or SFace was created. The CPU target was requested, OpenCL was disabled, and no CUDA devices were present. No per-operation hardware profiler was used; timings below are local feasibility measurements, not a hardware benchmark.

## Model provenance and licences

| Model | Local file | Pinned OpenCV Zoo revision | Verification | Licence and attribution |
|---|---|---|---|---|
| YuNet dynamic-input face detector | `models/face_detection_yunet_2026may.onnx` | `47534e27c9851bb1128ccc0102f1145e27f23f98` | 229,738 bytes; SHA-256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`; verified not an LFS pointer or HTML response; loaded by `FaceDetectorYN.create()` | OpenCV Zoo model-directory MIT licence in `models/LICENSE`; pinned source and attribution in `models/ATTRIBUTION.txt`. The Zoo README links the original YuNet ONNX source and paper citation. |
| SFace FP32 face feature model | `models/sface/face_recognition_sface_2021dec.onnx` | `48b10fa505aeb51fbd127fc60a5c660040e823b7` | 38,696,353 bytes; SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`; verified not an LFS pointer or HTML response; loaded by `FaceRecognizerSF.create()` | OpenCV Zoo SFace directory declares Apache License 2.0; `models/sface/LICENSE` and `models/sface/ATTRIBUTION.txt` are kept beside the model. Training-data provenance was not independently established. |

Pinned source pages: [YuNet in OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet) and [SFace in OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface). The licence statements above refer to those pinned model directories and the retained local notices; the SFace notice does not independently establish training-data provenance.

## YuNet detection feasibility

The first synthetic YuNet run used `face_detection_yunet_2026may.onnx`, score threshold `0.90`, NMS threshold `0.30`, `top_k=5000`, OpenCV DNN, and each image's actual dimensions through `FaceDetectorYN.setInputSize()`.

| Image | Dimensions | Faces at 0.90 | Detection time | Detection detail |
|---|---:|---:|---:|---|
| `Person_01.png` | 896 × 1200 | 1 | 0.033116 s | Box `(253.83, 233.37, 388.48, 526.96)`; confidence `0.909253` |
| `Person_02.png` | 896 × 1200 | 1 | 0.026927 s | Box `(286.91, 213.64, 327.00, 454.72)`; confidence `0.922488` |
| `Person_03.png` | 896 × 1200 | 1 | 0.026172 s | Box `(289.97, 237.71, 330.32, 472.52)`; confidence `0.923090` |
| `Person_04.png` | 896 × 1200 | 1 | 0.029022 s | Box `(284.85, 270.91, 343.55, 480.82)`; confidence `0.918187` |
| `Person_05.png` | 928 × 1136 | 0 | 0.026302 s | Miss at score threshold 0.90 |
| `group_01.png` | 1376 × 768 | 8 | 0.028813 s | Eight detections; candidate comparisons are listed below |

The first run total was `0.697331 s`, including model loading, image reading, detection, annotation, and local report writing. Annotated copies were written only under ignored `outputs/yunet_experiment_1/`.

### Score-threshold sensitivity

NMS `0.30`, `top_k=5000`, and actual image dimensions were held fixed.

| Score threshold | Person_01 | Person_02 | Person_03 | Person_04 | Person_05 | Group |
|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 1 | 1 | 1 | 1 | 0 | 8 |
| 0.85 | 1 | 1 | 1 | 1 | 1 | 8 |
| 0.80 | 1 | 1 | 1 | 1 | 1 | 8 |
| 0.75 | 1 | 1 | 1 | 1 | 1 | 8 |
| 0.70 | 1 | 1 | 1 | 1 | 1 | 8 |

`Person_05.png` was recovered at `0.85` with confidence `0.890300` and box `(257.92, 193.61, 409.87, 551.19)`. Lowering the threshold further did not change counts. All eight group detections remained present with unchanged boxes and scores at the five tested thresholds. No count-based extra detections appeared in this small sample; this does not establish a false-positive rate. The provisional YuNet detection score used in subsequent implementation is `0.85`, separate from the uncalibrated matching threshold.

The sweep took `3.294436 s` total, including setup, image I/O, annotation, and report writing. Per-threshold details and annotated copies remain in ignored `outputs/yunet_threshold_experiment/`.

## SFace loading and embedding smoke test

`cv2.FaceRecognizerSF`, `.create()`, `.alignCrop()`, `.feature()`, and `.match()` were available in the recorded environment. The pinned SFace model loaded through `FaceRecognizerSF.create()`. The smoke test used `Person_01.png` and the highest-confidence YuNet detection in `group_01.png`.

| Image | YuNet faces | Alignment | Feature result | Raw feature L2 norm | Detect / align / feature time |
|---|---:|---|---|---:|---|
| `Person_01.png` | 1 | Succeeded; 112 × 112 × 3 crop | `(1, 128)`, `float32`, finite | 10.922237 | 0.030498 / 0.000292 / 0.023562 s |
| `group_01.png` | 8 | Succeeded; 112 × 112 × 3 crop | `(1, 128)`, `float32`, finite | 10.804312 | 0.026419 / 0.000081 / 0.008498 s |

Both models were created in `0.093976 s`; this setup time is separate from per-image timings. The optional cosine calculation was `0.378541`, used only as a technical sanity check, not an identity result. Raw feature norms show why the application explicitly L2-normalizes features before storage or comparison. No wrapper default cosine threshold was adopted.

## SFace 8 × 5 group-to-reference similarity experiment

YuNet used score threshold `0.85`, NMS `0.30`, and `top_k=5000`. It detected one face per reference portrait and eight in the group photo. SFace produced 13 finite `(1, 128)` `float32` features; all were explicitly L2-normalized to unit norm within `1e-6`. Vectors remained in memory and were not saved.

Rows preserve YuNet order; columns are `Person_01` through `Person_05`. Similarities are measurements, not identity probabilities.

| YuNet group face | Person_01 | Person_02 | Person_03 | Person_04 | Person_05 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.378541 | 0.235660 | 0.172197 | 0.627533 | 0.180171 |
| 2 | 0.243496 | 0.466316 | 0.271561 | 0.204067 | 0.141512 |
| 3 | 0.257092 | 0.373448 | 0.317605 | 0.417879 | 0.216032 |
| 4 | 0.288969 | 0.138561 | 0.248765 | 0.277335 | 0.227411 |
| 5 | 0.175780 | 0.151661 | 0.196008 | 0.209204 | 0.799442 |
| 6 | 0.250115 | 0.089164 | 0.315190 | 0.242553 | 0.249944 |
| 7 | 0.226000 | 0.312134 | 0.236594 | 0.249197 | 0.249969 |
| 8 | 0.361694 | 0.181884 | 0.228313 | 0.083645 | 0.264100 |

These are candidate rankings only; no acceptance threshold was applied in that experiment. Boxes are `(x, y, width, height)` in original coordinates.

| Face | YuNet box | Detector score | Top candidate (score) | Runner-up (score) | Margin |
|---:|---|---:|---|---|---:|
| 1 | `(167.94, 161.80, 100.60, 141.13)` | 0.952730 | Person_04 (0.627533) | Person_01 (0.378541) | 0.248992 |
| 2 | `(405.67, 301.67, 102.24, 129.27)` | 0.946424 | Person_02 (0.466316) | Person_03 (0.271561) | 0.194755 |
| 3 | `(767.27, 276.64, 97.25, 128.52)` | 0.941202 | Person_04 (0.417879) | Person_02 (0.373448) | 0.044431 |
| 4 | `(1148.43, 277.87, 96.42, 130.73)` | 0.940281 | Person_01 (0.288969) | Person_04 (0.277335) | 0.011634 |
| 5 | `(1000.09, 146.39, 91.59, 127.68)` | 0.936402 | Person_05 (0.799442) | Person_04 (0.209204) | 0.590237 |
| 6 | `(427.45, 120.48, 82.79, 112.14)` | 0.936112 | Person_03 (0.315190) | Person_01 (0.250115) | 0.065075 |
| 7 | `(611.54, 224.38, 84.55, 109.28)` | 0.932517 | Person_02 (0.312134) | Person_05 (0.249969) | 0.062165 |
| 8 | `(840.73, 121.05, 88.76, 122.71)` | 0.916743 | Person_01 (0.361694) | Person_05 (0.264100) | 0.097594 |

| Timing stage | Time |
|---|---:|
| Loading both models | 0.090573 s |
| Detection across six images | 0.171756 s |
| Alignment across 13 faces | 0.001806 s |
| SFace feature extraction across 13 faces | 0.130866 s |
| L2 normalization across 13 embeddings | 0.002467 s |
| Similarity matrix and ranking | 0.000288 s |

## 5 × 5 reference-gallery consistency experiment

YuNet detected exactly one face in each of the five references. All SFace features were finite, `(1, 128)`, and unit-normalized within `1e-6`.

|  | Person_01 | Person_02 | Person_03 | Person_04 | Person_05 |
|---|---:|---:|---:|---:|---:|
| **Person_01** | **1.000000** | 0.334059 | 0.374297 | 0.414417 | 0.323260 |
| **Person_02** | 0.334059 | **0.99999994** | 0.372676 | 0.529392 | 0.197930 |
| **Person_03** | 0.374297 | 0.372676 | **1.000000** | 0.290111 | 0.242067 |
| **Person_04** | 0.414417 | 0.529392 | 0.290111 | **0.99999994** | 0.228862 |
| **Person_05** | 0.323260 | 0.197930 | 0.242067 | 0.228862 | **1.000000** |

The ten distinct-identity pairs ranked by similarity were:

| Rank | Pair | Cosine similarity |
|---:|---|---:|
| 1 | Person_02 – Person_04 | 0.529392 |
| 2 | Person_01 – Person_04 | 0.414417 |
| 3 | Person_01 – Person_03 | 0.374297 |
| 4 | Person_02 – Person_03 | 0.372676 |
| 5 | Person_01 – Person_02 | 0.334059 |
| 6 | Person_01 – Person_05 | 0.323260 |
| 7 | Person_03 – Person_04 | 0.290111 |
| 8 | Person_03 – Person_05 | 0.242067 |
| 9 | Person_04 – Person_05 | 0.228862 |
| 10 | Person_02 – Person_05 | 0.197930 |

The off-diagonal values span about `0.198` to `0.529`, with Person_02/Person_04 the most similar pair. One reference image per identity gives no within-person variation and cannot establish real-person separation or calibrate a threshold.

| Timing stage | Time |
|---|---:|
| Loading both models | 0.105594 s |
| Detection across five images | 0.160795 s |
| Alignment across five faces | 0.008446 s |
| SFace feature extraction across five embeddings | 0.081058 s |
| L2 normalization across five embeddings | 0.002032 s |
| Similarity matrix calculation | 0.002056 s |

## Implementation choices and alternatives

### Selected pipeline

- **YuNet** is used through OpenCV's `FaceDetectorYN` API with a local, pinned OpenCV Zoo ONNX model. The implementation uses a provisional detector score threshold of `0.85`, NMS `0.30`, and `top_k=5000`. The synthetic sweep recovered the fifth reference at `0.85`; that six-image result is not a general detector evaluation.
- **SFace** is used through OpenCV's `FaceRecognizerSF` API. The implementation passes the YuNet detection row and the same working image to `alignCrop()`, then extracts a 128-value feature and validates/normalizes it as finite `float32`.
- **Cosine similarity** is the selected comparison because gallery/query features are L2-normalized; the dot product is then a compact score for ranking. The score is not a probability and does not itself indicate an identity.
- **Similarity plus top-two margin** provides explicit open-set rejection: a candidate must clear the absolute threshold and be sufficiently separated from alternatives. A large margin alone cannot establish a match. The current configured pair may be used only as experimental policy after explicit opt-in; no calibration is claimed.
- **One-to-one batch assignment** extends per-face ranking to a whole photo. It prevents duplicate assignment of an identity, considers valid alternative identities, maximizes the count of policy-valid assignments and then total score, and rejects tied global outcomes as Unknown. This behavior is implemented and tested, but its identity accuracy is not established by the original synthetic benchmark.

Comparative face-recognition research in the project conversation considered OpenCV YuNet/SFace, InsightFace/ArcFace, dlib/`face_recognition`, and DeepFace. Those alternatives were not benchmarked head-to-head in this repository. The recorded implementation rationale for YuNet/SFace is their OpenCV integration, local ONNX model files, CPU-capable APIs, and pinned local license/attribution notices—not a claim that they outperform those alternatives. No comparative accuracy or install-cost result is asserted here.

### Current runtime configuration

`backend/.env.example` documents `OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD=0.40`, `OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN=0.10`, and `OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED=false`. These values are explicitly experimental, not calibrated. Both thresholds must be present together. The local `backend/.env`, if created, is loaded automatically with process-environment precedence; explicit injected environment mappings bypass dotenv. YuNet/SFace load from local model paths during FastAPI lifespan, on OpenCV's CPU backend/target, with OpenCL disabled. The frontend is locally bundled and defaults to the loopback backend; runtime network use outside that local API is not part of the design.

## Real-person validation and interpretation limits

The project owner reports the current local validation collection contains 34 reference identities and 29 group photographs. Earlier local diagnostic and similarity reports in ignored `outputs/` concern a 21-reference/six-group subset, and a separate group-seven experiment. Those reports contain exploratory predictions but do not constitute complete independent ground truth or a complete evaluation of the current collection. No aggregate accuracy, false-accept, false-reject, or calibration metric is published here. Current counts are user-provided; they have not been independently reconstructed into a public evaluation manifest.

The project history contains manually supplied ground-truth corrections for some earlier groups and a confirmed known-to-known confusion, plus unknown faces that received candidate labels. This is evidence that nearest-candidate ranking alone can misidentify known people and that Unknown rejection matters; it is not a complete or independently held-out evaluation. No experimental prediction is treated as ground truth.

The current default experimental values `0.40` and `0.10` have not been selected from a documented disjoint calibration/evaluation split. They must not be presented as validated operating points. Recognition remains marked `calibrated=false` and `recognition_ready=false` even when experimental execution is explicitly enabled.

## Limitations and next validation work

- The recorded benchmark dataset is five synthetic individual portraits and one synthetic group image. It cannot establish unseen-data performance, demographics, difficult capture robustness, or operational error rates.
- The older real-person pilot had only one reference image per identity and a small set of scenes; image quality, pose, lighting, and duplicate detections can affect similarity. The current 34/29 collection still needs complete verified per-face labels and a protocol that separates calibration from final evaluation.
- Use independently labeled known and unknown people, multiple scenes/conditions, and a held-out set. Report known identification accuracy, known false rejection, unknown false acceptance, unknown rejection, and per-face failure counts with denominators. Do not tune and report on the same faces as independent validation.
- The `0.85` YuNet score threshold is provisional. Image resizing to a maximum detection dimension of 1200 was explored for difficult real-person references; that feasibility comparison is not evidence of improved recognition accuracy.
- Experimental thresholds, one-to-one conflict policy, and the full end-to-end UI/API remain subject to broader validation. A clean-machine test with networking disabled and packaged deployment verification are also outstanding.
- Model checksum, loading, license files, and source revisions were verified as listed above. Training-data provenance for SFace and general license implications beyond the retained upstream notices were not independently established.

Raw embeddings, original photographs, annotated photos, and local reports remain excluded from Git. This document records only experiment results and implementation facts available for verification; it does not claim general accuracy.
