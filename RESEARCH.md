# Research and Experimental Results

## Scope and status

This log records local feasibility experiments run on the supplied five synthetic reference portraits and one synthetic group photo. The measurements describe these six images and this laptop environment only. They are not general face-recognition accuracy claims. No application code, identity-specific training, final open-set decision, or matching-threshold calibration was performed.

Raw photographs, annotated copies, and embedding vectors are intentionally absent from this document and from the public repository. Local experiment outputs are under the ignored `outputs/` directory.

## Environment and hardware

| Item | Recorded configuration |
|---|---|
| OS | Windows 11, AMD64 |
| Python | 3.12.6, project `.venv` |
| OpenCV | `opencv-python-headless` 5.1.0.95 (`cv2` 5.1.0) |
| NumPy | 2.5.5 |
| Processor information available to the experiment | Intel64 Family 6, Model 186, Stepping 3; 12 logical processors reported by Windows |
| Inference configuration | OpenCV DNN backend; CPU target requested; OpenCL disabled; CUDA-enabled device count 0 |

OpenCV printed `Targets are not supported by the new graph engine for now` when YuNet or SFace was created. The CPU target was requested, OpenCL was disabled, and no CUDA devices were present. No per-operation hardware profiler was used, so timings below are local feasibility timings rather than a hardware benchmark.

## Model provenance and licences

| Model | Local file | Pinned OpenCV Zoo revision | Verification | Licence and attribution |
|---|---|---|---|---|
| YuNet dynamic-input face detector | `models/face_detection_yunet_2026may.onnx` | `47534e27c9851bb1128ccc0102f1145e27f23f98` | 229,738 bytes; SHA-256 `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`; not an LFS pointer or HTML response; loaded successfully by `FaceDetectorYN.create()` | OpenCV Zoo model-directory MIT licence in `models/LICENSE`; attribution and upstream source in `models/ATTRIBUTION.txt`. Zoo README links the original YuNet ONNX source and gives the paper citation. |
| SFace FP32 face feature model | `models/sface/face_recognition_sface_2021dec.onnx` | `48b10fa505aeb51fbd127fc60a5c660040e823b7` | 38,696,353 bytes; SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`; not an LFS pointer or HTML response; loaded successfully by `FaceRecognizerSF.create()` | OpenCV Zoo SFace directory declares Apache License 2.0; `models/sface/LICENSE` and `models/sface/ATTRIBUTION.txt` are retained beside the model. Training-data provenance was not independently established. |

The pinned model source pages are [YuNet in OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet) and [SFace in OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface). Their respective model-directory READMEs declare the licences above. SFace's Apache notice does not by itself document the training datasets or independently resolve their terms.

## YuNet detection feasibility

The YuNet experiment used `face_detection_yunet_2026may.onnx`, score threshold `0.90`, NMS threshold `0.30`, `top_k=5000`, OpenCV DNN, and the actual dimensions of each image through `FaceDetectorYN.setInputSize()`.

| Image | Dimensions | Faces at 0.90 | Detection time | Detection detail |
|---|---:|---:|---:|---|
| `Person_01.png` | 896 × 1200 | 1 | 0.033116 s | Box `(253.83, 233.37, 388.48, 526.96)`; confidence `0.909253` |
| `Person_02.png` | 896 × 1200 | 1 | 0.026927 s | Box `(286.91, 213.64, 327.00, 454.72)`; confidence `0.922488` |
| `Person_03.png` | 896 × 1200 | 1 | 0.026172 s | Box `(289.97, 237.71, 330.32, 472.52)`; confidence `0.923090` |
| `Person_04.png` | 896 × 1200 | 1 | 0.029022 s | Box `(284.85, 270.91, 343.55, 480.82)`; confidence `0.918187` |
| `Person_05.png` | 928 × 1136 | 0 | 0.026302 s | Miss at score threshold 0.90 |
| `group_01.png` | 1376 × 768 | 8 | 0.028813 s | Eight detections; details and boxes are listed with the SFace group-face rows below |

The first run's total was `0.697331 s`, including model loading, image reading, detection, annotation, and writing its local report. Annotated copies were written only under ignored `outputs/yunet_experiment_1/`.

### Score-threshold sensitivity

All other settings were held at NMS `0.30`, `top_k=5000`, and actual image input dimensions.

| Score threshold | Person_01 | Person_02 | Person_03 | Person_04 | Person_05 | Group |
|---:|---:|---:|---:|---:|---:|---:|
| 0.90 | 1 | 1 | 1 | 1 | 0 | 8 |
| 0.85 | 1 | 1 | 1 | 1 | 1 | 8 |
| 0.80 | 1 | 1 | 1 | 1 | 1 | 8 |
| 0.75 | 1 | 1 | 1 | 1 | 1 | 8 |
| 0.70 | 1 | 1 | 1 | 1 | 1 | 8 |

`Person_05.png` was recovered at `0.85` with confidence `0.890300` and box `(257.92, 193.61, 409.87, 551.19)`. Lowering further did not change counts. All eight group-photo detections remained present with unchanged boxes and scores across the five tested thresholds. No count-based extra detections appeared on these images; the small sample cannot establish a false-positive rate. The provisional YuNet detection score threshold for subsequent experiments is `0.85`. This is separate from the uncalibrated face-matching threshold.

The threshold sweep took `3.294436 s` total, including detector setup, image I/O, annotation, and report writing. Its annotated copies and full per-threshold detections remain only in ignored `outputs/yunet_threshold_experiment/`.

## SFace loading and embedding smoke test

In the existing environment, `cv2.FaceRecognizerSF`, `.create()`, `.alignCrop()`, `.feature()`, and `.match()` were available. The pinned SFace model loaded through `FaceRecognizerSF.create()`. Two images were checked: `Person_01.png` and `group_01.png`; the highest-confidence YuNet group detection was selected for the smoke test.

| Image | YuNet faces | Alignment | Feature result | Raw feature L2 norm | Detect / align / feature time |
|---|---:|---|---|---:|---:|
| `Person_01.png` | 1 | Succeeded; 112 × 112 × 3 crop | `(1, 128)`, `float32`, finite | 10.922237 | 0.030498 / 0.000292 / 0.023562 s |
| `group_01.png` | 8 | Succeeded; 112 × 112 × 3 crop | `(1, 128)`, `float32`, finite | 10.804312 | 0.026419 / 0.000081 / 0.008498 s |

Both models were created in `0.093976 s`; this setup time is separate from the per-image stage timings.

The smoke test's one optional cosine calculation was `0.378541` and was only a technical sanity check, not an identity result. The raw feature norms show why the embedding-generation step explicitly L2-normalizes SFace features before storage and comparison. No wrapper default cosine threshold was adopted.

## SFace 8 × 5 group-to-reference similarity experiment

YuNet used score threshold `0.85`, NMS `0.30`, and `top_k=5000`. It detected one face in each reference portrait and eight in the group photo. SFace produced 13 finite `(1, 128)` `float32` features; all were explicitly L2-normalized and had unit norm within `1e-6`. Embeddings remained in memory and were not saved.

The matrix rows preserve YuNet's returned order for the group photo. Columns are `Person_01` through `Person_05`. Values are cosine similarities, not identity probabilities.

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

The top entries below are **candidate rankings only**. No known/unknown/uncertain result or acceptance threshold was applied. Boxes use `(x, y, width, height)` in original-image coordinates; detector confidence is YuNet's score.

| Face | YuNet box | Detector confidence | Top candidate (score) | Second candidate (score) | Top-two margin |
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

The five reference images were processed in `Person_01` through `Person_05` order. YuNet detected exactly one face in each; all SFace features were finite, shape `(1, 128)`, and L2-normalized to unit norm within `1e-6`.

|  | Person_01 | Person_02 | Person_03 | Person_04 | Person_05 |
|---|---:|---:|---:|---:|---:|
| **Person_01** | **1.000000** | 0.334059 | 0.374297 | 0.414417 | 0.323260 |
| **Person_02** | 0.334059 | **0.99999994** | 0.372676 | 0.529392 | 0.197930 |
| **Person_03** | 0.374297 | 0.372676 | **1.000000** | 0.290111 | 0.242067 |
| **Person_04** | 0.414417 | 0.529392 | 0.290111 | **0.99999994** | 0.228862 |
| **Person_05** | 0.323260 | 0.197930 | 0.242067 | 0.228862 | **1.000000** |

The ten distinct-identity pairs ranked from most to least similar were:

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

These five single-image vectors are numerically distinct from one another in this sample, with pairwise off-diagonal values from about `0.198` to `0.529`. Person_02 and Person_04 are the closest pair. One image per identity provides no within-identity variation, so this result cannot establish real-person identity separation or set a matching threshold.

| Timing stage | Time |
|---|---:|
| Loading both models | 0.105594 s |
| Detection across five images | 0.160795 s |
| Alignment across five faces | 0.008446 s |
| SFace feature extraction across five embeddings | 0.081058 s |
| L2 normalization across five embeddings | 0.002032 s |
| Similarity matrix calculation | 0.002056 s |

## Limitations and next validation work

- The dataset is small and synthetic: five individual portraits and one group image. It does not represent the judges' unseen data or establish general accuracy, demographic performance, or robustness.
- There is only one reference image per identity, so genuine within-person variation could not be measured. Similarity scores are not identity probabilities.
- The `0.85` YuNet detection score is provisional and was selected only to recover the fifth synthetic reference while preserving the observed eight group detections in this sample. More representative tests are needed.
- No open-set matching threshold, ambiguity margin, or known/unknown/uncertain rule was calibrated. No face in the group photo was confirmed as an identity.
- The full application is not implemented. Local model loading and image experiments do not validate the React/FastAPI flow, privacy controls, clean-machine setup, or offline runtime of a packaged application.
- Biometric vectors and generated/annotated images are excluded from this repository. The local reports under `outputs/` are ignored and are not source artifacts.
