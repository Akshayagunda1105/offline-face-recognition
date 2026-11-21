"""Print local YuNet + SFace stage timings without saving biometric data."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.engine import FaceAnalysisEngine
from offline_face_recognition.imaging.decode import decode_image
from offline_face_recognition.models.runtime import ModelRuntime

ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATHS = [
    *(ROOT / "synthetic_dataset" / "reference_gallery" / f"Person_{index:02d}.png" for index in range(1, 6)),
    ROOT / "synthetic_dataset" / "group_photos" / "group_01.png",
]


def _summary(values: list[float]) -> str:
    milliseconds = [value * 1000.0 for value in values]
    ordered = sorted(milliseconds)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return (
        f"n={len(values)}, median={statistics.median(milliseconds):.3f} ms, "
        f"p95={p95:.3f} ms, mean={statistics.fmean(milliseconds):.3f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5, help="Measured passes after one warm-up (default: 5)")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    missing = [path for path in IMAGE_PATHS if not path.is_file()]
    if missing:
        parser.error("The local synthetic pilot dataset is incomplete")

    settings = Settings.from_environment(environ={})
    runtime = ModelRuntime(settings)
    started = time.perf_counter()
    runtime.load()
    model_load_seconds = time.perf_counter() - started
    if not runtime.readiness.models_ready:
        print(f"Model loading failed: {runtime.error_code}")
        runtime.close()
        return 1

    engine = FaceAnalysisEngine(runtime)
    timings: dict[str, list[float]] = {
        "decode": [], "detection": [], "alignment": [], "feature": [], "normalization": []
    }
    face_operations = 0

    def run_pass(record: bool) -> None:
        nonlocal face_operations
        for path in IMAGE_PATHS:
            started_decode = time.perf_counter()
            image = decode_image(path.read_bytes(), path.name, settings)
            if record:
                timings["decode"].append(time.perf_counter() - started_decode)

            started_detection = time.perf_counter()
            detected = engine.detector.detect(image)
            if record:
                timings["detection"].append(time.perf_counter() - started_detection)
                face_operations += len(detected.detections)
            for index, detection in enumerate(detected.detections):
                started_alignment = time.perf_counter()
                crop = engine.embedder.align_crop(image, detection, detection_index=index)
                if record:
                    timings["alignment"].append(time.perf_counter() - started_alignment)

                started_feature = time.perf_counter()
                feature = engine.embedder.extract_feature(crop, detection_index=index)
                if record:
                    timings["feature"].append(time.perf_counter() - started_feature)

                started_normalization = time.perf_counter()
                engine.embedder.normalize_feature(feature, detection_index=index)
                if record:
                    timings["normalization"].append(time.perf_counter() - started_normalization)

    try:
        run_pass(record=False)  # warm-up is excluded from the reported stage distributions
        for _ in range(args.repeat):
            run_pass(record=True)
    finally:
        runtime.close()

    print("Offline CPU face-engine benchmark (no embeddings or images written)")
    print(f"Model load: {model_load_seconds * 1000.0:.3f} ms")
    print(
        f"Image count: {len(IMAGE_PATHS)}; measured passes: {args.repeat}; "
        f"detected-face operations: {face_operations}"
    )
    for stage, values in timings.items():
        print(f"{stage.capitalize()}: {_summary(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
