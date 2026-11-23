"""
test_pipeline.py
----------------
End-to-end pipeline tester for RoadSight AI.

Tests the full flow:
    Input image → YOLOv8 detection → Bounding box extraction
    → Severity assessment → Annotated image + JSON results

Usage
-----
    python test_pipeline.py --image images/image.jpeg
    python test_pipeline.py --image images/image.jpeg --conf 0.25 --save-dir tests
    python test_pipeline.py --help

Output
------
    tests/annotated_<basename>.jpg  — image with bounding boxes and severity labels
    tests/results.json              — full structured results

Notes
-----
- Does NOT modify models/best.pt or any module file.
- Severity is assessed by modules/severity.py — no rules are re-implemented here.
- YOLO confidence is NOT used as a severity indicator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional OpenCV import — used for annotation only
# ---------------------------------------------------------------------------
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[WARNING] opencv-python not found. Annotated image will not be saved.")

# ---------------------------------------------------------------------------
# Validate project root and module path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from modules.severity import assess_all_severities
except ImportError as e:
    sys.exit(
        f"[ERROR] Cannot import severity engine from modules/severity.py.\n"
        f"        Make sure you are running from the project root.\n"
        f"        Detail: {e}"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH        = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_SAVE_DIR  = PROJECT_ROOT / "tests"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

# Severity → BGR colour for OpenCV annotation
SEVERITY_COLOURS = {
    "High":         (0,   0,   220),   # Red
    "Medium":       (0,   165, 255),   # Orange
    "Low":          (0,   200, 0),     # Green
    "Not Assessed": (160, 160, 160),   # Grey
}

# Severity → compact label shown on the annotated image
SEVERITY_SHORT = {
    "High":         "HIGH",
    "Medium":       "MED",
    "Low":          "LOW",
    "Not Assessed": "N/A",
}


# ===========================================================================
# Step 1 — Image validation
# ===========================================================================

def validate_image(image_path: str) -> Path:
    """
    Validate that the supplied path points to a supported, readable image.

    Returns the resolved Path on success; exits with an error message otherwise.
    """
    p = Path(image_path).resolve()

    if not p.exists():
        sys.exit(f"[ERROR] Image not found: {p}")

    if p.suffix.lower() not in SUPPORTED_FORMATS:
        sys.exit(
            f"[ERROR] Unsupported file format '{p.suffix}'. "
            f"Accepted formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    if _CV2_AVAILABLE:
        test = cv2.imread(str(p))
        if test is None:
            sys.exit(f"[ERROR] OpenCV could not open the image: {p}")

    return p


# ===========================================================================
# Step 2 — YOLOv8 inference
# ===========================================================================

def run_yolo_inference(image_path: Path, conf_threshold: float) -> tuple:
    """
    Load models/best.pt and run inference on the supplied image.

    Returns
    -------
    (results, image_width, image_height, model_names)
        results       — raw Ultralytics Results object list
        image_width   — int
        image_height  — int
        model_names   — dict {class_id: class_name}
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "[ERROR] 'ultralytics' package is not installed.\n"
            "        Install it with: pip install ultralytics"
        )

    if not MODEL_PATH.exists():
        sys.exit(f"[ERROR] Trained model not found at: {MODEL_PATH}")

    print(f"[INFO]  Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print(f"[INFO]  Running inference on: {image_path.name}  (conf ≥ {conf_threshold})")
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        save=False,     # we do our own annotated save below
        verbose=False,
    )

    # orig_shape is (H, W) in Ultralytics
    img_h, img_w = results[0].orig_shape
    return results, int(img_w), int(img_h), model.names


# ===========================================================================
# Step 3 — Extract structured detections from YOLO results
# ===========================================================================

def extract_detections(results, model_names: dict) -> list[dict]:
    """
    Convert raw Ultralytics Results boxes into plain Python dicts compatible
    with the severity engine.

    Each dict contains:
        class_id, class_name, confidence, bbox {x1,y1,x2,y2},
        bbox_width, bbox_height, bbox_area
    """
    detections = []

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = model_names.get(class_id, f"class_{class_id}")
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            bbox_width  = x2 - x1
            bbox_height = y2 - y1
            bbox_area   = bbox_width * bbox_height

            detections.append({
                "class_id":   class_id,
                "class_name": class_name,
                "confidence": round(confidence, 6),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
                "bbox_width":  round(bbox_width,  2),
                "bbox_height": round(bbox_height, 2),
                "bbox_area":   round(bbox_area,   2),
            })

    return detections


# ===========================================================================
# Step 4 — Severity assessment (delegates entirely to modules/severity.py)
# ===========================================================================

def run_severity_assessment(
    detections: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Pass each detection to the severity engine and merge the result back
    with the original detection data.

    Returns a list of fully-enriched result dicts.
    """
    if not detections:
        return []

    severity_results = assess_all_severities(detections, image_width, image_height)

    # Merge: preserve class_id, bbox_width, bbox_height, bbox_area from the
    # raw detection (the severity engine returns its own visual_metrics subset).
    merged = []
    for raw, sev in zip(detections, severity_results):
        entry = dict(raw)           # start from raw detection
        entry.update(sev)           # overlay severity fields (class_name, bbox,
                                    # visual_metrics, severity, severity_method,
                                    # severity_note)
        merged.append(entry)

    return merged


# ===========================================================================
# Step 5 — Terminal output
# ===========================================================================

def _divider(char: str = "-", width: int = 50) -> str:
    return char * width


def print_results(
    image_path: Path,
    image_width: int,
    image_height: int,
    results: list[dict],
) -> None:
    """Print a human-readable summary of all detections to stdout."""

    W = 50
    print()
    print("=" * W)
    print("        ROADSIGHT AI - TEST RESULTS")
    print("=" * W)
    print()
    print(f"Image              : {image_path.name}")
    print(f"Image Dimensions   : {image_width} x {image_height}")
    print(f"Total Defects Found: {len(results)}")
    print()

    if not results:
        print("  [No defects detected above the confidence threshold.]")
        print()

    for idx, r in enumerate(results, start=1):
        print(_divider("-", W))
        print(f"DETECTION {idx}")
        print(_divider("-", W))

        vm = r.get("visual_metrics", {})

        print(f"Defect Class  : {r.get('class_name', 'Unknown')}")
        print(f"Confidence    : {r.get('confidence', 0):.4f}")
        print()
        print("Bounding Box:")
        bbox = r.get("bbox", {})
        print(f"  X1: {bbox.get('x1', 0):.2f}    Y1: {bbox.get('y1', 0):.2f}")
        print(f"  X2: {bbox.get('x2', 0):.2f}    Y2: {bbox.get('y2', 0):.2f}")
        print()
        print("Visual Metrics:")
        print(f"  Width       : {vm.get('bbox_width',  r.get('bbox_width',  'N/A'))}")
        print(f"  Height      : {vm.get('bbox_height', r.get('bbox_height', 'N/A'))}")
        print(f"  Area        : {vm.get('bbox_area',   r.get('bbox_area',   'N/A'))}")
        print(f"  Area Ratio  : {vm.get('area_ratio',  'N/A')} % of image")
        print(f"  Aspect Ratio: {vm.get('aspect_ratio','N/A')}")
        print()
        print(f"Severity      : {r.get('severity', 'Unknown').upper()}")
        print(f"Method        : {r.get('severity_method', 'Unknown')}")
        print()
        note = r.get("severity_note", "")
        if note:
            wrapped = textwrap.fill(note, width=W - 2, initial_indent="  ", subsequent_indent="  ")
            print("Reason:")
            print(wrapped)
        print()

    # --- Summary ---
    print("=" * W)
    print("SUMMARY")
    print("=" * W)
    print()
    print(f"Total Detections: {len(results)}")
    print()

    class_counts: dict[str, int] = defaultdict(int)
    sev_counts:   dict[str, int] = defaultdict(int)
    not_assessed_entries = []

    for r in results:
        class_counts[r.get("class_name", "Unknown")] += 1
        sev = r.get("severity", "Unknown")
        sev_counts[sev] += 1
        if sev == "Not Assessed":
            not_assessed_entries.append(r.get("class_name", "Unknown"))

    if class_counts:
        print("Detections by Class:")
        for cls, cnt in sorted(class_counts.items()):
            print(f"  {cls}: {cnt}")
        print()

    print("Severity Distribution:")
    for label in ["High", "Medium", "Low", "Not Assessed"]:
        print(f"  {label:12s}: {sev_counts.get(label, 0)}")
    print()

    if not_assessed_entries:
        print("Not Assessed Classes:")
        for cls in not_assessed_entries:
            print(f"  - {cls}")
        print()

    print("=" * W)
    print()


# ===========================================================================
# Step 6 — Annotated image
# ===========================================================================

def save_annotated_image(
    image_path: Path,
    results: list[dict],
    save_dir: Path,
) -> Path | None:
    """
    Draw bounding boxes with class name, confidence, and severity label on
    the original image using OpenCV.  Returns the saved output path, or None
    when OpenCV is unavailable.

    Colour coding:
        High         → Red
        Medium       → Orange
        Low          → Green
        Not Assessed → Grey
    """
    if not _CV2_AVAILABLE:
        print("[WARNING] Annotated image skipped — opencv-python not available.")
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARNING] Could not read image for annotation: {image_path}")
        return None

    img_h, img_w = image.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(img_w, img_h) / 1000)
    thickness  = max(1, int(min(img_w, img_h) / 400))

    for r in results:
        severity = r.get("severity", "Not Assessed")
        colour   = SEVERITY_COLOURS.get(severity, (160, 160, 160))
        bbox     = r.get("bbox", {})

        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        # Skip boxes that are still invalid after extraction
        if x2 <= x1 or y2 <= y1:
            continue

        # Bounding box rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness + 1)

        # Label: "ClassName  conf  [SEV]"
        class_name  = r.get("class_name", "Unknown")
        confidence  = r.get("confidence", 0.0)
        sev_short   = SEVERITY_SHORT.get(severity, "?")
        label       = f"{class_name}  {confidence:.2f}  [{sev_short}]"

        # Background rectangle for legibility
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 5, text_h + baseline + 2)
        cv2.rectangle(
            image,
            (x1, label_y - text_h - baseline - 2),
            (x1 + text_w + 4, label_y + 2),
            colour,
            cv2.FILLED,
        )

        # Text — white on coloured background
        cv2.putText(
            image, label,
            (x1 + 2, label_y - baseline),
            font, font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Severity legend — bottom-left corner
    legend_items = [
        ("HIGH",  SEVERITY_COLOURS["High"]),
        ("MED",   SEVERITY_COLOURS["Medium"]),
        ("LOW",   SEVERITY_COLOURS["Low"]),
        ("N/A",   SEVERITY_COLOURS["Not Assessed"]),
    ]
    legend_x, legend_y = 10, img_h - 10
    for label_text, col in reversed(legend_items):
        (lw, lh), lb = cv2.getTextSize(label_text, font, font_scale, thickness)
        legend_y -= lh + lb + 6
        cv2.rectangle(image, (legend_x, legend_y - lh), (legend_x + lw + 4, legend_y + lb + 2), col, cv2.FILLED)
        cv2.putText(image, label_text, (legend_x + 2, legend_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Save
    stem       = image_path.stem
    out_name   = f"annotated_{stem}.jpg"
    out_path   = save_dir / out_name
    cv2.imwrite(str(out_path), image)
    print(f"[INFO]  Annotated image saved: {out_path}")
    return out_path


# ===========================================================================
# Step 7 — JSON results
# ===========================================================================

def _make_json_safe(obj):
    """Recursively convert numpy scalars / non-serialisable types to Python natives."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    # Handle numpy scalar types without importing numpy
    type_name = type(obj).__name__
    if type_name in {"float32", "float64", "float16"}:
        return float(obj)
    if type_name in {"int32", "int64", "int16", "int8", "uint8", "uint16", "uint32", "uint64"}:
        return int(obj)
    if type_name in {"bool_"}:
        return bool(obj)
    return obj


def save_json_results(
    image_path: Path,
    image_width: int,
    image_height: int,
    results: list[dict],
    save_dir: Path,
    conf_threshold: float,
    annotated_path: Path | None,
) -> Path:
    """
    Serialise the full pipeline output to JSON.

    Includes image metadata, every detection, visual metrics, severity data,
    and an overall summary.
    """
    # Build severity distribution summary
    sev_dist: dict[str, int] = {"High": 0, "Medium": 0, "Low": 0, "Not Assessed": 0}
    class_counts: dict[str, int] = defaultdict(int)
    for r in results:
        sev = r.get("severity", "Not Assessed")
        sev_dist[sev] = sev_dist.get(sev, 0) + 1
        class_counts[r.get("class_name", "Unknown")] += 1

    output = {
        "run_timestamp":    datetime.now().isoformat(),
        "model_path":       str(MODEL_PATH),
        "conf_threshold":   conf_threshold,
        "image": {
            "name":   image_path.name,
            "path":   str(image_path),
            "width":  image_width,
            "height": image_height,
        },
        "annotated_image":  str(annotated_path) if annotated_path else None,
        "total_detections": len(results),
        "detections":       results,
        "summary": {
            "detections_by_class":    dict(class_counts),
            "severity_distribution":  sev_dist,
            "not_assessed_count":     sev_dist.get("Not Assessed", 0),
        },
        "engine_limitations": [
            "Severity is estimated from 2-D bounding box geometry only.",
            "Bounding box area is used as a proxy for visible extent, not exact defect area.",
            "Pothole depth, crack width, and structural condition cannot be inferred from a single image.",
            "YOLO confidence score is intentionally excluded from severity logic.",
            "All thresholds are provisional PoC values — not official PWD/ASTM/IRC standards.",
            "Manhole cover severity is always 'Not Assessed': the YOLO class only confirms presence.",
        ],
    }

    # Ensure no numpy or non-serialisable objects remain
    output = _make_json_safe(output)

    out_path = save_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[INFO]  JSON results saved  : {out_path}")
    return out_path


# ===========================================================================
# Step 8 — Evaluation summary
# ===========================================================================

def print_evaluation_summary(
    results: list[dict],
    inference_ok: bool,
    warnings: list[str],
) -> None:
    """Print a concise pipeline health / evaluation section."""
    W = 50
    print("=" * W)
    print("PIPELINE EVALUATION")
    print("=" * W)
    print()
    print(f"  Model inference        : {'✓ OK' if inference_ok else '✗ FAILED'}")
    print(f"  Detections returned    : {len(results)}")

    sev_counts: dict[str, int] = defaultdict(int)
    for r in results:
        sev_counts[r.get("severity", "Unknown")] += 1

    print(f"  Severity assessed      : {sum(v for k,v in sev_counts.items() if k != 'Not Assessed')}")
    print(f"  Not Assessed           : {sev_counts.get('Not Assessed', 0)}")

    if warnings:
        print()
        print("  Warnings:")
        for w in warnings:
            wrapped = textwrap.fill(w, width=W - 4, initial_indent="    • ", subsequent_indent="      ")
            print(wrapped)

    print()
    print("  Known Severity Engine Limitations:")
    limitations = [
        "Severity is estimated from 2-D bounding box geometry only.",
        "Bounding box area ≠ exact defect area.",
        "Pothole depth and crack width cannot be measured from a single image.",
        "YOLO confidence is NOT a severity indicator.",
        "Thresholds are provisional PoC values requiring field calibration.",
        "Manhole cover detection always yields 'Not Assessed'.",
    ]
    for lim in limitations:
        wrapped = textwrap.fill(lim, width=W - 4, initial_indent="    • ", subsequent_indent="      ")
        print(wrapped)

    print()
    print("=" * W)
    print()


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RoadSight AI — End-to-End Pipeline Tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python test_pipeline.py --image images/image.jpeg
              python test_pipeline.py --image images/image2.jpg --conf 0.30
              python test_pipeline.py --image images/image3.jpeg --save-dir tests
        """),
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input road image (JPG / JPEG / PNG).",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--save-dir", "-o",
        default=str(DEFAULT_SAVE_DIR),
        help=f"Directory to save annotated image and JSON (default: {DEFAULT_SAVE_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Validate image
    # ------------------------------------------------------------------
    image_path = validate_image(args.image)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. YOLOv8 inference
    # ------------------------------------------------------------------
    inference_ok = False
    try:
        yolo_results, img_w, img_h, model_names = run_yolo_inference(
            image_path, conf_threshold=args.conf
        )
        inference_ok = True
        print(f"[INFO]  Image dimensions   : {img_w} x {img_h}")
        print(f"[INFO]  Model classes      : {dict(model_names)}")
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit(f"[ERROR] YOLO inference failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Extract detections
    # ------------------------------------------------------------------
    raw_detections = extract_detections(yolo_results, model_names)
    print(f"[INFO]  Raw detections     : {len(raw_detections)}")

    if not raw_detections:
        warnings.append(
            "No objects detected above the confidence threshold "
            f"({args.conf}). Try lowering --conf if detections are expected."
        )

    # ------------------------------------------------------------------
    # 4. Severity assessment
    # ------------------------------------------------------------------
    try:
        enriched_results = run_severity_assessment(raw_detections, img_w, img_h)
    except Exception as exc:
        warnings.append(f"Severity assessment raised an exception: {exc}")
        enriched_results = raw_detections   # fall back to raw

    # ------------------------------------------------------------------
    # 5. Terminal output
    # ------------------------------------------------------------------
    print_results(image_path, img_w, img_h, enriched_results)

    # ------------------------------------------------------------------
    # 6. Annotated image
    # ------------------------------------------------------------------
    annotated_path = save_annotated_image(image_path, enriched_results, save_dir)

    # ------------------------------------------------------------------
    # 7. JSON
    # ------------------------------------------------------------------
    save_json_results(
        image_path, img_w, img_h,
        enriched_results, save_dir,
        conf_threshold=args.conf,
        annotated_path=annotated_path,
    )

    # ------------------------------------------------------------------
    # 8. Evaluation summary
    # ------------------------------------------------------------------
    print_evaluation_summary(enriched_results, inference_ok, warnings)


if __name__ == "__main__":
    main()
