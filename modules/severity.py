"""
modules/severity.py
-------------------
Rule-based severity assessment engine for RoadSight AI.

This module receives YOLO detection data and applies class-specific,
configurable rules to assign a preliminary visual severity category:

    Low | Medium | High | Not Assessed

IMPORTANT LIMITATIONS
---------------------
- Severity is estimated from 2-D bounding box geometry only.
- Bounding box area ≠ exact defect area.
- Bounding box area ratio is used as a *proxy* for visible extent —
  not as a precise physical measurement.
- This engine does NOT measure pothole depth, crack width, or structural
  load-bearing capacity.
- YOLO confidence score is intentionally excluded from severity logic.
  High confidence means the model is confident an object exists, not that
  the defect is severe.
- Thresholds are provisional PoC values.  See severity_config.py for details.

Usage
-----
    from modules.severity import assess_severity, assess_all_severities

    result = assess_severity(detection_dict, image_width=1280, image_height=720)
    results = assess_all_severities(list_of_detections, image_width, image_height)
"""

from __future__ import annotations

from typing import Any

from modules import severity_config as cfg

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_visual_metrics(
    x1: float, y1: float, x2: float, y2: float,
    image_width: int, image_height: int,
) -> dict[str, float] | None:
    """
    Compute bounding-box visual metrics.

    Returns a dict with bbox_width, bbox_height, bbox_area, area_ratio, and
    aspect_ratio, or None when the bounding box or image dimensions are invalid.
    """
    # Validate image dimensions
    if image_width <= 0 or image_height <= 0:
        return None

    bbox_width  = x2 - x1
    bbox_height = y2 - y1

    # Guard against zero or negative boxes (can occur with malformed detections)
    if bbox_width <= 0 or bbox_height <= 0:
        return None

    bbox_area   = bbox_width * bbox_height
    image_area  = image_width * image_height
    area_ratio  = (bbox_area / image_area) * 100.0  # expressed as a percentage
    aspect_ratio = bbox_width / bbox_height           # > 1 → wide box, < 1 → tall box

    return {
        "bbox_width":   round(bbox_width,   2),
        "bbox_height":  round(bbox_height,  2),
        "bbox_area":    round(bbox_area,    2),
        "area_ratio":   round(area_ratio,   4),
        "aspect_ratio": round(aspect_ratio, 4),
    }


def _not_assessed_result(
    detection: dict[str, Any],
    visual_metrics: dict[str, float] | None,
    note: str = cfg.SEVERITY_NOTE_NOT_ASSESSED,
) -> dict[str, Any]:
    """Return a standardised 'Not Assessed' severity result."""
    result: dict[str, Any] = {
        "class_name":        detection.get("class_name", "Unknown"),
        "confidence":        detection.get("confidence"),
        "bbox":              detection.get("bbox"),
        "severity":          cfg.SEVERITY_NOT_ASSESSED,
        "severity_method":   cfg.SEVERITY_METHOD_INSUFFICIENT,
        "severity_note":     note,
    }
    if visual_metrics:
        result["visual_metrics"] = visual_metrics
    return result


# ---------------------------------------------------------------------------
# Class-specific severity rules
# ---------------------------------------------------------------------------

def _assess_potholes(metrics: dict[str, float]) -> str:
    """
    Potholes — severity driven by visible bounding-box area ratio.

    area_ratio < low_max                 → Low
    low_max ≤ area_ratio < medium_max    → Medium
    area_ratio ≥ medium_max              → High
    """
    t  = cfg.POTHOLE_THRESHOLDS
    ar = metrics["area_ratio"]

    if ar < t["area_ratio_low_max"]:
        return cfg.SEVERITY_LOW
    if ar < t["area_ratio_medium_max"]:
        return cfg.SEVERITY_MEDIUM
    return cfg.SEVERITY_HIGH


def _assess_alligator_cracks(metrics: dict[str, float]) -> str:
    """
    Alligator (fatigue) cracks — severity driven by bounding-box area ratio.
    A larger affected zone relative to the image indicates wider crack spread.
    """
    t  = cfg.ALLIGATOR_CRACK_THRESHOLDS
    ar = metrics["area_ratio"]

    if ar < t["area_ratio_low_max"]:
        return cfg.SEVERITY_LOW
    if ar < t["area_ratio_medium_max"]:
        return cfg.SEVERITY_MEDIUM
    return cfg.SEVERITY_HIGH


def _assess_longitudinal_cracks(metrics: dict[str, float]) -> str:
    """
    Longitudinal cracks — severity assessed from aspect ratio AND area ratio.

    Longitudinal cracks run parallel to the road axis.  In a landscape-
    oriented image the bounding box should be wider than it is tall
    (aspect_ratio = width / height > 1).

    Thresholds:
      - A short or poorly elongated detection (low AR)   → Low
      - Clearly elongated box, moderate extent           → Medium
      - Very elongated box AND significant area coverage → High

    Both AR and area_ratio must jointly satisfy the High condition to avoid
    a large-but-stubby box being promoted to High.
    """
    t  = cfg.LONGITUDINAL_CRACK_THRESHOLDS
    ar  = metrics["aspect_ratio"]   # width / height
    pct = metrics["area_ratio"]

    if ar < t["ar_low_max"] or pct < t["area_ratio_minimum_for_medium"]:
        return cfg.SEVERITY_LOW
    if ar < t["ar_medium_max"] or pct < t["area_ratio_minimum_for_high"]:
        return cfg.SEVERITY_MEDIUM
    return cfg.SEVERITY_HIGH


def _assess_transverse_cracks(metrics: dict[str, float]) -> str:
    """
    Transverse cracks — severity assessed from aspect ratio AND area ratio.

    Transverse cracks run perpendicular to the road axis.  In a landscape-
    oriented image the bounding box should be taller than it is wide
    (aspect_ratio = width / height < 1).

    Note: the aspect-ratio thresholds are INVERTED compared to longitudinal
    cracks.  A lower aspect ratio means the box is more transverse (taller
    relative to its width), indicating a wider crack span.

    Thresholds:
      - ar > ar_low_max     → barely transverse / short     → Low
      - ar_medium_max ≤ ar ≤ ar_low_max → moderately transverse → Medium
      - ar < ar_medium_max AND sufficient area coverage      → High
    """
    t   = cfg.TRANSVERSE_CRACK_THRESHOLDS
    ar  = metrics["aspect_ratio"]   # width / height; < 1 means taller than wide
    pct = metrics["area_ratio"]

    if ar > t["ar_low_max"] or pct < t["area_ratio_minimum_for_medium"]:
        return cfg.SEVERITY_LOW
    if ar > t["ar_medium_max"] or pct < t["area_ratio_minimum_for_high"]:
        return cfg.SEVERITY_MEDIUM
    return cfg.SEVERITY_HIGH


def _assess_damaged_crosswalk(metrics: dict[str, float]) -> str:
    """
    Damaged crosswalk — severity driven by visible area ratio.
    A larger degraded region suggests more urgent repainting / resurfacing need.
    """
    t  = cfg.DAMAGED_CROSSWALK_THRESHOLDS
    ar = metrics["area_ratio"]

    if ar < t["area_ratio_low_max"]:
        return cfg.SEVERITY_LOW
    if ar < t["area_ratio_medium_max"]:
        return cfg.SEVERITY_MEDIUM
    return cfg.SEVERITY_HIGH


def _assess_damaged_paint(metrics: dict[str, float]) -> str:
    """
    Damaged paint — severity driven by visible area ratio.
    A larger faded / missing marking region carries a higher delineation risk.
    """
    t  = cfg.DAMAGED_PAINT_THRESHOLDS
    ar = metrics["area_ratio"]

    if ar < t["area_ratio_low_max"]:
        return cfg.SEVERITY_LOW
    if ar < t["area_ratio_medium_max"]:
        return cfg.SEVERITY_MEDIUM
    return cfg.SEVERITY_HIGH


# Map class names to their assessment functions
_CLASS_ASSESSORS = {
    "Potholes":           _assess_potholes,
    "Alligator Cracks":   _assess_alligator_cracks,
    "Longitudinal Cracks": _assess_longitudinal_cracks,
    "Transverse Cracks":  _assess_transverse_cracks,
    "Damaged crosswalk":  _assess_damaged_crosswalk,
    "Damaged paint":      _assess_damaged_paint,
    # "Manhole cover" is intentionally absent — handled separately below.
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assess_severity(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """
    Assess the visual severity of a single YOLO detection.

    Parameters
    ----------
    detection : dict
        A detection dictionary with the following expected keys:
            class_name  : str   — defect class label
            confidence  : float — YOLO detection confidence (0–1)
            bbox        : dict  — keys x1, y1, x2, y2 (pixel coordinates)

        Example::

            {
                "class_name": "Potholes",
                "confidence": 0.91,
                "bbox": {"x1": 100, "y1": 150, "x2": 340, "y2": 280}
            }

    image_width : int
        Width of the source image in pixels.

    image_height : int
        Height of the source image in pixels.

    Returns
    -------
    dict
        Enriched detection dictionary including visual_metrics and severity.
        On invalid input, returns a "Not Assessed" result with an error note.
    """
    class_name = detection.get("class_name", "Unknown")
    bbox       = detection.get("bbox", {})

    # --- Extract bounding box coordinates -----------------------------------
    try:
        x1 = float(bbox.get("x1", 0))
        y1 = float(bbox.get("y1", 0))
        x2 = float(bbox.get("x2", 0))
        y2 = float(bbox.get("y2", 0))
    except (TypeError, ValueError):
        return _not_assessed_result(
            detection, None,
            note="Invalid bounding box coordinates. Severity cannot be computed.",
        )

    # --- Compute visual metrics ---------------------------------------------
    metrics = _compute_visual_metrics(x1, y1, x2, y2, image_width, image_height)

    if metrics is None:
        return _not_assessed_result(
            detection, None,
            note=(
                "Bounding box or image dimensions are zero or negative. "
                "Severity cannot be computed."
            ),
        )

    # --- Manhole cover: always Not Assessed ---------------------------------
    if class_name == "Manhole cover":
        return _not_assessed_result(
            detection, metrics,
            note=cfg.SEVERITY_NOTE_NOT_ASSESSED,
        )

    # --- Class-specific severity assessment ---------------------------------
    assessor = _CLASS_ASSESSORS.get(class_name)

    if assessor is None:
        # Unknown class — cannot apply any rule
        return _not_assessed_result(
            detection, metrics,
            note=f"No severity rule defined for class '{class_name}'.",
        )

    severity_label = assessor(metrics)

    return {
        "class_name":      class_name,
        "confidence":      detection.get("confidence"),
        "bbox":            bbox,
        "visual_metrics":  metrics,
        "severity":        severity_label,
        "severity_method": cfg.SEVERITY_METHOD_RULE_BASED,
        "severity_note":   cfg.SEVERITY_NOTE_STANDARD,
    }


def assess_all_severities(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """
    Assess visual severity for a list of YOLO detections.

    Parameters
    ----------
    detections : list[dict]
        List of detection dictionaries (same format as assess_severity).

    image_width : int
        Width of the source image in pixels.

    image_height : int
        Height of the source image in pixels.

    Returns
    -------
    list[dict]
        List of enriched result dictionaries, one per detection.
        Empty list → empty list returned (no error raised).
    """
    if not isinstance(detections, list):
        raise TypeError(
            f"detections must be a list, got {type(detections).__name__}."
        )

    return [
        assess_severity(det, image_width, image_height)
        for det in detections
    ]
