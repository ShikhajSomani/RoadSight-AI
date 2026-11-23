"""
tests/test_severity.py
----------------------
Basic tests for the RoadSight AI rule-based severity assessment engine.

Run with:
    python -m pytest tests/test_severity.py -v
  or simply:
    python tests/test_severity.py

No pytest installation required for the plain-runner at the bottom.
"""

import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.severity import assess_severity, assess_all_severities
from modules import severity_config as cfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_W = 1280
IMAGE_H = 720


def _make_det(class_name, x1, y1, x2, y2, confidence=0.85):
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


# ---------------------------------------------------------------------------
# Tests — valid bounding boxes
# ---------------------------------------------------------------------------

def test_pothole_low():
    # Small pothole — area_ratio well below 1.5 %
    # bbox: 60x40 = 2400 px²  / (1280*720 = 921600) * 100 = 0.26 %
    det = _make_det("Potholes", 100, 100, 160, 140)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_LOW, (
        f"Expected Low, got {result['severity']} (area_ratio={result['visual_metrics']['area_ratio']})"
    )
    assert result["severity_method"] == cfg.SEVERITY_METHOD_RULE_BASED


def test_pothole_medium():
    # Medium pothole — area_ratio between 1.5 % and 5.0 %
    # bbox: 200x100 = 20000 / 921600 * 100 = 2.17 %
    det = _make_det("Potholes", 100, 100, 300, 200)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_MEDIUM, (
        f"Expected Medium, got {result['severity']} (area_ratio={result['visual_metrics']['area_ratio']})"
    )


def test_pothole_high():
    # Large pothole — area_ratio > 5.0 %
    # bbox: 400x200 = 80000 / 921600 * 100 = 8.68 %
    det = _make_det("Potholes", 100, 100, 500, 300)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_HIGH, (
        f"Expected High, got {result['severity']} (area_ratio={result['visual_metrics']['area_ratio']})"
    )


def test_alligator_crack_severity_levels():
    # Low: bbox 100x80 = 8000 / 921600 * 100 = 0.87 %  < 2.0
    low  = assess_severity(_make_det("Alligator Cracks", 0, 0, 100, 80), IMAGE_W, IMAGE_H)
    # Medium: 200x120 = 24000 / 921600 * 100 = 2.60 %  2.0–8.0
    med  = assess_severity(_make_det("Alligator Cracks", 0, 0, 200, 120), IMAGE_W, IMAGE_H)
    # High: 500x300 = 150000 / 921600 * 100 = 16.3 %  > 8.0
    high = assess_severity(_make_det("Alligator Cracks", 0, 0, 500, 300), IMAGE_W, IMAGE_H)

    assert low["severity"]  == cfg.SEVERITY_LOW,    f"Alligator Low failed: {low['severity']}"
    assert med["severity"]  == cfg.SEVERITY_MEDIUM,  f"Alligator Medium failed: {med['severity']}"
    assert high["severity"] == cfg.SEVERITY_HIGH,    f"Alligator High failed: {high['severity']}"


def test_longitudinal_crack_low():
    # Short, low AR box — not elongated enough → Low
    # bbox: 100x80, AR = 1.25 < 1.5
    det = _make_det("Longitudinal Cracks", 0, 0, 100, 80)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_LOW, f"Got {result['severity']}"


def test_longitudinal_crack_high():
    # Very elongated box with meaningful area
    # bbox: 700x60, AR = 11.67 > 4.0, area = 42000 / 921600 * 100 = 4.56 % > 0.5
    det = _make_det("Longitudinal Cracks", 0, 0, 700, 60)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_HIGH, (
        f"Expected High, got {result['severity']} "
        f"ar={result['visual_metrics']['aspect_ratio']} "
        f"area_ratio={result['visual_metrics']['area_ratio']}"
    )


def test_transverse_crack_low():
    # Wide relative to height — NOT transverse → Low
    # bbox: 200x100, AR = 2.0 > 0.8
    det = _make_det("Transverse Cracks", 0, 0, 200, 100)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_LOW, f"Got {result['severity']}"


def test_transverse_crack_high():
    # Tall narrow box covering significant area → High
    # bbox: 60x600, AR = 0.1 < 0.4, area = 36000 / 921600 * 100 = 3.9 % > 0.5
    det = _make_det("Transverse Cracks", 0, 0, 60, 600)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_HIGH, (
        f"Expected High, got {result['severity']} "
        f"ar={result['visual_metrics']['aspect_ratio']} "
        f"area_ratio={result['visual_metrics']['area_ratio']}"
    )


def test_damaged_crosswalk_severity_levels():
    low  = assess_severity(_make_det("Damaged crosswalk", 0, 0, 100, 60),  IMAGE_W, IMAGE_H)
    med  = assess_severity(_make_det("Damaged crosswalk", 0, 0, 300, 200), IMAGE_W, IMAGE_H)
    high = assess_severity(_make_det("Damaged crosswalk", 0, 0, 600, 400), IMAGE_W, IMAGE_H)

    assert low["severity"]  == cfg.SEVERITY_LOW,   f"Crosswalk Low failed: {low['severity']}"
    assert med["severity"]  == cfg.SEVERITY_MEDIUM, f"Crosswalk Medium failed: {med['severity']}"
    assert high["severity"] == cfg.SEVERITY_HIGH,   f"Crosswalk High failed: {high['severity']}"


def test_damaged_paint_severity_levels():
    low  = assess_severity(_make_det("Damaged paint", 0, 0, 80, 40),  IMAGE_W, IMAGE_H)
    med  = assess_severity(_make_det("Damaged paint", 0, 0, 200, 80), IMAGE_W, IMAGE_H)
    high = assess_severity(_make_det("Damaged paint", 0, 0, 400, 150), IMAGE_W, IMAGE_H)

    assert low["severity"]  == cfg.SEVERITY_LOW,   f"Paint Low failed: {low['severity']}"
    assert med["severity"]  == cfg.SEVERITY_MEDIUM, f"Paint Medium failed: {med['severity']}"
    assert high["severity"] == cfg.SEVERITY_HIGH,   f"Paint High failed: {high['severity']}"


def test_manhole_cover_always_not_assessed():
    # Any size manhole cover box → Not Assessed
    for bbox in [(0, 0, 50, 50), (0, 0, 300, 300), (0, 0, 800, 600)]:
        det = _make_det("Manhole cover", *bbox)
        result = assess_severity(det, IMAGE_W, IMAGE_H)
        assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED, (
            f"Manhole cover should always be Not Assessed, got {result['severity']}"
        )
        assert result["severity_method"] == cfg.SEVERITY_METHOD_INSUFFICIENT


# ---------------------------------------------------------------------------
# Tests — invalid / edge-case bounding boxes
# ---------------------------------------------------------------------------

def test_zero_width_bbox():
    det = _make_det("Potholes", 100, 100, 100, 200)  # x2 == x1 → width = 0
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED, (
        f"Zero-width bbox should give Not Assessed, got {result['severity']}"
    )


def test_zero_height_bbox():
    det = _make_det("Potholes", 100, 100, 200, 100)  # y2 == y1 → height = 0
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED


def test_negative_dimension_bbox():
    # x2 < x1 → negative width
    det = _make_det("Alligator Cracks", 300, 100, 100, 200)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED


def test_zero_image_dimensions():
    det = _make_det("Potholes", 0, 0, 100, 100)
    result = assess_severity(det, 0, 720)
    assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED
    result2 = assess_severity(det, 1280, 0)
    assert result2["severity"] == cfg.SEVERITY_NOT_ASSESSED


def test_invalid_bbox_coordinates_type():
    det = {
        "class_name": "Potholes",
        "confidence": 0.9,
        "bbox": {"x1": "bad", "y1": "data", "x2": "here", "y2": "!"},
    }
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED


def test_unknown_class():
    det = _make_det("Flying Saucer", 0, 0, 100, 100)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    assert result["severity"] == cfg.SEVERITY_NOT_ASSESSED
    assert result["severity_method"] == cfg.SEVERITY_METHOD_INSUFFICIENT


# ---------------------------------------------------------------------------
# Tests — assess_all_severities
# ---------------------------------------------------------------------------

def test_assess_all_severities_multiple():
    detections = [
        _make_det("Potholes", 100, 100, 500, 300),          # High
        _make_det("Alligator Cracks", 0, 0, 100, 80),        # Low
        _make_det("Manhole cover", 200, 200, 350, 350),       # Not Assessed
    ]
    results = assess_all_severities(detections, IMAGE_W, IMAGE_H)
    assert len(results) == 3
    assert results[0]["severity"] == cfg.SEVERITY_HIGH
    assert results[1]["severity"] == cfg.SEVERITY_LOW
    assert results[2]["severity"] == cfg.SEVERITY_NOT_ASSESSED


def test_assess_all_severities_empty_list():
    results = assess_all_severities([], IMAGE_W, IMAGE_H)
    assert results == []


def test_assess_all_severities_wrong_type():
    try:
        assess_all_severities("not a list", IMAGE_W, IMAGE_H)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_visual_metrics_present_in_valid_result():
    det = _make_det("Potholes", 100, 150, 340, 280)
    result = assess_severity(det, IMAGE_W, IMAGE_H)
    vm = result.get("visual_metrics", {})
    assert vm["bbox_width"]   == 240
    assert vm["bbox_height"]  == 130
    assert vm["bbox_area"]    == 31200
    assert abs(vm["area_ratio"] - 3.3854) < 0.01
    assert abs(vm["aspect_ratio"] - 1.8462) < 0.01


def test_confidence_not_used_as_severity():
    """High confidence must NOT by itself change the severity label."""
    low_conf  = _make_det("Potholes", 100, 100, 110, 110, confidence=0.10)  # tiny box
    high_conf = _make_det("Potholes", 100, 100, 110, 110, confidence=0.99)  # same tiny box

    r_low  = assess_severity(low_conf,  IMAGE_W, IMAGE_H)
    r_high = assess_severity(high_conf, IMAGE_W, IMAGE_H)

    assert r_low["severity"] == r_high["severity"], (
        "Confidence score must not influence severity — same bbox should give same severity."
    )


# ---------------------------------------------------------------------------
# Plain runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_pothole_low,
        test_pothole_medium,
        test_pothole_high,
        test_alligator_crack_severity_levels,
        test_longitudinal_crack_low,
        test_longitudinal_crack_high,
        test_transverse_crack_low,
        test_transverse_crack_high,
        test_damaged_crosswalk_severity_levels,
        test_damaged_paint_severity_levels,
        test_manhole_cover_always_not_assessed,
        test_zero_width_bbox,
        test_zero_height_bbox,
        test_negative_dimension_bbox,
        test_zero_image_dimensions,
        test_invalid_bbox_coordinates_type,
        test_unknown_class,
        test_assess_all_severities_multiple,
        test_assess_all_severities_empty_list,
        test_assess_all_severities_wrong_type,
        test_visual_metrics_present_in_valid_result,
        test_confidence_not_used_as_severity,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}  →  {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
    sys.exit(0 if failed == 0 else 1)
