"""
modules/severity_config.py
--------------------------
Severity threshold configuration for the RoadSight AI rule-based engine.

IMPORTANT — READ BEFORE EDITING THRESHOLDS
-------------------------------------------
All thresholds below are ILLUSTRATIVE / PROVISIONAL values chosen for the
Proof-of-Concept stage.  They are NOT official Public Works Department (PWD)
or ASTM / IRC engineering standards.

Thresholds are expressed as `area_ratio` (percentage of image area occupied
by the bounding box) unless noted otherwise.  They should be revisited and
calibrated once field-validated ground-truth annotations are available.

Confidence score is intentionally excluded from severity logic.
A high YOLO confidence means the model is sure it detected an object —
it says nothing about how severe that defect is physically.
"""

# ---------------------------------------------------------------------------
# Potholes
# Severity is estimated from the visible bounding-box area relative to the
# full image.  The aspect ratio flag can optionally distinguish roughly
# circular potholes (ar ≈ 1) from elongated road depressions.
# ---------------------------------------------------------------------------
POTHOLE_THRESHOLDS = {
    # area_ratio (%) boundaries:  < LOW_MAX → Low, < MEDIUM_MAX → Medium, else High
    "area_ratio_low_max":    1.5,   # < 1.5 % of image → Low
    "area_ratio_medium_max": 5.0,   # 1.5–5.0 % → Medium, > 5.0 % → High
    # Aspect-ratio hint: values close to 1.0 suggest a compact/circular shape.
    # Currently used only for the severity_note, not for threshold overrides.
    "compact_ar_max":        2.5,   # ar ≤ 2.5 → "roughly compact"
}

# ---------------------------------------------------------------------------
# Alligator (Fatigue) Cracks
# Network cracking typically spreads over a large wheel-path zone.  A larger
# bounding box relative to the image suggests more advanced deterioration.
# ---------------------------------------------------------------------------
ALLIGATOR_CRACK_THRESHOLDS = {
    "area_ratio_low_max":    2.0,   # < 2.0 % → Low
    "area_ratio_medium_max": 8.0,   # 2.0–8.0 % → Medium, > 8.0 % → High
}

# ---------------------------------------------------------------------------
# Longitudinal Cracks
# These run roughly parallel to the road centreline.  A tall, narrow bounding
# box (high aspect ratio, i.e. width >> height when image is landscape) is
# expected.  Severity is estimated from both visible length (via bbox_area
# ratio) and elongation.
# ---------------------------------------------------------------------------
LONGITUDINAL_CRACK_THRESHOLDS = {
    # Aspect-ratio (width / height) thresholds — longitudinal cracks are
    # expected to be wider than they are tall in a landscape-oriented image.
    "ar_low_max":    1.5,   # ar < 1.5  → Low    (short or poorly defined)
    "ar_medium_max": 4.0,   # 1.5–4.0  → Medium  (clearly elongated)
                            # ar > 4.0  → High    (very long relative to width)

    # Area-ratio modifier: even a high-AR crack is not High unless it also
    # covers a meaningful portion of the image.
    "area_ratio_minimum_for_high": 0.5,  # must be > 0.5 % to reach High
    "area_ratio_minimum_for_medium": 0.2,
}

# ---------------------------------------------------------------------------
# Transverse Cracks
# These run roughly perpendicular to the road centreline.  The bounding box
# should be taller than it is wide (height > width), so we invert the aspect
# ratio check compared to longitudinal cracks.
# ---------------------------------------------------------------------------
TRANSVERSE_CRACK_THRESHOLDS = {
    # Aspect-ratio (width / height):
    # Transverse cracks should be tall → ar < 1 ideally, but detection boxes
    # may not be perfect.  We flag long spans with ar < thresholds.
    "ar_low_max":    0.8,   # ar > 0.8  → barely transverse (Low)
    "ar_medium_max": 0.4,   # 0.4–0.8  → moderately transverse (Medium)
                            # ar < 0.4  → clearly full-width crack (High)

    "area_ratio_minimum_for_high":   0.5,
    "area_ratio_minimum_for_medium": 0.2,
}

# ---------------------------------------------------------------------------
# Damaged Crosswalk
# Assessed purely by relative visible area of the damaged portion.
# ---------------------------------------------------------------------------
DAMAGED_CROSSWALK_THRESHOLDS = {
    "area_ratio_low_max":    3.0,   # < 3.0 % → Low
    "area_ratio_medium_max": 10.0,  # 3.0–10.0 % → Medium, > 10.0 % → High
}

# ---------------------------------------------------------------------------
# Damaged Paint (lane markings, arrows, symbols, etc.)
# ---------------------------------------------------------------------------
DAMAGED_PAINT_THRESHOLDS = {
    "area_ratio_low_max":    1.0,   # < 1.0 % → Low
    "area_ratio_medium_max": 4.0,   # 1.0–4.0 % → Medium, > 4.0 % → High
}

# ---------------------------------------------------------------------------
# Manhole Cover
# A detected manhole cover is NOT automatically assessed for severity.
# The current YOLO class only identifies the cover's presence; there is no
# visual signal in the bounding box alone to determine whether the cover is
# damaged, missing, or raised/sunken.
# Severity is therefore always returned as "Not Assessed".
# ---------------------------------------------------------------------------
MANHOLE_COVER_THRESHOLDS = {}  # intentionally empty — see above

# ---------------------------------------------------------------------------
# Severity label strings — do not change these without updating severity.py
# ---------------------------------------------------------------------------
SEVERITY_LOW           = "Low"
SEVERITY_MEDIUM        = "Medium"
SEVERITY_HIGH          = "High"
SEVERITY_NOT_ASSESSED  = "Not Assessed"

SEVERITY_METHOD_RULE_BASED   = "rule_based"
SEVERITY_METHOD_INSUFFICIENT = "insufficient_visual_information"

SEVERITY_NOTE_STANDARD = (
    "Preliminary visual extent estimate based on bounding box measurements. "
    "Does not reflect actual physical depth, crack width, or structural condition."
)
SEVERITY_NOTE_NOT_ASSESSED = (
    "Severity cannot be reliably inferred from a 2D bounding box for this class. "
    "Manual inspection required."
)
