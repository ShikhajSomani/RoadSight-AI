import cv2
from pathlib import Path
from ultralytics import YOLO
from modules.severity import assess_severity

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR/"models"/"best.pt"
CONFIDENCE_THRESHOLD = 0.25

SEVERITY_COLORS = {
    "Low": (0, 200, 0),             # Green
    "Medium": (0, 200, 255),        # Yellow
    "High": (0, 0, 255),            # Red
    "Not Assessed": (128, 128, 128) # Grey
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("RoadSight AI YOLOv8 model loaded successfully.")

def load_image(image_path):
    """
    Load and validate an input road image.

    Args:
        image_path: Path to the input image.

    Returns:
        Loaded image as an OpenCV BGR image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png"
    }:
        raise ValueError(
            "Unsupported image format. Use JPG, JPEG or PNG."
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "Unable to read the image. Check the image file."
        )

    return image

def annotate_image(image, detections):
    """
    Draw bounding boxes, defect labels,
    confidence scores and severity labels.
    """

    annotated_image = image.copy()

    for detection in detections:

        bbox = detection["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        class_name = detection["class_name"]
        confidence = detection["confidence"]
        severity = detection["severity"]["severity"]

        color = SEVERITY_COLORS.get(
            severity,
            SEVERITY_COLORS["Not Assessed"]
        )

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Create label
        label = (
            f"{class_name} | "
            f"{confidence:.2f} | "
            f"{severity}"
        )

        # Calculate label position
        text_y = max(y1 - 10, 20)

        cv2.putText(
            annotated_image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return annotated_image

def detect_road_defects(
    image_path,
    confidence_threshold=CONFIDENCE_THRESHOLD
):
    """
    Detect road defects and assess their visual severity.

    Args:
        image_path: Path to the input road image.
        confidence_threshold: Minimum YOLO confidence.

    Returns:
        Dictionary containing image dimensions,
        detections, severity assessments and
        annotated image.
    """

    # 1. Load image
    image = load_image(image_path)

    # 2. Get original image dimensions
    image_height, image_width = image.shape[:2]

    # 3. Run YOLO inference
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        save=False,
        verbose=False
    )

    detections = []

    # 4. Extract detections
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0].item())

            class_name = model.names[class_id]

            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height

            # Skip invalid bounding boxes
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),

                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                },

                "bbox_width": round(bbox_width, 2),
                "bbox_height": round(bbox_height, 2),
                "bbox_area": round(bbox_area, 2)
            }

            # 5. Assess visual severity
            severity_result = assess_severity(
                detection,
                image_width=image_width,
                image_height=image_height
            )

            detection["severity"] = severity_result

            detections.append(detection)

    # 6. Generate annotated image
    annotated_image = annotate_image(
        image,
        detections
    )

    # 7. Return complete results
    return {
        "image_name": Path(image_path).name,

        "image_width": image_width,
        "image_height": image_height,

        "total_detections": len(detections),

        "detections": detections,

        "annotated_image": annotated_image
    }

def save_annotated_image(
    annotated_image,
    output_path="outputs/detections/annotated_image.jpg"
):
    """
    Save the annotated image without modifying
    the original image.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        annotated_image
    )

    if not success:
        raise IOError(
            f"Could not save annotated image: {output_path}"
        )

    return str(output_path)

