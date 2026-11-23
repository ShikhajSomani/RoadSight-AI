
import cv2

from modules.detection import (
    detect_road_defects,
    save_annotated_image
)

# Input image
image_path = "images/image3.jpeg"

# Run detection and severity assessment
results = detect_road_defects(image_path)

print("\n========== ROADSIGHT AI RESULTS ==========")

print("Image:", results["image_name"])

print(
    "Image dimensions:",
    results["image_width"],
    "x",
    results["image_height"]
)

print("Total detections:", results["total_detections"])

for index, detection in enumerate(
    results["detections"], start=1
):

    print(f"\nDetection {index}")

    print("Class:", detection["class_name"])
    print("Confidence:", detection["confidence"])
    print("Bounding box:", detection["bbox"])

    print(
        "Severity:",
        detection["severity"]["severity"]
    )

# Save annotated image
output_path = save_annotated_image(
    results["annotated_image"]
)

print("\nAnnotated image saved at:", output_path)
