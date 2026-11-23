from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Display model classes
print("Model Classes:")
print(model.names)

# Run inference on an image
results = model.predict(
    source="images/image.jpeg",
    conf=0.25,
    save=True
)

print("Detection completed!")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"Defect: {model.names[class_id]}")
        print(f"Confidence: {confidence:.4f}")
        print(f"X1: {x1:.2f}")
        print(f"Y1: {y1:.2f}")
        print(f"X2: {x2:.2f}")
        print(f"Y2: {y2:.2f}")
        print("-" * 40)