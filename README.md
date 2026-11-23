# RoadSight AI

RoadSight AI is a Streamlit-based proof-of-concept application for detecting
road defects in images with a trained YOLOv8 model. It provides an annotated
defect view, a rule-based preliminary severity assessment, SQLite-backed
inspection records, analytics, and an interactive map of saved locations.

## Features

- Upload JPG, JPEG, or PNG road images
- Detect road defects with `models/best.pt`
- Show bounding boxes, defect classes, confidence scores, and severity labels
- Assess preliminary severity as Low, Medium, High, or Not Assessed
- Resolve inspection locations through OpenStreetMap's Nominatim geocoder
- Save inspections and detections to a local SQLite database
- View defect and severity distributions in the Overview dashboard
- Map saved inspection locations with Folium

## Supported Defect Classes

The severity engine provides class-specific rules for:

- Potholes
- Alligator Cracks
- Longitudinal Cracks
- Transverse Cracks
- Damaged crosswalk
- Damaged paint

Manhole covers are identified by the detection model but are reported as
**Not Assessed** because a 2-D bounding box does not indicate the cover's
physical condition.

## Technology Stack

- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- Pandas and Plotly
- Folium and streamlit-folium
- GeoPy / Nominatim
- SQLite

## Severity Assessment

Severity is estimated from 2-D bounding-box geometry, including image area
coverage and, for longitudinal and transverse cracks, aspect ratio.
Confidence scores are intentionally not used to decide severity.

These thresholds are provisional proof-of-concept values configured in
`modules/severity_config.py`. They are not engineering, PWD, ASTM, or IRC
standards. The system does not measure crack width, pothole depth, structural
capacity, or exact defect area.

## Project Structure

```text
.
|-- app.py                     # Streamlit application
|-- init_db.py                 # Database initialization script
|-- check_db.py                # Inspect SQLite table names
|-- requirements.txt           # Python dependencies
|-- roadsight.db               # SQLite database
|-- models/
|   `-- best.pt                # Trained YOLOv8 model
|-- modules/
|   |-- detection.py           # Image loading, YOLO inference, annotation
|   |-- severity.py            # Rule-based severity engine
|   |-- severity_config.py     # Severity thresholds and labels
|   `-- database.py            # SQLite schema and CRUD helpers
|-- tests/
|   |-- test_severity.py       # Severity rule tests
|   |-- test_detection.py      # Detection smoke test
|   |-- test_model.py          # Model and inference test
|   `-- test_pipeline.py       # End-to-end pipeline utility
|-- images/                    # Sample images
|-- uploads/                   # Uploaded images
`-- outputs/detections/         # Generated annotated images
```

## Prerequisites

- Python 3.10 or newer
- A virtual environment is recommended
- The trained model file at `models/best.pt`
- Internet access for Nominatim location lookup and map tiles
- Internet or model compatibility support for the PyTorch/Ultralytics runtime

## Installation

Open a terminal in the project root.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install geopy
```

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install geopy
```

`geopy` is required by `app.py` but is not currently listed in
`requirements.txt`.

## Initialize the Database

The Streamlit app initializes the database automatically. It can also be
initialized manually:

```bash
python init_db.py
```

To list the tables in the existing database:

```bash
python check_db.py
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`.

### Using the App

1. Open **Road Inspection**.
2. Enter a road name or location and select **Find Location**.
3. Upload a supported road image.
4. Select **Detect Road Defects**.
5. Review the annotated image and detected defect details.
6. Select **Save Inspection** to store the result in SQLite.
7. Open **Overview** for dashboard statistics or **Road Monitoring** for the map.

## Run Detection from the Terminal

The detection module can be used directly:

```python
from modules.detection import detect_road_defects, save_annotated_image

result = detect_road_defects(
    image_path="images/image.jpeg",
    confidence_threshold=0.25,
)

print(result["total_detections"])
print(result["detections"])

save_annotated_image(
    result["annotated_image"],
    "outputs/detections/annotated_image.jpg",
)
```

## Testing

Run the severity test suite directly without pytest:

```bash
python tests/test_severity.py
```

With pytest installed, the suite can also be run as:

```bash
python -m pytest tests/test_severity.py -v
```

Run all available tests with:

```bash
python -m pytest tests -v
```

Some test scripts require sample images under `images/` and should be reviewed
before relying on them as automated model benchmarks.

## Database Schema

The application uses two SQLite tables:

### `inspections`

- `inspection_id`
- `image_name`
- `image_path`
- `latitude`
- `longitude`
- `inspection_date`
- `total_detections`

### `detections`

- `detection_id`
- `inspection_id`
- `defect_class`
- `confidence`
- `x1`, `y1`, `x2`, `y2`
- `severity`

Detections reference inspections with a foreign key and are deleted when their
parent inspection is deleted.

## Known Limitations

- The severity labels are visual estimates, not maintenance priority
  certifications or engineering assessments.
- Location lookup depends on Nominatim availability and network connectivity.
- The model may perform differently across roads, camera angles, lighting,
  weather, and image resolution.
- Uploaded images are stored under `uploads/`.
- The SQLite database is local and is not designed for concurrent multi-user
  deployment without additional infrastructure.

## Troubleshooting

### `Trained model not found`

Ensure `models/best.pt` exists and that the app is run from the project root.

### `No module named geopy`

Install the missing package:

```bash
python -m pip install geopy
```

### Location lookup fails

Check internet connectivity and try a more specific location, such as a road
and city name.

### Slow first startup

The first YOLO/PyTorch startup may take additional time while dependencies
and model runtimes initialize.

## Contributing

1. Create a feature branch.
2. Make a focused change.
3. Run the relevant tests.
4. Open a pull request with setup and testing details.
