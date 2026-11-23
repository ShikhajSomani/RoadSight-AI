
import sqlite3
from pathlib import Path
from datetime import datetime


# ==================================================
# DATABASE CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "roadsight.db"


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():
    """
    Establish a connection with the SQLite database.
    """

    conn = sqlite3.connect(DB_PATH)

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ==================================================
# INITIALIZE DATABASE
# ==================================================

def initialize_database():
    """
    Create the required database tables
    if they do not already exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # ----------------------------------------------
    # INSPECTIONS TABLE
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (

            inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_name TEXT NOT NULL,

            image_path TEXT NOT NULL,

            latitude REAL,

            longitude REAL,

            inspection_date TEXT NOT NULL,

            total_detections INTEGER DEFAULT 0

        )
    """)

    # ----------------------------------------------
    # DETECTIONS TABLE
    # ----------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (

            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,

            inspection_id INTEGER NOT NULL,

            defect_class TEXT NOT NULL,

            confidence REAL,

            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL,

            severity TEXT,

            FOREIGN KEY (inspection_id)
                REFERENCES inspections(inspection_id)
                ON DELETE CASCADE

        )
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully!")


# ==================================================
# SAVE INSPECTION
# ==================================================

def save_inspection(
    image_name,
    image_path,
    latitude,
    longitude,
    detections
):
    """
    Save an inspection and all its detections.

    Returns:
        inspection_id
    """

    conn = get_connection()
    cursor = conn.cursor()

    try:

        inspection_date = datetime.now().isoformat(
            timespec="seconds"
        )

        # Insert inspection record
        cursor.execute("""
            INSERT INTO inspections (
                image_name,
                image_path,
                latitude,
                longitude,
                inspection_date,
                total_detections
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            image_name,
            image_path,
            latitude,
            longitude,
            inspection_date,
            len(detections)
        ))

        inspection_id = cursor.lastrowid

        # Insert individual detections
        for detection in detections:

            bbox = detection["bbox"]

            severity_result = detection["severity"]

            cursor.execute("""
                INSERT INTO detections (
                    inspection_id,
                    defect_class,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                    severity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id,
                detection["class_name"],
                detection["confidence"],
                bbox["x1"],
                bbox["y1"],
                bbox["x2"],
                bbox["y2"],
                severity_result["severity"]
            ))

        conn.commit()

        return inspection_id

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()


# ==================================================
# FETCH ALL INSPECTIONS
# ==================================================

def get_all_inspections():
    """
    Retrieve all inspection records.
    """

    conn = get_connection()

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM inspections
        ORDER BY inspection_date DESC
    """)

    records = [
        dict(row)
        for row in cursor.fetchall()
    ]

    conn.close()

    return records


# ==================================================
# FETCH ALL DETECTIONS
# ==================================================

def get_all_detections():
    """
    Retrieve all detected defects.
    """

    conn = get_connection()

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM detections
        ORDER BY detection_id DESC
    """)

    records = [
        dict(row)
        for row in cursor.fetchall()
    ]

    conn.close()

    return records


# ==================================================
# FETCH DETECTIONS FOR AN INSPECTION
# ==================================================

def get_inspection_detections(inspection_id):
    """
    Retrieve all detections belonging
    to a particular inspection.
    """

    conn = get_connection()

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM detections
        WHERE inspection_id = ?
    """, (inspection_id,))

    records = [
        dict(row)
        for row in cursor.fetchall()
    ]

    conn.close()

    return records


# ==================================================
# FETCH INSPECTION SUMMARY
# ==================================================

def get_inspection_summary():
    """
    Retrieve basic inspection statistics.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) AS total_inspections,
            COALESCE(SUM(total_detections), 0)
                AS total_detections
        FROM inspections
    """)

    result = cursor.fetchone()

    conn.close()

    return {
        "total_inspections": result[0],
        "total_detections": result[1]
    }
