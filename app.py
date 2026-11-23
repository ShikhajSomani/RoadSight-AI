
import streamlit as st
import uuid
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from pathlib import Path
from modules.detection import detect_road_defects
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from modules.database import (
    initialize_database,
    get_all_inspections,
    get_all_detections,
    get_inspection_summary,
    save_inspection
)

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RoadSight AI",
    page_icon="🛣️",
    layout="wide"
)

# Initialize database
initialize_database()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛣️ RoadSight AI")
st.sidebar.caption("Road Defect Detection & Monitoring")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Road Inspection",
        "Road Monitoring"
    ]
)


def get_coordinates(location_name):

    geolocator = Nominatim(
        user_agent="roadsight_ai_poc"
    )

    location = geolocator.geocode(
        location_name,
        timeout=10
    )

    if location is None:
        return None

    return {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "address": location.address
    }

# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------


def render_overview():

    st.title("🛣️ RoadSight AI")
    st.subheader("Road Monitoring Overview")

    st.write(
        "Monitor road inspections, detected defects "
        "and preliminary visual severity assessments."
    )

    # =========================================
    # 1. FETCH DATA FROM SQLITE DATABASE
    # =========================================

    try:

        summary = get_inspection_summary()

        inspections = get_all_inspections()

        detections = get_all_detections()

    except Exception as e:

        st.error(f"Unable to fetch dashboard data: {e}")
        return

    # =========================================
    # 2. KEY PERFORMANCE INDICATORS
    # =========================================

    st.subheader("Inspection Statistics")

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Inspections",
        summary["total_inspections"]
    )

    col2.metric(
        "Total Defects Detected",
        summary["total_detections"]
    )

    # =========================================
    # 3. SEVERITY DISTRIBUTION
    # =========================================

    st.divider()

    st.subheader("Defect Severity Overview")

    severity_counts = {
        "High": 0,
        "Medium": 0,
        "Low": 0,
        "Not Assessed": 0
    }

    for detection in detections:

        severity = detection.get("severity", "Not Assessed")

        if severity in severity_counts:
            severity_counts[severity] += 1

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🔴 High", severity_counts["High"])

    col2.metric("🟡 Medium", severity_counts["Medium"])

    col3.metric("🟢 Low", severity_counts["Low"])

    col4.metric(
        "⚪ Not Assessed",
        severity_counts["Not Assessed"]
    )

    # =========================================
    # 4. DEFECT CATEGORY DISTRIBUTION
    # =========================================

    st.divider()

    st.subheader("Defect Category Distribution")

    if detections:

        defect_df = pd.DataFrame(detections)

        category_counts = (
            defect_df["defect_class"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "Defect Category",
            "Count"
        ]

        fig_category = px.bar(
            category_counts,
            x="Defect Category",
            y="Count",
            color="Defect Category",
            text="Count",
            title="Number of Detections by Defect Category"
        )

        fig_category.update_layout(
            showlegend=False,
            xaxis_title="Defect Category",
            yaxis_title="Number of Detections"
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    else:

        st.info("No defect records available yet.")

    # =========================================
    # 5. SEVERITY DISTRIBUTION CHART
    # =========================================

    st.divider()

    st.subheader("Severity Distribution")

    severity_df = pd.DataFrame(
        [
            {"Severity": severity, "Count": count}
            for severity, count in severity_counts.items()
            if count > 0
        ]
    )

    if not severity_df.empty:

        fig_severity = px.pie(
            severity_df,
            names="Severity",
            values="Count",
            hole=0.45,
            title="Distribution of Detected Defects by Severity"
        )

        st.plotly_chart(
            fig_severity,
            use_container_width=True
        )

    else:

        st.info("No severity data available yet.")

    # =========================================
    # 6. RECENT INSPECTIONS
    # =========================================

    st.divider()

    st.subheader("Recent Inspections")

    if inspections:

        recent_df = pd.DataFrame(inspections)

        recent_df = recent_df.head(10)

        display_columns = [
            "inspection_id",
            "image_name",
            "latitude",
            "longitude",
            "inspection_date",
            "total_detections"
        ]

        recent_df = recent_df[display_columns]

        recent_df.columns = [
            "Inspection ID",
            "Image",
            "Latitude",
            "Longitude",
            "Inspection Date",
            "Total Defects"
        ]

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No inspections have been recorded yet. "
            "Go to Road Inspection to create your first inspection."
        )


# --------------------------------------------------
# PAGE 2: ROAD INSPECTION
# --------------------------------------------------


def render_road_inspection():

    st.title("Road Inspection")

    st.write(
        "Upload a road image to detect defects "
        "and assess their preliminary visual severity."
    )

    # =========================================
    # 1. INSPECTION LOCATION
    # =========================================

    st.subheader("📍 Inspection Location")

    location_name = st.text_input(
        "Enter road name or location",
        placeholder="Example: MG Road, Bengaluru",
        key="inspection_location_input"
    )

    if st.button("Find Location", key="find_location"):

        if not location_name.strip():

            st.warning("Please enter a location name.")

        else:

            try:

                with st.spinner("Finding location..."):

                    result = get_coordinates(
                        location_name.strip()
                    )

                if result is None:

                    st.session_state.pop(
                        "resolved_location", None
                    )

                    st.session_state.pop(
                        "resolved_location_query", None
                    )

                    st.error(
                        "Location not found. "
                        "Try entering a more specific address."
                    )

                else:

                    st.session_state["resolved_location"] = result

                    st.session_state["resolved_location_query"] = (
                        location_name.strip()
                    )

            except (GeocoderTimedOut, GeocoderUnavailable):

                st.error(
                    "Location service is temporarily unavailable. "
                    "Please try again."
                )

            except Exception as e:

                st.error(f"Location search failed: {e}")

    # Display resolved location

    resolved = st.session_state.get("resolved_location")

    location_confirmed = (
        resolved is not None
        and st.session_state.get("resolved_location_query")
        == location_name.strip()
        and bool(location_name.strip())
    )

    if location_confirmed:

        st.success("Location identified successfully!")

        st.write(
            "**Matched Address:**",
            resolved["address"]
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Latitude",
            f'{resolved["latitude"]:.6f}'
        )

        col2.metric(
            "Longitude",
            f'{resolved["longitude"]:.6f}'
        )

    elif resolved is not None:

        st.info(
            "The location name has changed. "
            "Click Find Location to resolve it again."
        )

    st.divider()

    # =========================================
    # 2. ROAD IMAGE UPLOAD
    # =========================================

    st.subheader("📷 Upload Road Image")

    uploaded_file = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png"],
        key="road_image_uploader"
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Road Image",
            use_container_width=True
        )

        # =====================================
        # 3. RUN DEFECT DETECTION
        # =====================================

        if st.button(
            "🔍 Detect Road Defects",
            type="primary",
            key="detect_road_defects"
        ):

            if not location_confirmed:

                st.warning(
                    "Please enter and confirm the inspection location first."
                )

            else:

                try:

                    # Create uploads directory
                    upload_dir = Path("uploads")

                    upload_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    # Save uploaded image
                    extension = Path(
                        uploaded_file.name
                    ).suffix.lower()

                    image_path = upload_dir / (
                        f"{uuid.uuid4().hex}{extension}"
                    )

                    with open(image_path, "wb") as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )

                    # Run YOLOv8 inference
                    with st.spinner(
                        "Detecting road defects. Please wait..."
                    ):

                        results = detect_road_defects(
                            image_path=str(image_path)
                        )

                    # Store results
                    st.session_state["detection_results"] = results

                    st.session_state["detection_image_path"] = (
                        str(image_path)
                    )

                    st.session_state["detection_location"] = resolved

                    st.session_state["detection_location_query"] = (
                        location_name.strip()
                    )

                    # Reset previous save status
                    st.session_state.pop(
                        "saved_inspection_id", None
                    )

                    st.success(
                        "Road defect detection completed successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Detection failed: {e}"
                    )

    # =========================================
    # 4. DISPLAY DETECTION RESULTS
    # =========================================

    results = st.session_state.get("detection_results")

    if results is not None:

        # Ensure results belong to current location
        if (
            st.session_state.get("detection_location_query")
            != location_name.strip()
            or not location_confirmed
        ):

            st.info(
                "The inspection location has changed. "
                "Confirm the location and run detection again."
            )

        else:

            st.divider()

            st.subheader("Detection Results")

            # Total detections
            st.metric(
                "Total Defects Detected",
                results["total_detections"]
            )

            # Annotated image
            st.image(
                results["annotated_image"],
                caption="Road Defect Detection Results",
                channels="BGR",
                use_container_width=True
            )

            # =================================
            # 5. INDIVIDUAL DEFECT DETAILS
            # =================================

            if results["total_detections"] == 0:

                st.info(
                    "No road defects were detected in this image."
                )

            else:

                st.subheader("Detected Defects")

                for index, detection in enumerate(
                    results["detections"],
                    start=1
                ):

                    severity_result = detection["severity"]

                    severity = severity_result["severity"]

                    with st.expander(
                        f'{index}. {detection["class_name"]} — {severity}',
                        expanded=True
                    ):

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Confidence",
                            f'{detection["confidence"] * 100:.2f}%'
                        )

                        col2.metric(
                            "Visual Severity",
                            severity
                        )


            # =================================
            # 6. SAVE INSPECTION TO DATABASE
            # =================================

            st.divider()

            st.subheader("Save Inspection")

            saved_inspection_id = st.session_state.get(
                "saved_inspection_id"
            )

            if saved_inspection_id is not None:

                st.success(
                    f"Inspection saved successfully! "
                    f"Inspection ID: {saved_inspection_id}"
                )

            else:

                if st.button(
                    "💾 Save Inspection",
                    type="primary",
                    key="save_inspection"
                ):

                    try:

                        location_data = st.session_state[
                            "detection_location"
                        ]

                        inspection_id = save_inspection(

                            image_name=results["image_name"],

                            image_path=st.session_state[
                                "detection_image_path"
                            ],

                            latitude=location_data["latitude"],

                            longitude=location_data["longitude"],

                            detections=results["detections"]
                        )

                        st.session_state["saved_inspection_id"] = (
                            inspection_id
                        )

                        st.success(
                            f"Inspection saved successfully! "
                            f"Inspection ID: {inspection_id}"
                        )

                    except Exception as e:

                        st.error(
                            f"Failed to save inspection: {e}"
                        )


# --------------------------------------------------
# PAGE 3: ROAD MONITORING
# --------------------------------------------------


def render_road_monitoring():

    st.title("🗺️ Road Monitoring")

    st.write(
        "Monitor inspected road locations, view detected defects "
        "and identify locations that may require maintenance."
    )

    # =========================================
    # 1. FETCH INSPECTION DATA
    # =========================================

    try:

        inspections = get_all_inspections()
        detections = get_all_detections()

    except Exception as e:

        st.error(f"Unable to fetch monitoring data: {e}")
        return

    if not inspections:

        st.info(
            "No inspection locations available yet. "
            "Go to Road Inspection and save an inspection first."
        )
        return

    # =========================================
    # 2. GROUP DETECTIONS BY INSPECTION
    # =========================================

    detection_groups = {}

    for detection in detections:

        inspection_id = detection["inspection_id"]

        if inspection_id not in detection_groups:
            detection_groups[inspection_id] = []

        detection_groups[inspection_id].append(detection)

    # =========================================
    # 3. PREPARE INSPECTION LOCATIONS
    # =========================================

    valid_inspections = []

    for inspection in inspections:

        latitude = inspection.get("latitude")
        longitude = inspection.get("longitude")

        if latitude is None or longitude is None:
            continue

        inspection_id = inspection["inspection_id"]

        inspection_detections = detection_groups.get(
            inspection_id, []
        )

        severity_counts = {
            "High": 0,
            "Medium": 0,
            "Low": 0,
            "Not Assessed": 0
        }

        defect_categories = {}

        for detection in inspection_detections:

            severity = detection.get(
                "severity", "Not Assessed"
            )

            if severity not in severity_counts:
                severity = "Not Assessed"

            severity_counts[severity] += 1

            defect_class = detection.get(
                "defect_class", "Unknown"
            )

            defect_categories[defect_class] = (
                defect_categories.get(defect_class, 0) + 1
            )

        # Determine highest detected severity
        if severity_counts["High"] > 0:
            highest_severity = "High"

        elif severity_counts["Medium"] > 0:
            highest_severity = "Medium"

        elif severity_counts["Low"] > 0:
            highest_severity = "Low"

        else:
            highest_severity = "Not Assessed"

        valid_inspections.append({
            "inspection": inspection,
            "latitude": latitude,
            "longitude": longitude,
            "detections": inspection_detections,
            "severity_counts": severity_counts,
            "defect_categories": defect_categories,
            "highest_severity": highest_severity
        })

    if not valid_inspections:

        st.warning(
            "No inspections have valid latitude and longitude values."
        )
        return

    # =========================================
    # 4. MONITORING SUMMARY
    # =========================================

    st.subheader("Monitoring Summary")

    total_inspections = len(inspections)
    total_locations = len(valid_inspections)

    total_defects = sum(
        len(item["detections"])
        for item in valid_inspections
    )

    high_priority_locations = sum(
        1 for item in valid_inspections
        if item["highest_severity"] == "High"
    )

    medium_priority_locations = sum(
        1 for item in valid_inspections
        if item["highest_severity"] == "Medium"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Inspections",
        total_inspections
    )

    col2.metric(
        "Mapped Locations",
        total_locations
    )

    col3.metric(
        "Total Defects",
        total_defects
    )

    col4.metric(
        "High Severity Locations",
        high_priority_locations
    )

    # =========================================
    # 5. INTERACTIVE INSPECTION MAP
    # =========================================

    st.divider()

    st.subheader("Inspection Location Map")

    st.caption(
        "Each marker represents one saved inspection. "
        "Marker colour indicates the highest detected severity."
    )

    # Calculate map center
    avg_latitude = sum(
        item["latitude"] for item in valid_inspections
    ) / total_locations

    avg_longitude = sum(
        item["longitude"] for item in valid_inspections
    ) / total_locations

    road_map = folium.Map(
        location=[avg_latitude, avg_longitude],
        zoom_start=12,
        control_scale=True
    )

    # Marker colours
    severity_colors = {
        "High": "red",
        "Medium": "orange",
        "Low": "green",
        "Not Assessed": "gray"
    }

    # Add one marker per inspection
    for item in valid_inspections:

        inspection = item["inspection"]

        inspection_id = inspection["inspection_id"]

        severity = item["highest_severity"]

        color = severity_colors[severity]

        popup_html = f"""
        <div style="width:250px">

            <h4>Inspection #{inspection_id}</h4>

            <b>Image:</b> {inspection["image_name"]}<br>

            <b>Date:</b> {inspection["inspection_date"]}<br>

            <b>Total Defects:</b> {len(item["detections"])}<br>

            <b>Highest Severity:</b> {severity}<br>

            <hr>

            <b>Severity Breakdown</b><br>

            High: {item["severity_counts"]["High"]}<br>
            Medium: {item["severity_counts"]["Medium"]}<br>
            Low: {item["severity_counts"]["Low"]}<br>
            Not Assessed: {item["severity_counts"]["Not Assessed"]}

            <hr>

            <b>Defect Categories</b><br>

            {", ".join(
                f"{name}: {count}"
                for name, count in item["defect_categories"].items()
            ) if item["defect_categories"] else "No defects detected"}

        </div>
        """

        folium.CircleMarker(
            location=[
                item["latitude"],
                item["longitude"]
            ],
            radius=9,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(
                popup_html,
                max_width=300
            ),
            tooltip=f"Inspection #{inspection_id} - {severity}"
        ).add_to(road_map)

    # Display map
    st_folium(
        road_map,
        use_container_width=True,
        height=550
    )

    # Map legend
    st.markdown(
        """
        **Map Legend**

        🔴 High Severity &nbsp;&nbsp;
        🟠 Medium Severity &nbsp;&nbsp;
        🟢 Low Severity &nbsp;&nbsp;
        ⚪ Not Assessed
        """
    )

    # =========================================
    # 6. INSPECTION DETAILS
    # =========================================

    st.divider()

    st.subheader("Inspection Details")

    inspection_options = {
        f'Inspection #{item["inspection"]["inspection_id"]} - '
        f'{item["inspection"]["image_name"]}': item
        for item in valid_inspections
    }

    selected_inspection_label = st.selectbox(
        "Select an inspection to view its details",
        options=list(inspection_options.keys())
    )

    selected = inspection_options[
        selected_inspection_label
    ]

    selected_inspection = selected["inspection"]

    st.write(
        f'**Inspection Date:** '
        f'{selected_inspection["inspection_date"]}'
    )

    st.write(
        f'**Latitude:** {selected["latitude"]:.6f}'
    )

    st.write(
        f'**Longitude:** {selected["longitude"]:.6f}'
    )

    st.write(
        f'**Image:** {selected_inspection["image_name"]}'
    )

    st.write(
        f'**Total Defects:** {len(selected["detections"])}'
    )

    st.write(
        f'**Highest Severity:** {selected["highest_severity"]}'
    )

    # Severity breakdown
    st.markdown("**Severity Breakdown**")

    severity_df = pd.DataFrame(
        [
            {"Severity": severity, "Count": count}
            for severity, count in selected["severity_counts"].items()
        ]
    )

    st.dataframe(
        severity_df,
        use_container_width=True,
        hide_index=True
    )

    # Defect category breakdown
    st.markdown("**Detected Defect Categories**")

    if selected["defect_categories"]:

        category_df = pd.DataFrame(
            [
                {"Defect Category": name, "Count": count}
                for name, count in selected["defect_categories"].items()
            ]
        )

        st.dataframe(
            category_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("No defects were detected in this inspection.")

    # =========================================
    # 7. PRELIMINARY MAINTENANCE PRIORITY
    # =========================================

    st.divider()

    st.subheader("Preliminary Maintenance Priority")

    st.caption(
        "This is a simple PoC classification based on the highest "
        "visual severity detected in each inspection. "
        "It is not an engineering-approved road condition rating."
    )

    priority_data = []

    for item in valid_inspections:

        inspection = item["inspection"]

        severity = item["highest_severity"]

        if severity == "High":
            priority = "High"

        elif severity == "Medium":
            priority = "Medium"

        elif severity == "Low":
            priority = "Low"

        else:
            priority = "Not Assessed"

        priority_data.append({
            "Inspection ID": inspection["inspection_id"],
            "Image": inspection["image_name"],
            "Date": inspection["inspection_date"],
            "Defects": len(item["detections"]),
            "Highest Severity": severity,
            "Maintenance Priority": priority
        })

    priority_df = pd.DataFrame(priority_data)

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2,
        "Not Assessed": 3
    }

    priority_df["_order"] = (
        priority_df["Maintenance Priority"]
        .map(priority_order)
    )

    priority_df = (
        priority_df
        .sort_values("_order")
        .drop(columns="_order")
    )

    st.dataframe(
        priority_df,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------

if page == "Overview":

    render_overview()

elif page == "Road Inspection":

    render_road_inspection()

elif page == "Road Monitoring":

    render_road_monitoring()
