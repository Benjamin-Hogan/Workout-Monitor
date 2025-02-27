# Standard library imports
from auth import check_password
import datetime
import io
import base64
import os
import sys

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pillow_heif
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

# Local imports
from db_utils import create_connection
from utils import (
    COLORS, format_metric, calculate_trend, get_date_range_options,
    apply_date_filter, bmi_category, estimate_body_fat, calculate_bmi,
    show_metric_card, show_date_filter, show_loading_spinner
)

# Add parent directory to path to import auth module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# First check authentication
if not check_password():
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Body Metrics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .element-container {
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def setup_database():
    """Initialize database tables and columns."""
    with create_connection() as conn:
        c = conn.cursor()

        # Create goals table
        c.execute("""
            CREATE TABLE IF NOT EXISTS body_metrics_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                target_value REAL NOT NULL,
                target_date DATE NOT NULL,
                created_date DATE DEFAULT CURRENT_DATE,
                achieved BOOLEAN DEFAULT 0
            )
        """)

        # Create progress photos table
        c.execute("""
            CREATE TABLE IF NOT EXISTS progress_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_date DATE NOT NULL,
                photo_data BLOB NOT NULL,
                photo_type TEXT NOT NULL,
                notes TEXT
            )
        """)

        # Add new columns to body_metrics if they don't exist
        c.execute("PRAGMA table_info(body_metrics)")
        columns = [col[1] for col in c.fetchall()]
        new_columns = {
            'glutes': 'REAL',
            'thigh': 'REAL',
            'calf': 'REAL',
            'neck': 'REAL'
        }
        for col, type_ in new_columns.items():
            if col not in columns:
                c.execute(f"ALTER TABLE body_metrics ADD COLUMN {col} {type_}")

        conn.commit()


def save_progress_photo(photo_file, photo_type, notes=""):
    """Save a progress photo to the database."""
    try:
        # Read the uploaded file
        photo_bytes = photo_file.getvalue()

        # Convert HEIC to JPEG if necessary
        if photo_file.type == "image/heic":
            heif_file = pillow_heif.read_heif(photo_bytes)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Save as JPEG in memory
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            photo_bytes = img_byte_arr.getvalue()

        with create_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO progress_photos (entry_date, photo_data, photo_type, notes)
                VALUES (CURRENT_DATE, ?, ?, ?)
            """, (photo_bytes, photo_type, notes))
            conn.commit()

            # Increment the database modification counter
            if "db_modification_counter" in st.session_state:
                st.session_state.db_modification_counter += 1
            else:
                st.session_state.db_modification_counter = 1

            return True
    except Exception as e:
        st.error(f"Error saving photo: {e}")
        return False


def load_progress_photos():
    """Load all progress photos from the database."""
    with create_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM progress_photos ORDER BY entry_date DESC")
        return c.fetchall()


def save_goal(metric_name, target_value, target_date):
    """Save a new goal to the database."""
    with create_connection() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO body_metrics_goals (metric_name, target_value, target_date)
            VALUES (?, ?, ?)
        """, (metric_name, target_value, target_date))
        conn.commit()

        # Increment the database modification counter
        if "db_modification_counter" in st.session_state:
            st.session_state.db_modification_counter += 1
        else:
            st.session_state.db_modification_counter = 1


def load_goals():
    """Load all active (unachieved) goals from the database."""
    with create_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM body_metrics_goals WHERE achieved = 0",
            conn
        )


def fix_image_orientation(img):
    """Fix image orientation based on EXIF data."""
    try:
        # Check if image has EXIF data
        if hasattr(img, '_getexif') and img._getexif() is not None:
            exif = dict(img._getexif().items())
            # EXIF orientation tag
            if 274 in exif:  # 274 is the orientation tag
                orientation = exif[274]
                # Rotate according to EXIF orientation
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
        return img
    except:
        return img


def show_photos_tab():
    """Display the progress photos management interface."""
    st.header("📸 Progress Photos")

    # Upload Section
    with st.expander("📤 Upload New Photo", expanded=False):
        photo_type = st.selectbox("Type", ["Front", "Side", "Back"])
        photo = st.file_uploader("Upload Photo", type=[
                                 'jpg', 'jpeg', 'png', 'heic'])
        notes = st.text_area("Notes")

        if photo and st.button("Save Photo"):
            if save_progress_photo(photo, photo_type, notes):
                st.success("✅ Photo saved!")
                st.rerun()

    # Gallery Section
    photos = load_progress_photos()
    if photos:
        st.subheader("Photo Gallery")

        # Group photos by type
        photos_by_type = {}
        for photo in photos:
            photo_type = photo[3]  # type is at index 3
            if photo_type not in photos_by_type:
                photos_by_type[photo_type] = []
            photos_by_type[photo_type].append(photo)

        # Create tabs for each photo type
        photo_tabs = st.tabs(["Front", "Side", "Back"])

        for tab_idx, photo_type in enumerate(["Front", "Side", "Back"]):
            with photo_tabs[tab_idx]:
                if photo_type in photos_by_type:
                    type_photos = photos_by_type[photo_type]

                    # Display photos in pairs for comparison
                    for i in range(0, len(type_photos), 2):
                        cols = st.columns(2)

                        # First photo in pair
                        with cols[0]:
                            photo1 = type_photos[i]
                            img1 = Image.open(io.BytesIO(photo1[2]))
                            img1 = fix_image_orientation(img1)

                            # Show date above image
                            st.markdown(f"**{photo1[1]}**")

                            # Display the image
                            st.image(img1, use_container_width=True)

                            # Add edit button below image
                            if st.button("✏️ Edit", key=f"edit_btn_{photo1[0]}"):
                                st.session_state[f"edit_{photo1[0]}"] = True

                            # Show edit interface if in edit mode
                            if st.session_state.get(f"edit_{photo1[0]}", False):
                                with st.expander("Edit Photo", expanded=True):
                                    new_date = st.date_input(
                                        "Date",
                                        value=datetime.datetime.strptime(
                                            photo1[1], '%Y-%m-%d').date() if isinstance(photo1[1], str) else photo1[1],
                                        key=f"date_{photo1[0]}"
                                    )
                                    new_notes = st.text_area(
                                        "Notes", value=photo1[4] if photo1[4] else "", key=f"notes_{photo1[0]}")

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if st.button("💾 Save", key=f"save_{photo1[0]}"):
                                            with create_connection() as conn:
                                                c = conn.cursor()
                                                c.execute("""
                                                    UPDATE progress_photos
                                                    SET entry_date = ?, notes = ?
                                                    WHERE id = ?
                                                """, (new_date, new_notes, photo1[0]))
                                                conn.commit()

                                                # Increment the database modification counter
                                                if "db_modification_counter" in st.session_state:
                                                    st.session_state.db_modification_counter += 1
                                                else:
                                                    st.session_state.db_modification_counter = 1

                                            st.session_state[f"edit_{photo1[0]}"] = False
                                            st.rerun()
                                    with col2:
                                        if st.button("❌ Cancel", key=f"cancel_{photo1[0]}"):
                                            st.session_state[f"edit_{photo1[0]}"] = False
                                            st.rerun()
                                    with col3:
                                        if st.button("🗑️ Delete", key=f"delete_{photo1[0]}"):
                                            with create_connection() as conn:
                                                c = conn.cursor()
                                                c.execute(
                                                    "DELETE FROM progress_photos WHERE id = ?", (photo1[0],))
                                                conn.commit()

                                                # Increment the database modification counter
                                                if "db_modification_counter" in st.session_state:
                                                    st.session_state.db_modification_counter += 1
                                                else:
                                                    st.session_state.db_modification_counter = 1

                                            st.rerun()

                        # Second photo in pair (if exists)
                        if i + 1 < len(type_photos):
                            with cols[1]:
                                photo2 = type_photos[i + 1]
                                img2 = Image.open(io.BytesIO(photo2[2]))
                                img2 = fix_image_orientation(img2)

                                # Show date above image
                                st.markdown(f"**{photo2[1]}**")

                                # Display the image
                                st.image(img2, use_container_width=True)

                                # Add edit button below image
                                if st.button("✏️ Edit", key=f"edit_btn_{photo2[0]}"):
                                    st.session_state[f"edit_{photo2[0]}"] = True

                                # Show edit interface if in edit mode
                                if st.session_state.get(f"edit_{photo2[0]}", False):
                                    with st.expander("Edit Photo", expanded=True):
                                        new_date = st.date_input(
                                            "Date",
                                            value=datetime.datetime.strptime(
                                                photo2[1], '%Y-%m-%d').date() if isinstance(photo2[1], str) else photo2[1],
                                            key=f"date_{photo2[0]}"
                                        )
                                        new_notes = st.text_area(
                                            "Notes", value=photo2[4] if photo2[4] else "", key=f"notes_{photo2[0]}")

                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            if st.button("💾 Save", key=f"save_{photo2[0]}"):
                                                with create_connection() as conn:
                                                    c = conn.cursor()
                                                    c.execute("""
                                                        UPDATE progress_photos
                                                        SET entry_date = ?, notes = ?
                                                        WHERE id = ?
                                                    """, (new_date, new_notes, photo2[0]))
                                                    conn.commit()

                                                    # Increment the database modification counter
                                                    if "db_modification_counter" in st.session_state:
                                                        st.session_state.db_modification_counter += 1
                                                    else:
                                                        st.session_state.db_modification_counter = 1

                                                st.session_state[f"edit_{photo2[0]}"] = False
                                                st.rerun()
                                        with col2:
                                            if st.button("❌ Cancel", key=f"cancel_{photo2[0]}"):
                                                st.session_state[f"edit_{photo2[0]}"] = False
                                                st.rerun()
                                        with col3:
                                            if st.button("🗑️ Delete", key=f"delete_{photo2[0]}"):
                                                with create_connection() as conn:
                                                    c = conn.cursor()
                                                    c.execute(
                                                        "DELETE FROM progress_photos WHERE id = ?", (photo2[0],))
                                                    conn.commit()

                                                    # Increment the database modification counter
                                                    if "db_modification_counter" in st.session_state:
                                                        st.session_state.db_modification_counter += 1
                                                    else:
                                                        st.session_state.db_modification_counter = 1

                                                st.rerun()

                        st.markdown("---")  # Separator between pairs
                else:
                    st.info(f"No {photo_type} photos uploaded yet.")


def show_metrics_entry_tab():
    """Display the metrics entry form and handle submissions."""
    st.header("📈 Body Metrics Entry")

    # Get last metrics entry
    with create_connection() as conn:
        last_metrics = pd.read_sql_query(
            "SELECT * FROM body_metrics ORDER BY entry_date DESC LIMIT 1",
            conn
        )

    # Measurement system selection
    system = st.radio(
        "📐 Measurement System",
        ["Imperial (lbs/in)", "Metric (kg/cm)"]
    )

    with st.form("add_body_metrics_form"):
        st.subheader("📝 Add New Metrics")
        entry_date = st.date_input("📅 Date", datetime.date.today())

        col1, col2, col3 = st.columns(3)

        with col1:
            # Basic measurements
            if system == "Imperial (lbs/in)":
                weight = st.number_input("Weight (lbs)", 0.0, 1000.0, step=0.1,
                                         value=float(last_metrics['user_weight'].iloc[0]) if not last_metrics.empty else 0.0)
                height = st.number_input("Height (in)", 0.0, 120.0, step=0.1,
                                         value=float(last_metrics['height'].iloc[0]) if not last_metrics.empty else 0.0)
            else:
                weight_kg = st.number_input("Weight (kg)", 0.0, 500.0, step=0.1,
                                            value=float(last_metrics['user_weight'].iloc[0] / 2.20462) if not last_metrics.empty else 0.0)
                height_cm = st.number_input("Height (cm)", 0.0, 300.0, step=0.1,
                                            value=float(last_metrics['height'].iloc[0] / 0.393701) if not last_metrics.empty else 0.0)
                weight = weight_kg * 2.20462 if weight_kg > 0 else None
                height = height_cm * 0.393701 if height_cm > 0 else None

            age = st.number_input("Age", 0, 150, step=1,
                                  value=int(last_metrics['age'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['age'].iloc[0]) else 0)
            gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"],
                                  index=["Select", "Male", "Female", "Other"].index(last_metrics['gender'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['gender'].iloc[0]) else 0)

        with col2:
            # Upper body measurements
            st.markdown("**Upper Body**")
            chest = st.number_input("Chest", 0.0, 200.0, step=0.1,
                                    value=float(last_metrics['chest'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['chest'].iloc[0]) else 0.0)
            arms = st.number_input("Arms", 0.0, 100.0, step=0.1,
                                   value=float(last_metrics['arms'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['arms'].iloc[0]) else 0.0)
            neck = st.number_input("Neck", 0.0, 100.0, step=0.1,
                                   value=float(last_metrics['neck'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['neck'].iloc[0]) else 0.0)

        with col3:
            # Lower body measurements
            st.markdown("**Lower Body**")
            waist = st.number_input("Waist", 0.0, 200.0, step=0.1,
                                    value=float(last_metrics['waist'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['waist'].iloc[0]) else 0.0)
            hips = st.number_input("Hips", 0.0, 200.0, step=0.1,
                                   value=float(last_metrics['hips'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['hips'].iloc[0]) else 0.0)
            thigh = st.number_input("Thigh", 0.0, 100.0, step=0.1,
                                    value=float(last_metrics['thigh'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['thigh'].iloc[0]) else 0.0)
            calf = st.number_input("Calf", 0.0, 100.0, step=0.1,
                                   value=float(last_metrics['calf'].iloc[0]) if not last_metrics.empty and not pd.isna(last_metrics['calf'].iloc[0]) else 0.0)

        if st.form_submit_button("Save Metrics"):
            if weight <= 0:
                st.error("Please enter a valid weight.")
                return

            try:
                # Calculate body fat using the Navy method
                body_fat = estimate_body_fat(
                    weight, height, age, gender, neck, waist, hips)

                with create_connection() as conn:
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO body_metrics (
                            entry_date, user_weight, height, age, gender,
                            chest, arms, neck, waist, hips, thigh, calf,
                            body_fat
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(entry_date), weight, height, age,
                        gender if gender != "Select" else None,
                        chest, arms, neck, waist, hips, thigh, calf,
                        body_fat
                    ))
                    conn.commit()
                st.success("✅ Metrics saved successfully!")

                # Increment the database modification counter
                if "db_modification_counter" in st.session_state:
                    st.session_state.db_modification_counter += 1
                else:
                    st.session_state.db_modification_counter = 1
            except Exception as e:
                st.error(f"Error saving metrics: {e}")


def show_goals_tab():
    """Display the goals management interface."""
    st.header("🎯 Goals")

    with st.form("goal_setting"):
        col1, col2, col3 = st.columns(3)
        with col1:
            metric = st.selectbox("Metric", [
                "Weight", "Body Fat", "Chest", "Waist", "Hips",
                "Arms", "Thigh", "Calf", "Neck"
            ])
        with col2:
            target = st.number_input("Target Value", min_value=0.0)
        with col3:
            target_date = st.date_input(
                "Target Date",
                min_value=datetime.date.today()
            )

        if st.form_submit_button("Set Goal"):
            save_goal(metric, target, target_date)
            st.success("✅ Goal saved!")

    goals = load_goals()
    if not goals.empty:
        st.subheader("Current Goals")
        st.dataframe(goals, use_container_width=True)


def show_analysis_tab():
    """Display the metrics analysis and visualization interface."""
    st.header("📊 Analysis")

    with create_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM body_metrics ORDER BY entry_date",
            conn
        )

    if df.empty:
        st.info("No data available for analysis.")
        return

    # Ensure consistent datetime handling
    df['entry_date'] = pd.to_datetime(df['entry_date'])

    # Time range filter
    with st.spinner("Loading data..."):
        start_date = show_date_filter()
        df_filtered = apply_date_filter(
            df, start_date, date_column='entry_date')

        if df_filtered.empty:
            st.info("No data available for the selected time range.")
            return

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            latest_weight = df_filtered['user_weight'].iloc[-1]
            first_weight = df_filtered['user_weight'].iloc[0]
            weight_change = calculate_trend(latest_weight, first_weight)
            show_metric_card("Weight", latest_weight,
                             weight_change, suffix=" lbs")

        with col2:
            latest_bmi = calculate_bmi(
                df_filtered['user_weight'].iloc[-1],
                df_filtered['height'].iloc[-1]
            )
            show_metric_card("BMI", latest_bmi)

        with col3:
            if 'body_fat' in df_filtered.columns:
                latest_bf = df_filtered['body_fat'].iloc[-1]
                first_bf = df_filtered['body_fat'].iloc[0]
                bf_change = calculate_trend(latest_bf, first_bf)
                show_metric_card("Body Fat", latest_bf, bf_change, suffix="%")

        with col4:
            if 'waist' in df_filtered.columns and 'hips' in df_filtered.columns:
                latest_whr = df_filtered['waist'].iloc[-1] / \
                    df_filtered['hips'].iloc[-1]
                show_metric_card("Waist-Hip Ratio", latest_whr, precision=3)

        # Charts
        st.subheader("Weight Trend")
        fig_weight = px.line(
            df_filtered,
            x='entry_date',
            y='user_weight',
            title="Weight Over Time",
            markers=True
        )
        fig_weight.update_xaxes(title="Date")
        fig_weight.update_yaxes(title="Weight (lbs)")
        st.plotly_chart(fig_weight, use_container_width=True)

        if 'body_fat' in df_filtered.columns:
            st.subheader("Body Composition")
            fig_bf = px.line(
                df_filtered,
                x='entry_date',
                y=['body_fat'],
                title="Body Fat % Over Time",
                markers=True
            )
            fig_bf.update_xaxes(title="Date")
            fig_bf.update_yaxes(title="Body Fat %")
            st.plotly_chart(fig_bf, use_container_width=True)

        st.subheader("Measurements Comparison")
        measurements = ['chest', 'waist', 'hips',
                        'arms', 'thigh', 'calf', 'neck']
        available_measurements = [
            m for m in measurements if m in df_filtered.columns]

        if available_measurements:
            fig_measurements = px.line(
                df_filtered,
                x='entry_date',
                y=available_measurements,
                title="Body Measurements Over Time",
                markers=True
            )
            fig_measurements.update_xaxes(title="Date")
            fig_measurements.update_yaxes(title="Measurement (inches)")
            st.plotly_chart(fig_measurements, use_container_width=True)


def show_dashboard_tab():
    """Display the main dashboard with key metrics and trends."""
    st.header("📊 Body Metrics Dashboard")

    with create_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM body_metrics ORDER BY entry_date",
            conn
        )

    if df.empty:
        st.info("No data available for the dashboard.")
        return

    df['entry_date'] = pd.to_datetime(df['entry_date'])

    # Overview Cards
    st.subheader("📈 Current Status")
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        weight_change = calculate_trend(
            latest['user_weight'], prev['user_weight']) if prev is not None else None
        show_metric_card("Current Weight",
                         latest['user_weight'], weight_change, suffix=" lbs")

    with col2:
        bmi = calculate_bmi(latest['user_weight'], latest['height'])
        bmi_status, _ = bmi_category(bmi)
        show_metric_card("BMI", bmi, None, suffix=f" ({bmi_status})")

    with col3:
        bf_change = calculate_trend(
            latest['body_fat'], prev['body_fat']) if prev is not None and 'body_fat' in df.columns else None
        show_metric_card(
            "Body Fat %", latest['body_fat'], bf_change, suffix="%")

    with col4:
        whr = latest['waist'] / latest['hips'] if not pd.isna(
            latest['waist']) and not pd.isna(latest['hips']) else None
        whr_prev = prev['waist'] / prev['hips'] if prev is not None and not pd.isna(
            prev['waist']) and not pd.isna(prev['hips']) else None
        whr_change = calculate_trend(
            whr, whr_prev) if whr is not None and whr_prev is not None else None
        show_metric_card("Waist-Hip Ratio", whr, whr_change, precision=3)

    # Goals Progress
    st.subheader("🎯 Goals Progress")
    goals = load_goals()
    if not goals.empty:
        for _, goal in goals.iterrows():
            metric = goal['metric_name']
            target = goal['target_value']
            target_date = pd.to_datetime(goal['target_date'])

            if metric.lower() == 'weight':
                current = latest['user_weight']
                col = 'user_weight'
            elif metric.lower() == 'body fat':
                current = latest['body_fat']
                col = 'body_fat'
            else:
                current = latest[metric.lower()]
                col = metric.lower()

            progress = (current - df.iloc[0][col]) / \
                (target - df.iloc[0][col]) * 100
            progress = min(max(progress, 0), 100)  # Clamp between 0 and 100

            days_left = (target_date - pd.Timestamp.now()).days

            st.markdown(f"**{metric} Goal**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.progress(progress/100)
            with col2:
                st.write(f"Current: {current:.1f} / Target: {target:.1f}")
            with col3:
                st.write(f"Days Left: {days_left}")


def main():
    """Main application entry point."""
    setup_database()

    st.title("📊 Body Metrics Dashboard")

    tabs = st.tabs(["📊 Dashboard", "📝 Entry",
                   "🎯 Goals", "📸 Photos", "📈 Analysis"])

    with tabs[0]:
        show_dashboard_tab()
    with tabs[1]:
        show_metrics_entry_tab()
    with tabs[2]:
        show_goals_tab()
    with tabs[3]:
        show_photos_tab()
    with tabs[4]:
        show_analysis_tab()


if __name__ == "__main__":
    main()
