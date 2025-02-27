from utils import estimate_body_fat
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import datetime
import time
import json
import io
import base64
from db_utils import create_connection
from auth import check_password

# Add parent directory to path to import utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# First check authentication
if not check_password():
    st.stop()

# Set page config for full width
st.set_page_config(layout="wide")


def main():
    st.header("Manage Data")

    tab1, tab2 = st.tabs(["Workouts", "Body Metrics"])

    with tab1:
        manage_workouts()

    with tab2:
        manage_body_metrics()


def import_csv_to_workouts(csv_file):
    """Import workout data from a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Validate required columns
        required_columns = ['workout', 'weight', 'sets', 'reps', 'date']
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"

        # Convert date format if needed
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Add missing columns with default values if they don't exist
        if 'muscle_type' not in df.columns:
            df['muscle_type'] = None
        if 'workout_type' not in df.columns:
            df['workout_type'] = None

        # Insert data into database
        with create_connection() as conn:
            cursor = conn.cursor()

            # Get existing exercises to update the exercises table if needed
            cursor.execute("SELECT name FROM exercises")
            existing_exercises = {row[0] for row in cursor.fetchall()}

            # Insert new exercises if they don't exist
            new_exercises = []
            for workout in df['workout'].unique():
                if workout not in existing_exercises:
                    # Get muscle_type and workout_type for this exercise if available
                    exercise_data = df[df['workout'] == workout].iloc[0]
                    muscle_type = exercise_data.get('muscle_type')
                    workout_type = exercise_data.get('workout_type')

                    new_exercises.append((workout, muscle_type, workout_type))

            if new_exercises:
                cursor.executemany(
                    "INSERT INTO exercises (name, muscle_type, workout_type) VALUES (?, ?, ?)",
                    new_exercises
                )

            # Insert workout data
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO workouts (workout, weight, sets, reps, date, muscle_type, workout_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['workout'],
                    row['weight'],
                    row['sets'],
                    row['reps'],
                    row['date'],
                    row.get('muscle_type'),
                    row.get('workout_type')
                ))

            conn.commit()

            # Increment the database modification counter
            if "db_modification_counter" in st.session_state:
                st.session_state.db_modification_counter += 1
            else:
                st.session_state.db_modification_counter = 1

        return True, f"Successfully imported {len(df)} workout records."

    except Exception as e:
        return False, f"Error importing data: {str(e)}"


def import_csv_to_body_metrics(csv_file):
    """Import body metrics data from a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Print column names for debugging
        st.write("CSV columns:", df.columns.tolist())

        # Validate required columns
        required_columns = ['entry_date', 'user_weight']
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"

        # Convert date format if needed
        df['entry_date'] = pd.to_datetime(
            df['entry_date']).dt.strftime('%Y-%m-%d')

        # Check for empty strings and convert to None/NaN
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'entry_date' and col != 'gender':
                df[col] = df[col].replace('', np.nan)

        # Display the DataFrame after initial processing
        st.write("DataFrame after initial processing:")
        st.dataframe(df.head(2))

        # Insert data into database
        with create_connection() as conn:
            cursor = conn.cursor()

            # Insert body metrics data
            success_count = 0
            error_count = 0

            for idx, row in df.iterrows():
                try:
                    # Convert values to appropriate types, handling NaN values
                    user_weight = float(row['user_weight']) if pd.notna(
                        row['user_weight']) else None
                    height = float(row['height']) if 'height' in row and pd.notna(
                        row['height']) else None
                    age = int(float(row['age'])) if 'age' in row and pd.notna(
                        row['age']) else None
                    gender = str(row['gender']) if 'gender' in row and pd.notna(
                        row['gender']) else None
                    body_fat = float(row['body_fat']) if 'body_fat' in row and pd.notna(
                        row['body_fat']) else None
                    chest = float(row['chest']) if 'chest' in row and pd.notna(
                        row['chest']) else None
                    waist = float(row['waist']) if 'waist' in row and pd.notna(
                        row['waist']) else None
                    hips = float(row['hips']) if 'hips' in row and pd.notna(
                        row['hips']) else None
                    arms = float(row['arms']) if 'arms' in row and pd.notna(
                        row['arms']) else None
                    glutes = float(row['glutes']) if 'glutes' in row and pd.notna(
                        row['glutes']) else None
                    thigh = float(row['thigh']) if 'thigh' in row and pd.notna(
                        row['thigh']) else None
                    calf = float(row['calf']) if 'calf' in row and pd.notna(
                        row['calf']) else None
                    neck = float(row['neck']) if 'neck' in row and pd.notna(
                        row['neck']) else None

                    # Debug the first row's values
                    if idx == 0:
                        st.write("First row values after conversion:")
                        st.write(f"entry_date: {row['entry_date']}")
                        st.write(f"user_weight: {user_weight}")
                        st.write(f"height: {height}")
                        st.write(f"age: {age}")
                        st.write(f"gender: {gender}")
                        st.write(f"body_fat: {body_fat}")
                        st.write(f"chest: {chest}")
                        st.write(f"waist: {waist}")
                        st.write(f"hips: {hips}")
                        st.write(f"arms: {arms}")
                        st.write(f"glutes: {glutes}")
                        st.write(f"thigh: {thigh}")
                        st.write(f"calf: {calf}")
                        st.write(f"neck: {neck}")

                    # Insert into database
                    cursor.execute("""
                        INSERT INTO body_metrics (
                            entry_date, user_weight, height, age, gender, body_fat, 
                            chest, waist, hips, arms, glutes, thigh, calf, neck
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['entry_date'],
                        user_weight,
                        height,
                        age,
                        gender,
                        body_fat,
                        chest,
                        waist,
                        hips,
                        arms,
                        glutes,
                        thigh,
                        calf,
                        neck
                    ))
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    st.error(
                        f"Error inserting row {idx+1} with date {row['entry_date']}: {e}")
                    st.write(f"Row data: {row.to_dict()}")

            conn.commit()

            # Increment the database modification counter
            if "db_modification_counter" in st.session_state:
                st.session_state.db_modification_counter += 1
            else:
                st.session_state.db_modification_counter = 1

            if error_count > 0:
                st.warning(f"{error_count} rows had errors during import.")

        return True, f"Successfully imported {success_count} body metrics records."

    except Exception as e:
        st.error(f"Error details: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False, f"Error importing data: {str(e)}"


def manage_workouts():
    st.subheader("Manage Workout Data")

    # -----------------------------------
    # Import Workouts from CSV
    # -----------------------------------
    st.write("### Import Workout Data")

    with st.expander("Import from CSV", expanded=False):
        st.write("""
        Upload a CSV file with workout data. The file must contain the following columns:
        - workout (exercise name)
        - weight (in lbs)
        - sets (number of sets)
        - reps (number of reps)
        - date (in YYYY-MM-DD format)
        
        Optional columns:
        - muscle_type (Chest, Back, Legs, etc.)
        - workout_type (Push, Pull, Leg, etc.)
        """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file", type="csv", key="workout_csv_uploader")

        if uploaded_file is not None:
            # Show a preview of the CSV
            df_preview = pd.read_csv(uploaded_file)
            st.write("Preview of CSV data:")
            st.dataframe(df_preview.head(5), use_container_width=True)

            # Reset file pointer to beginning
            uploaded_file.seek(0)

            if st.button("Import Data", key="import_workout_button"):
                with st.spinner("Importing data..."):
                    success, message = import_csv_to_workouts(uploaded_file)

                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

    # Fetch data from the database
    with create_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM workouts", conn)

    if df.empty:
        st.info("No workout data found.")
    else:
        # Convert date to datetime for editing
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        # Display the data editor
        edited_df = st.data_editor(
            df, num_rows="dynamic", use_container_width=True
        )
        # `data_editor` returns a DataFrame with potential user edits

        if st.button("Save Workout Changes"):
            with create_connection() as conn:
                cursor = conn.cursor()
                # Compare edited_df with original df, update DB for changed rows
                for i in range(len(df)):
                    original_row = df.loc[i]
                    edited_row = edited_df.loc[i]

                    # Check if something changed
                    if not original_row.equals(edited_row):
                        # Update the DB using the row's ID
                        update_query = """
                            UPDATE workouts
                            SET workout = ?, weight = ?, sets = ?, reps = ?, date = ?, muscle_type = ?, workout_type = ?
                            WHERE id = ?
                        """
                        cursor.execute(update_query, (
                            edited_row["workout"],
                            edited_row["weight"],
                            edited_row["sets"],
                            edited_row["reps"],
                            str(edited_row["date"]),
                            edited_row["muscle_type"],
                            edited_row["workout_type"],
                            edited_row["id"]
                        ))
                conn.commit()

                # Increment the database modification counter
                if "db_modification_counter" in st.session_state:
                    st.session_state.db_modification_counter += 1
                else:
                    st.session_state.db_modification_counter = 1

            st.success("Workout table updated successfully!")

    # -----------------------------------
    # Bulk Delete Workouts
    # -----------------------------------
    if not df.empty:
        st.write("---")
        st.subheader("Delete Workout Rows")

        # Multiselect for selecting multiple IDs to delete
        delete_ids = st.multiselect(
            "Select row ID(s) to delete",
            options=df["id"].tolist(),
            format_func=lambda x: f"ID {x}"
        )

        if st.button("Delete Selected Workout Rows"):
            if delete_ids:
                with create_connection() as conn:
                    cursor = conn.cursor()
                    placeholders = ','.join(['?'] * len(delete_ids))
                    delete_query = f"DELETE FROM workouts WHERE id IN ({placeholders})"
                    cursor.execute(delete_query, tuple(delete_ids))
                    conn.commit()

                    # Increment the database modification counter
                    if "db_modification_counter" in st.session_state:
                        st.session_state.db_modification_counter += 1
                    else:
                        st.session_state.db_modification_counter = 1

                st.success(
                    f"Deleted row ID(s): {', '.join(map(str, delete_ids))}")
                # Refresh the dataframe after deletion
                with create_connection() as conn:
                    df = pd.read_sql_query("SELECT * FROM workouts", conn)
                    if not df.empty:
                        df["date"] = pd.to_datetime(
                            df["date"], errors="coerce").dt.date
            else:
                st.warning("Please select at least one row ID to delete.")

    # -----------------------------------
    # Update Workout Types (Cascade Changes)
    # -----------------------------------
    st.write("---")
    st.subheader("Update Workout Attributes")

    # Fetch existing exercises
    with create_connection() as conn:
        exercise_df = pd.read_sql_query("SELECT * FROM exercises", conn)

    if not exercise_df.empty:
        workout_to_update = st.selectbox(
            "Select a workout to update", exercise_df["name"].tolist())

        # Get the existing attributes
        existing_muscle_type = exercise_df.loc[exercise_df["name"]
                                               == workout_to_update, "muscle_type"].values[0]
        existing_workout_type = exercise_df.loc[exercise_df["name"]
                                                == workout_to_update, "workout_type"].values[0]

        updated_muscle_type = st.selectbox("New Muscle Type", [
            "Chest", "Back", "Legs", "Arms", "Shoulders", "Core", "Full Body"
        ], index=["Chest", "Back", "Legs", "Arms", "Shoulders", "Core", "Full Body"].index(existing_muscle_type))

        updated_workout_type = st.selectbox("New Workout Type", [
            "Push", "Pull", "Leg", "Full-Body", "Core"
        ], index=["Push", "Pull", "Leg", "Full-Body", "Core"].index(existing_workout_type))

        if st.button("Update Workout Type Across Records"):
            with create_connection() as conn:
                cursor = conn.cursor()

                # Update exercises table
                cursor.execute("""
                    UPDATE exercises
                    SET muscle_type = ?, workout_type = ?
                    WHERE name = ?
                """, (updated_muscle_type, updated_workout_type, workout_to_update))

                # Cascade update in workouts table
                cursor.execute("""
                    UPDATE workouts
                    SET muscle_type = ?, workout_type = ?
                    WHERE workout = ?
                """, (updated_muscle_type, updated_workout_type, workout_to_update))

                conn.commit()

                # Increment the database modification counter
                if "db_modification_counter" in st.session_state:
                    st.session_state.db_modification_counter += 1
                else:
                    st.session_state.db_modification_counter = 1

            st.success(
                f"Updated '{workout_to_update}' to {updated_muscle_type} - {updated_workout_type}.")

    # -----------------------------------
    # Export Workouts as CSV
    # -----------------------------------
    if not df.empty:
        st.write("---")
        st.write("### Export Workout Data")
        csv_workouts = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Workouts as CSV",
            data=csv_workouts,
            file_name='workouts.csv',
            mime='text/csv',
        )


def manage_body_metrics():
    st.subheader("Manage Body Metrics Data")

    # -----------------------------------
    # Import Body Metrics from CSV
    # -----------------------------------
    st.write("### Import Body Metrics Data")

    with st.expander("Import from CSV", expanded=False):
        st.write("""
        Upload a CSV file with body metrics data. The file must contain the following columns:
        - entry_date (in YYYY-MM-DD format)
        - user_weight (in lbs)
        
        Optional columns:
        - height (in inches)
        - age
        - gender
        - body_fat (percentage)
        - chest, waist, hips, arms, glutes, thigh, calf, neck (all in inches)
        
        Note: If body_fat is not provided but height, age, and gender are available, 
        it will be automatically calculated.
        """)

        # Option to import from file upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type="csv", key="metrics_csv_uploader")

        if uploaded_file is not None:
            # Show a preview of the CSV
            df_preview = pd.read_csv(uploaded_file)
            st.write("Preview of CSV data:")
            st.dataframe(df_preview.head(5), use_container_width=True)

            # Reset file pointer to beginning
            uploaded_file.seek(0)

            if st.button("Import Data", key="import_metrics_button"):
                with st.spinner("Importing data..."):
                    success, message = import_csv_to_body_metrics(
                        uploaded_file)

                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        # Option to import from a specific file path
        st.write("### Or Import from File Path")
        if st.button("Import from body_metrics (1).csv"):
            try:
                file_path = "body_metrics (1).csv"
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        success, message = import_csv_to_body_metrics(f)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.error(f"File not found: {file_path}")
            except Exception as e:
                st.error(f"Error importing from file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Fetch data from the database
    with create_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM body_metrics", conn)

    if df.empty:
        st.info("No body metrics data found.")
    else:
        df["entry_date"] = pd.to_datetime(
            df["entry_date"], errors="coerce").dt.date

        # Display the data editor
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn(
                    "ID",
                    help="Record ID",
                    disabled=True,
                ),
                "entry_date": st.column_config.DateColumn(
                    "Date",
                    help="Entry date",
                ),
                "user_weight": st.column_config.NumberColumn(
                    "Weight (lbs)",
                    help="Body weight in pounds",
                    min_value=0,
                    format="%.1f"
                ),
                "height": st.column_config.NumberColumn(
                    "Height (in)",
                    help="Height in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "body_fat": st.column_config.NumberColumn(
                    "Body Fat %",
                    help="Body fat percentage",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "chest": st.column_config.NumberColumn(
                    "Chest (in)",
                    help="Chest measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "waist": st.column_config.NumberColumn(
                    "Waist (in)",
                    help="Waist measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "hips": st.column_config.NumberColumn(
                    "Hips (in)",
                    help="Hips measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "arms": st.column_config.NumberColumn(
                    "Arms (in)",
                    help="Arms measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "glutes": st.column_config.NumberColumn(
                    "Glutes (in)",
                    help="Glutes measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "thigh": st.column_config.NumberColumn(
                    "Thigh (in)",
                    help="Thigh measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "calf": st.column_config.NumberColumn(
                    "Calf (in)",
                    help="Calf measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
                "neck": st.column_config.NumberColumn(
                    "Neck (in)",
                    help="Neck measurement in inches",
                    min_value=0,
                    format="%.1f"
                ),
            },
            disabled=["id"],
            key="body_metrics_editor"
        )

        # Handle auto-fill for weight
        if 'user_weight' in edited_df.columns:
            mask = edited_df['user_weight'].isna()
            if mask.any():
                # For each row with missing weight, find the most recent previous weight
                for idx in edited_df[mask].index:
                    date = edited_df.loc[idx, 'entry_date']
                    # Find the most recent weight before this date
                    prev_weights = edited_df[
                        (edited_df['entry_date'] < date) &
                        (edited_df['user_weight'].notna())
                    ]['user_weight']

                    if not prev_weights.empty:
                        prev_weight = prev_weights.iloc[-1]
                        edited_df.loc[idx, 'user_weight'] = prev_weight
                        st.info(
                            f"Auto-filled weight for {date} with previous weight: {prev_weight}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes", type="primary"):
                try:
                    with create_connection() as conn:
                        cursor = conn.cursor()
                        for idx, row in edited_df.iterrows():
                            original_row = df.loc[df['id'] == row['id']
                                                  ].iloc[0] if row['id'] in df['id'].values else None

                            if original_row is None or not row.equals(original_row):
                                update_query = """
                                    UPDATE body_metrics
                                    SET entry_date = ?, user_weight = ?, height = ?,
                                        age = ?, gender = ?, body_fat = ?, 
                                        chest = ?, waist = ?, hips = ?, arms = ?,
                                        glutes = ?, thigh = ?, calf = ?, neck = ?
                                    WHERE id = ?
                                """
                                cursor.execute(update_query, (
                                    str(row["entry_date"]),
                                    row["user_weight"],
                                    row["height"],
                                    row["age"],
                                    row["gender"],
                                    row["body_fat"],
                                    row["chest"],
                                    row["waist"],
                                    row["hips"],
                                    row["arms"],
                                    row["glutes"],
                                    row["thigh"],
                                    row["calf"],
                                    row["neck"],
                                    row["id"]
                                ))
                        conn.commit()

                        # Increment the database modification counter for metrics too
                        if "db_modification_counter" in st.session_state:
                            st.session_state.db_modification_counter += 1
                        else:
                            st.session_state.db_modification_counter = 1

                    st.success("Changes saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        with col2:
            if st.button("Revert Changes"):
                st.rerun()

        # -----------------------------------
        # Bulk Delete Body Metrics
        # -----------------------------------
        st.write("---")
        st.subheader("Delete Body Metrics Rows")

        # Multiselect for selecting multiple IDs to delete
        delete_ids = st.multiselect(
            "Select row ID(s) to delete",
            options=df["id"].tolist(),
            format_func=lambda x: f"ID {x} ({df.loc[df['id'] == x, 'entry_date'].iloc[0]})"
        )

        if st.button("Delete Selected Rows"):
            if delete_ids:
                with create_connection() as conn:
                    cursor = conn.cursor()
                    placeholders = ','.join(['?'] * len(delete_ids))
                    delete_query = f"DELETE FROM body_metrics WHERE id IN ({placeholders})"
                    cursor.execute(delete_query, tuple(delete_ids))
                    conn.commit()

                    # Increment the database modification counter
                    if "db_modification_counter" in st.session_state:
                        st.session_state.db_modification_counter += 1
                    else:
                        st.session_state.db_modification_counter = 1

                st.success(
                    f"Deleted row ID(s): {', '.join(map(str, delete_ids))}")
                st.rerun()  # Refresh the page after deletion
            else:
                st.warning("Please select at least one row ID to delete.")

        # Add export functionality
        st.write("---")
        st.subheader("Export Data")

        # Export as CSV - use the original df instead of edited_df to ensure all data is exported
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name='body_metrics.csv',
            mime='text/csv',
        )


if __name__ == "__main__":
    main()
