import streamlit as st
import pandas as pd
import numpy as np
from db_utils import create_connection


def main():
    st.header("Manage Data")

    tab1, tab2 = st.tabs(["Workouts", "Body Metrics"])

    with tab1:
        manage_workouts()

    with tab2:
        manage_body_metrics()


def manage_workouts():
    st.subheader("Manage Workout Data")

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
            df, num_rows="dynamic", use_container_width=True)

        if st.button("Save Body Metrics Changes"):
            with create_connection() as conn:
                cursor = conn.cursor()
                for i in range(len(df)):
                    original_row = df.loc[i]
                    edited_row = edited_df.loc[i]

                    if not original_row.equals(edited_row):
                        update_query = """
                            UPDATE body_metrics
                            SET entry_date = ?, user_weight = ?, height = ?,
                                body_fat = ?, chest = ?, waist = ?, hips = ?, arms = ?
                            WHERE id = ?
                        """
                        cursor.execute(update_query, (
                            str(edited_row["entry_date"]),
                            edited_row["user_weight"],
                            edited_row["height"],
                            edited_row["body_fat"],
                            edited_row["chest"],
                            edited_row["waist"],
                            edited_row["hips"],
                            edited_row["arms"],
                            edited_row["id"]
                        ))
                conn.commit()
            st.success("Body Metrics updated successfully!")


if __name__ == "__main__":
    main()
