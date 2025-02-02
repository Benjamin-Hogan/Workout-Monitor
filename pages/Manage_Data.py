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
                            SET workout = ?, weight = ?, sets = ?, reps = ?, date = ?
                            WHERE id = ?
                        """
                        cursor.execute(update_query, (
                            edited_row["workout"],
                            edited_row["weight"],
                            edited_row["sets"],
                            edited_row["reps"],
                            str(edited_row["date"]),
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

    # -----------------------------------
    # Upload Workouts from CSV or Excel
    # -----------------------------------
    st.write("---")
    st.subheader("Import Workout Data")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file to import workouts",
        type=['csv', 'xlsx']
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)

            required_columns = {"workout", "weight", "sets", "reps", "date"}
            if not required_columns.issubset(new_df.columns):
                st.error(
                    f"Uploaded file must contain these columns: {required_columns}")
            else:
                # Convert date to string to store in DB
                new_df["date"] = pd.to_datetime(
                    new_df["date"], errors="coerce").dt.date.astype(str)

                # Insert each row into the database
                with create_connection() as conn:
                    cursor = conn.cursor()
                    insert_query = """
                        INSERT INTO workouts (workout, weight, sets, reps, date)
                        VALUES (?, ?, ?, ?, ?)
                    """
                    for _, row in new_df.iterrows():
                        cursor.execute(insert_query, (
                            row["workout"],
                            row["weight"],
                            row["sets"],
                            row["reps"],
                            row["date"]
                        ))
                    conn.commit()
                st.success("Workout data imported successfully!")
                # Refresh the dataframe after import
                with create_connection() as conn:
                    df = pd.read_sql_query("SELECT * FROM workouts", conn)
                    if not df.empty:
                        df["date"] = pd.to_datetime(
                            df["date"], errors="coerce").dt.date
        except Exception as e:
            st.error(f"An error occurred while importing data: {e}")


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
            df, num_rows="dynamic", use_container_width=True
        )

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

    # -----------------------------------
    # Bulk Delete Body Metrics
    # -----------------------------------
    if not df.empty:
        st.write("---")
        st.subheader("Delete Body Metrics Rows")

        # Multiselect for selecting multiple IDs to delete
        delete_ids = st.multiselect(
            "Select row ID(s) to delete",
            options=df["id"].tolist(),
            format_func=lambda x: f"ID {x}"
        )

        if st.button("Delete Selected Body Metrics Rows"):
            if delete_ids:
                with create_connection() as conn:
                    cursor = conn.cursor()
                    placeholders = ','.join(['?'] * len(delete_ids))
                    delete_query = f"DELETE FROM body_metrics WHERE id IN ({placeholders})"
                    cursor.execute(delete_query, tuple(delete_ids))
                    conn.commit()
                st.success(
                    f"Deleted row ID(s): {', '.join(map(str, delete_ids))}")
                # Refresh the dataframe after deletion
                with create_connection() as conn:
                    df = pd.read_sql_query("SELECT * FROM body_metrics", conn)
                    if not df.empty:
                        df["entry_date"] = pd.to_datetime(
                            df["entry_date"], errors="coerce").dt.date
            else:
                st.warning("Please select at least one row ID to delete.")

    # -----------------------------------
    # Export Body Metrics as CSV
    # -----------------------------------
    if not df.empty:
        st.write("---")
        st.write("### Export Body Metrics Data")
        csv_body_metrics = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Body Metrics as CSV",
            data=csv_body_metrics,
            file_name='body_metrics.csv',
            mime='text/csv',
        )

    # -----------------------------------
    # Upload Body Metrics from CSV or Excel
    # -----------------------------------
    st.write("---")
    st.subheader("Import Body Metrics Data")

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file to import body metrics",
        type=['csv', 'xlsx']
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)

            required_columns = {"entry_date", "user_weight", "height"}
            # Include optional columns if present
            optional_columns = {"body_fat", "chest", "waist", "hips", "arms"}
            all_required = required_columns.union(optional_columns)
            if not required_columns.issubset(new_df.columns):
                st.error(
                    f"Uploaded file must contain at least these columns: {required_columns}")
            else:
                # Fill missing optional columns with None
                for col in optional_columns:
                    if col not in new_df.columns:
                        new_df[col] = None

                # Convert entry_date to string to store in DB
                new_df["entry_date"] = pd.to_datetime(
                    new_df["entry_date"], errors="coerce").dt.date.astype(str)

                # Insert each row into the database
                with create_connection() as conn:
                    cursor = conn.cursor()
                    insert_query = """
                        INSERT INTO body_metrics (entry_date, user_weight, height, body_fat, chest, waist, hips, arms)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    for _, row in new_df.iterrows():
                        cursor.execute(insert_query, (
                            row["entry_date"],
                            row["user_weight"],
                            row["height"],
                            row["body_fat"] if pd.notna(
                                row["body_fat"]) else None,
                            row["chest"] if pd.notna(row["chest"]) else None,
                            row["waist"] if pd.notna(row["waist"]) else None,
                            row["hips"] if pd.notna(row["hips"]) else None,
                            row["arms"] if pd.notna(row["arms"]) else None,
                        ))
                    conn.commit()
                st.success("Body Metrics data imported successfully!")
                # Refresh the dataframe after import
                with create_connection() as conn:
                    df = pd.read_sql_query("SELECT * FROM body_metrics", conn)
                    if not df.empty:
                        df["entry_date"] = pd.to_datetime(
                            df["entry_date"], errors="coerce").dt.date
        except Exception as e:
            st.error(f"An error occurred while importing data: {e}")


if __name__ == "__main__":
    main()
