import streamlit as st
import datetime
from db_utils import create_connection


def main():
    st.header("Add Workout with Presets")

    # Fetch existing exercises from DB
    conn = create_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, name, muscle_type, workout_type FROM exercises ORDER BY name ASC"
    )
    all_exercises = c.fetchall()  # list of tuples (id, name, muscle_type, workout_type)
    conn.close()

    exercise_dict = {
        row[1]: {"id": row[0], "muscle_type": row[2], "workout_type": row[3]}
        for row in all_exercises
    }
    exercise_names = list(exercise_dict.keys())

    # Create two tabs
    tab_log, tab_manage = st.tabs(["Log Workout", "Manage Workout Types"])

    # ---------------------- TAB 1: Log Workout -----------------------
    with tab_log:
        st.subheader("Log a Workout Entry")

        # Select preset workout (inherits muscle_type and workout_type)
        selected_workout = (
            st.selectbox("Select a Preset Workout",
                         exercise_names) if exercise_names else None
        )

        if selected_workout:
            # Retrieve attributes from selected workout
            muscle_type = exercise_dict[selected_workout]["muscle_type"]
            workout_type = exercise_dict[selected_workout]["workout_type"]
        else:
            muscle_type, workout_type = None, None

        # Other workout details
        weight = st.number_input(
            "Weight Used (lbs)", min_value=0.0, max_value=2000.0, step=5.0)
        sets = st.number_input("Sets", min_value=1, max_value=50, step=1)
        reps = st.number_input(
            "Reps (per set)", min_value=1, max_value=100, step=1)
        workout_date = st.date_input("Date", datetime.date.today())

        if st.button("Add Workout Record"):
            if selected_workout:
                conn = create_connection()
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO workouts (workout, weight, sets, reps, date, muscle_type, workout_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (selected_workout, weight, sets, reps, str(
                        workout_date), muscle_type, workout_type),
                )
                conn.commit()
                conn.close()
                st.success(
                    f"Workout added: {selected_workout} on {workout_date}.")
            else:
                st.warning("Please select a workout preset before adding.")

    # ---------------------- TAB 2: Manage Workout Types -----------------------
    with tab_manage:
        st.subheader("Add New Preset, Rename, or Delete")

        # 1) Add a new workout type with Muscle Type and Workout Type
        st.write("### Add a New Workout Type")
        new_exercise = st.text_input("Workout Name", key="new_exercise")
        new_muscle_type = st.selectbox(
            "Muscle Type", ["Chest", "Back", "Legs",
                            "Arms", "Shoulders", "Core", "Full Body"]
        )
        new_workout_type = st.selectbox(
            "Workout Type", ["Push", "Pull", "Leg", "Full-Body", "Core"]
        )

        if st.button("Add New Workout Type"):
            if new_exercise.strip():
                conn = create_connection()
                c = conn.cursor()
                try:
                    c.execute(
                        """
                        INSERT INTO exercises (name, muscle_type, workout_type) 
                        VALUES (?, ?, ?)
                        """,
                        (new_exercise.strip(), new_muscle_type, new_workout_type),
                    )
                    conn.commit()
                    st.success(
                        f"'{new_exercise.strip()}' added with {new_muscle_type} - {new_workout_type}."
                    )
                except Exception as e:
                    st.error(
                        f"Could not add '{new_exercise.strip()}'. Error: {e}")
                conn.close()
            else:
                st.warning("Please enter a valid workout name before adding.")

        st.write("---")

        # 2) Edit or delete existing presets
        st.write("### Edit or Delete a Preset")
        if exercise_names:
            selected_for_edit = st.selectbox(
                "Select a Preset to Manage", exercise_names, key="edit_select"
            )
            action = st.radio(
                "Action", ["Rename", "Delete", "Change Attributes"], key="edit_action")
            new_name_input = st.text_input(
                "New name (if renaming)", key="rename_input")
            updated_muscle_type = st.selectbox(
                "New Muscle Type", ["Chest", "Back", "Legs",
                                    "Arms", "Shoulders", "Core", "Full Body"]
            )
            updated_workout_type = st.selectbox(
                "New Workout Type", ["Push", "Pull",
                                     "Leg", "Full-Body", "Core"]
            )

            if st.button("Apply Changes", key="apply_changes"):
                conn = create_connection()
                c = conn.cursor()
                if action == "Delete":
                    c.execute("DELETE FROM exercises WHERE name = ?",
                              (selected_for_edit,))
                    c.execute("DELETE FROM workouts WHERE workout = ?",
                              (selected_for_edit,))
                    st.warning(
                        f"Deleted '{selected_for_edit}' from presets and all associated workouts!")
                elif action == "Rename":
                    if not new_name_input.strip():
                        st.error("Please enter a new name to rename.")
                    else:
                        c.execute("UPDATE exercises SET name=? WHERE name=?",
                                  (new_name_input.strip(), selected_for_edit))
                        c.execute("UPDATE workouts SET workout=? WHERE workout=?",
                                  (new_name_input.strip(), selected_for_edit))
                        st.success(
                            f"Renamed '{selected_for_edit}' to '{new_name_input.strip()}'.")
                elif action == "Change Attributes":
                    # Update exercises table
                    c.execute(
                        """
                        UPDATE exercises 
                        SET muscle_type=?, workout_type=? 
                        WHERE name=?
                        """,
                        (updated_muscle_type, updated_workout_type, selected_for_edit),
                    )

                    # Cascade update in workouts table
                    c.execute(
                        """
                        UPDATE workouts 
                        SET muscle_type=?, workout_type=? 
                        WHERE workout=?
                        """,
                        (updated_muscle_type, updated_workout_type, selected_for_edit),
                    )

                    st.success(
                        f"Updated '{selected_for_edit}' to {updated_muscle_type} - {updated_workout_type} in both workouts and presets."
                    )
                conn.commit()
                conn.close()
        else:
            st.info("No presets to edit or delete yet.")


if __name__ == "__main__":
    main()
