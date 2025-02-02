import streamlit as st
import datetime
from db_utils import create_connection


def main():
    st.header("Add Workout with Presets")

    # Fetch existing exercises from DB
    conn = create_connection()
    c = conn.cursor()
    c.execute("SELECT id, name FROM exercises ORDER BY name ASC")
    all_exercises = c.fetchall()  # list of tuples (id, name)
    conn.close()

    exercise_names = [row[1] for row in all_exercises]

    # Create two tabs
    tab_log, tab_manage = st.tabs(["Log Workout", "Manage Workout Types"])

    # ---------------------- TAB 1: Log Workout -----------------------
    with tab_log:
        st.subheader("Log a Workout Entry")

        # If we have existing presets, show them in a selectbox
        if exercise_names:
            selected_workout = st.selectbox(
                "Select a Preset Workout", exercise_names)
        else:
            st.info("No presets yet. Add one in the 'Manage Workout Types' tab!")
            selected_workout = None

        # Also allow a custom name input if user wants a one-off
        custom_workout = st.text_input(
            "Or type a custom workout name (optional)")

        # Determine final workout name
        final_workout = custom_workout.strip() if custom_workout.strip() else selected_workout

        # Other workout details
        weight = st.number_input(
            "Weight Used (lbs)", min_value=0.0, max_value=2000.0, step=5.0)
        sets = st.number_input("Sets", min_value=1, max_value=50, step=1)
        reps = st.number_input(
            "Reps (per set)", min_value=1, max_value=100, step=1)
        workout_date = st.date_input("Date", datetime.date.today())

        if st.button("Add Workout Record"):
            if final_workout:
                conn = create_connection()
                c = conn.cursor()
                c.execute("""
                    INSERT INTO workouts (workout, weight, sets, reps, date)
                    VALUES (?, ?, ?, ?, ?)
                """, (final_workout, weight, sets, reps, str(workout_date)))
                conn.commit()
                conn.close()
                st.success(
                    f"Workout added: {final_workout} on {workout_date}.")
            else:
                st.warning(
                    "Please select or enter a workout name before adding.")

    # ---------------------- TAB 2: Manage Workout Types -----------------------
    with tab_manage:
        st.subheader("Add New Preset, Rename, or Delete")

        # 1) Add a new workout type
        st.write("### Add a New Workout Type")
        new_exercise = st.text_input(
            "e.g., 'Barbell Rows', 'Sprints', 'Leg Press'", key="new_exercise")

        if st.button("Add New Workout Type"):
            if new_exercise.strip():
                conn = create_connection()
                c = conn.cursor()
                try:
                    c.execute("INSERT INTO exercises (name) VALUES (?)",
                              (new_exercise.strip(),))
                    conn.commit()
                    st.success(
                        f"'{new_exercise.strip()}' has been added to your presets!")
                except Exception as e:
                    st.error(
                        f"Could not add '{new_exercise.strip()}'. Error: {e}")
                conn.close()
            else:
                st.warning("Please enter a valid name before adding.")

        st.write("---")

        # 2) Edit or delete existing presets
        st.write("### Edit or Delete a Preset")
        if exercise_names:
            selected_for_edit = st.selectbox(
                "Select a Preset to Manage", exercise_names, key="edit_select")
            action = st.radio(
                "Action", ["Rename", "Delete"], key="edit_action")
            new_name_input = st.text_input(
                "New name (if renaming)", key="rename_input")

            if st.button("Apply Changes", key="apply_changes"):
                conn = create_connection()
                c = conn.cursor()
                if action == "Delete":
                    c.execute("DELETE FROM exercises WHERE name = ?",
                              (selected_for_edit,))
                    st.warning(f"Deleted '{selected_for_edit}' from presets!")
                elif action == "Rename":
                    if not new_name_input.strip():
                        st.error("Please enter a new name to rename.")
                    else:
                        try:
                            c.execute("UPDATE exercises SET name=? WHERE name=?",
                                      (new_name_input.strip(), selected_for_edit))
                            st.success(
                                f"Renamed '{selected_for_edit}' to '{new_name_input.strip()}'!")
                        except Exception as e:
                            st.error(f"Could not rename. Error: {e}")
                conn.commit()
                conn.close()
        else:
            st.info("No presets to edit or delete yet.")


if __name__ == "__main__":
    main()
