import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from db_utils import create_connection
import plotly.figure_factory as ff
import plotly.io as pio
import numpy as np


def main():
    st.header("📈 Body Metrics Dashboard")

    st.write(
        "Log or view your body metrics, including weight, height, age, gender, and optional measurements like chest, waist, hips, and arms. Choose your preferred measurement system below."
    )

    # Measurement system preference
    system = st.radio("📐 Measurement System", [
                      "Imperial (lbs/in)", "Metric (kg/cm)"])

    st.write("---")

    # Add or Update Body Metrics Section
    with st.form("add_body_metrics_form"):
        st.subheader("📝 Add or Update Body Metrics")

        # Date Input
        entry_date = st.date_input("📅 Date", datetime.date.today())

        # Initialize variables
        user_weight = None
        height = None
        age = None
        gender = None
        body_fat = None
        chest = None
        waist = None
        hips = None
        arms = None

        # Layout inputs using columns for better UI
        col1, col2 = st.columns(2)

        with col1:
            if system == "Imperial (lbs/in)":
                user_weight = st.number_input(
                    "🏋️ Body Weight (lbs)", min_value=0.0, max_value=2000.0, step=1.0, format="%.1f"
                )
                height_input = st.number_input(
                    "📏 Height (in) [Optional]", min_value=0.0, max_value=120.0, step=0.1, format="%.1f"
                )
                height = height_input if height_input > 0 else None
            else:
                w_kg = st.number_input(
                    "🏋️ Body Weight (kg)", min_value=0.0, max_value=1000.0, step=0.1, format="%.1f"
                )
                h_cm_input = st.number_input(
                    "📏 Height (cm) [Optional]", min_value=0.0, max_value=300.0, step=0.1, format="%.1f"
                )
                # Convert to imperial internally for DB
                user_weight = w_kg * 2.20462 if w_kg > 0 else None
                height = h_cm_input * 0.393701 if h_cm_input > 0 else None

        with col2:
            # Age and Gender Inputs
            age_input = st.text_input("🎂 Age [Optional]")
            gender_input = st.selectbox(
                "🚻 Gender [Optional]",
                options=["Select", "Male", "Female", "Other"],
                index=0
            )

            # Optional Body Measurements
            body_fat_input = st.text_input("🧬 Body Fat (%) [Optional]")
            chest_input = st.text_input("💪 Chest [Optional]")
            waist_input = st.text_input("🩳 Waist [Optional]")
            hips_input = st.text_input("👖 Hips [Optional]")
            arms_input = st.text_input("💪 Arms [Optional]")

        # Process optional inputs
        age = int(age_input) if age_input.strip().isdigit() else None
        gender = gender_input if gender_input != "Select" else None
        body_fat = float(
            body_fat_input) if body_fat_input.strip() != "" else None

        if system == "Imperial (lbs/in)":
            chest = float(chest_input) if chest_input.strip() != "" else None
            waist = float(waist_input) if waist_input.strip() != "" else None
            hips = float(hips_input) if hips_input.strip() != "" else None
            arms = float(arms_input) if arms_input.strip() != "" else None
        else:
            # Convert optional measurements to imperial if provided
            chest = float(chest_input) * \
                0.393701 if chest_input.strip() != "" else None
            waist = float(waist_input) * \
                0.393701 if waist_input.strip() != "" else None
            hips = float(hips_input) * \
                0.393701 if hips_input.strip() != "" else None
            arms = float(arms_input) * \
                0.393701 if arms_input.strip() != "" else None

        # Submit Button
        submit_button = st.form_submit_button("✅ Save Body Metrics")

        if submit_button:
            if user_weight is None:
                st.error("⚠️ Please provide your body weight.")
            else:
                try:
                    with create_connection() as conn:
                        c = conn.cursor()

                        # If height is not provided, retrieve the last known height
                        if height is None:
                            c.execute("""
                                SELECT height FROM body_metrics 
                                WHERE height IS NOT NULL 
                                ORDER BY entry_date DESC 
                                LIMIT 1
                            """)
                            last_height = c.fetchone()
                            if last_height and last_height[0]:
                                height = last_height[0]
                                st.info(
                                    "ℹ️ Height not provided. Using the last recorded height.")
                            else:
                                st.error("⚠️ Please provide your height.")
                                height = None

                        # If age is not provided, retrieve the last known age
                        if age is None:
                            c.execute("""
                                SELECT age FROM body_metrics 
                                WHERE age IS NOT NULL 
                                ORDER BY entry_date DESC 
                                LIMIT 1
                            """)
                            last_age = c.fetchone()
                            if last_age and last_age[0]:
                                age = last_age[0]
                                st.info(
                                    "ℹ️ Age not provided. Using the last recorded age.")
                            else:
                                st.info(
                                    "ℹ️ Age not provided. Body fat estimation will be less accurate.")
                                age = None

                        # If gender is not provided, retrieve the last known gender
                        if gender is None:
                            c.execute("""
                                SELECT gender FROM body_metrics 
                                WHERE gender IS NOT NULL 
                                ORDER BY entry_date DESC 
                                LIMIT 1
                            """)
                            last_gender = c.fetchone()
                            if last_gender and last_gender[0]:
                                gender = last_gender[0]
                                st.info(
                                    "ℹ️ Gender not provided. Using the last recorded gender.")
                            else:
                                st.info(
                                    "ℹ️ Gender not provided. Body fat estimation will be less accurate.")
                                gender = None

                        # If body fat is not provided, estimate it
                        if body_fat is None:
                            body_fat = estimate_body_fat(
                                user_weight, height, age, gender)
                            if body_fat is not None:
                                st.info(
                                    f"📊 Estimated Body Fat: {body_fat:.2f}%")
                            else:
                                st.warning(
                                    "⚠️ Unable to estimate body fat. Please provide age and gender for a better estimate.")

                        # Auto-fill missing body measurements with last recorded values
                        measurements = {
                            'chest': chest, 'waist': waist, 'hips': hips, 'arms': arms}
                        for key, value in measurements.items():
                            if value is None:
                                c.execute(f"""
                                    SELECT {key} FROM body_metrics 
                                    WHERE {key} IS NOT NULL 
                                    ORDER BY entry_date DESC 
                                    LIMIT 1
                                """)
                                last_measure = c.fetchone()
                                if last_measure and last_measure[0]:
                                    measurements[key] = last_measure[0]
                                    st.info(
                                        f"ℹ️ {key.capitalize()} not provided. Using the last recorded value.")
                                else:
                                    # Remain None if no previous record
                                    measurements[key] = None

                        # Insert data into the database
                        c.execute("""
                            INSERT INTO body_metrics (
                                entry_date, user_weight, height, age, gender, body_fat, chest, waist, hips, arms
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(entry_date),
                            user_weight,
                            height,
                            age,
                            gender,
                            body_fat,
                            measurements['chest'],
                            measurements['waist'],
                            measurements['hips'],
                            measurements['arms']
                        ))
                        conn.commit()
                    st.success("✅ Body metrics saved successfully!")
                except Exception as e:
                    st.error(f"❌ An error occurred while saving data: {e}")

    st.write("---")

    # Load and Display Body Metrics
    st.subheader("📊 My Body Metrics Over Time")
    with create_connection() as conn:
        df_metrics = pd.read_sql_query("SELECT * FROM body_metrics", conn)

    if df_metrics.empty:
        st.info("No body metrics logged yet.")
    else:
        # Convert dates
        df_metrics["entry_date"] = pd.to_datetime(df_metrics["entry_date"])

        # Sort by date
        df_metrics = df_metrics.sort_values("entry_date")

        # Determine measurement system based on user preference
        latest_system = system  # Use the current selection

        # Create display columns based on measurement system
        if latest_system == "Metric (kg/cm)":
            # lbs -> kg
            df_metrics["Weight (kg)"] = df_metrics["user_weight"] * 0.453592
            df_metrics["Height (cm)"] = df_metrics["height"] * 2.54  # in -> cm
            df_metrics["Chest (cm)"] = df_metrics["chest"] * 2.54
            df_metrics["Waist (cm)"] = df_metrics["waist"] * 2.54
            df_metrics["Hips (cm)"] = df_metrics["hips"] * 2.54
            df_metrics["Arms (cm)"] = df_metrics["arms"] * 2.54
            weight_label = "Weight (kg)"
            height_label = "Height (cm)"
            chest_label = "Chest (cm)"
            waist_label = "Waist (cm)"
            hips_label = "Hips (cm)"
            arms_label = "Arms (cm)"
        else:
            df_metrics["Weight (lbs)"] = df_metrics["user_weight"]
            df_metrics["Height (in)"] = df_metrics["height"]
            df_metrics["Chest (in)"] = df_metrics["chest"]
            df_metrics["Waist (in)"] = df_metrics["waist"]
            df_metrics["Hips (in)"] = df_metrics["hips"]
            df_metrics["Arms (in)"] = df_metrics["arms"]
            weight_label = "Weight (lbs)"
            height_label = "Height (in)"
            chest_label = "Chest (in)"
            waist_label = "Waist (in)"
            hips_label = "Hips (in)"
            arms_label = "Arms (in)"

        # Select columns to display
        display_columns = [
            "entry_date",
            "Weight (kg)" if latest_system == "Metric (kg/cm)" else "Weight (lbs)",
            "Height (cm)" if latest_system == "Metric (kg/cm)" else "Height (in)",
            "Chest (cm)" if latest_system == "Metric (kg/cm)" else "Chest (in)",
            "Waist (cm)" if latest_system == "Metric (kg/cm)" else "Waist (in)",
            "Hips (cm)" if latest_system == "Metric (kg/cm)" else "Hips (in)",
            "Arms (cm)" if latest_system == "Metric (kg/cm)" else "Arms (in)",
            "body_fat"
        ]

        # Rename columns for display
        display_labels = {
            "entry_date": "Date",
            "Weight (kg)": "Weight (kg)",
            "Weight (lbs)": "Weight (lbs)",
            "Height (cm)": "Height (cm)",
            "Height (in)": "Height (in)",
            "Chest (cm)": "Chest (cm)",
            "Chest (in)": "Chest (in)",
            "Waist (cm)": "Waist (cm)",
            "Waist (in)": "Waist (in)",
            "Hips (cm)": "Hips (cm)",
            "Hips (in)": "Hips (in)",
            "Arms (cm)": "Arms (cm)",
            "Arms (in)": "Arms (in)",
            "body_fat": "Body Fat (%)"
        }

        df_display = df_metrics[display_columns].rename(columns=display_labels)

        st.dataframe(df_display, use_container_width=True)

        # Plot Weight Over Time
        st.write("---")
        st.subheader("📉 Body Weight Over Time")

        fig_weight_time = px.line(
            df_display,
            x="Date",
            y=display_labels["Weight (kg)"] if latest_system == "Metric (kg/cm)" else display_labels["Weight (lbs)"],
            title="📈 Body Weight Over Time",
            labels={
                "Date": "Date",
                display_labels["Weight (kg)"] if latest_system == "Metric (kg/cm)" else display_labels["Weight (lbs)"]: weight_label
            },
            markers=True
        )
        st.plotly_chart(fig_weight_time, use_container_width=True)

        # Calculate BMI
        st.write("---")
        st.subheader("🧮 BMI Over Time")

        if latest_system == "Imperial (lbs/in)":
            df_metrics["BMI"] = df_metrics.apply(
                lambda row: (row["user_weight"] * 703) /
                (row["height"] ** 2) if row["height"] else None,
                axis=1
            )
        else:
            df_metrics["BMI"] = df_metrics.apply(
                lambda row: (row["Weight (kg)"]) / ((row["Height (cm)"] / 100)
                                                    ** 2) if row["Height (cm)"] else None,
                axis=1
            )

        df_display["BMI"] = df_metrics["BMI"]

        fig_bmi = px.line(
            df_display,
            x="Date",
            y="BMI",
            title="📈 BMI Over Time",
            labels={"Date": "Date", "BMI": "BMI"},
            markers=True
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

        # BMI Categories
        df_display["BMI Category"] = df_display["BMI"].apply(bmi_category)

        st.write("---")
        st.subheader("📊 BMI Categories Over Time")

        if 'BMI Category' in df_display.columns and df_display['BMI Category'].notnull().any():
            fig_bmi_cat = px.scatter(
                df_display,
                x="Date",
                y="BMI",
                color="BMI Category",
                title="📈 BMI Categories Over Time",
                labels={"Date": "Date", "BMI": "BMI"},
                hover_data=["BMI Category"],
                size_max=15
            )
            st.plotly_chart(fig_bmi_cat, use_container_width=True)
        else:
            st.info("No BMI category data available to display.")

        # Body Fat Percentage Over Time
        st.write("---")
        st.subheader("📉 Body Fat Percentage Over Time")

        if 'Body Fat (%)' in df_display.columns and df_display['Body Fat (%)'].notnull().any():
            fig_body_fat = px.line(
                df_display,
                x="Date",
                y="Body Fat (%)",
                title="📈 Body Fat Percentage Over Time",
                labels={"Date": "Date", "Body Fat (%)": "Body Fat (%)"},
                markers=True
            )
            st.plotly_chart(fig_body_fat, use_container_width=True)
        else:
            st.info("No body fat data available to display.")

        # Correlation Heatmap
        st.write("---")
        st.subheader("🔍 Correlation Between Body Metrics")

        # Select numeric columns for correlation
        numeric_cols = ["Weight (kg)" if latest_system == "Metric (kg/cm)" else "Weight (lbs)",
                        "Height (cm)" if latest_system == "Metric (kg/cm)" else "Height (in)",
                        "BMI",
                        "Body Fat (%)",
                        "Chest (cm)" if latest_system == "Metric (kg/cm)" else "Chest (in)",
                        "Waist (cm)" if latest_system == "Metric (kg/cm)" else "Waist (in)",
                        "Hips (cm)" if latest_system == "Metric (kg/cm)" else "Hips (in)",
                        "Arms (cm)" if latest_system == "Metric (kg/cm)" else "Arms (in)",
                        "Lean Body Mass",
                        "Fat Mass"]

        # Filter out columns with all NaN
        numeric_cols = [
            col for col in numeric_cols if col in df_display.columns and df_display[col].notnull().any()]

        if len(numeric_cols) > 1:
            corr_matrix = df_display[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="📈 Correlation Heatmap",
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric data to display correlation heatmap.")

        # Scatter Plot: Weight vs. Body Fat
        st.write("---")
        st.subheader("🔍 Weight vs. Body Fat Percentage")

        if 'Body Fat (%)' in df_display.columns and df_display['Body Fat (%)'].notnull().any():
            fig_scatter_wt_bf = px.scatter(
                df_display,
                x=display_labels["Weight (kg)"] if latest_system == "Metric (kg/cm)" else display_labels["Weight (lbs)"],
                y="Body Fat (%)",
                color="BMI Category",
                title="⚖️ Weight vs. Body Fat Percentage",
                labels={
                    "Body Fat (%)": "Body Fat (%)",
                    display_labels["Weight (kg)"] if latest_system == "Metric (kg/cm)" else display_labels["Weight (lbs)"]: weight_label
                },
                trendline="ols",
                hover_data=["BMI Category"]
            )
            st.plotly_chart(fig_scatter_wt_bf, use_container_width=True)
        else:
            st.info("No body fat data available to display scatter plot.")

        # Lean Body Mass (LBM) and Fat Mass Over Time
        st.write("---")
        st.subheader("💪 Lean Body Mass and Fat Mass Over Time")

        if 'Body Fat (%)' in df_display.columns and df_display['Body Fat (%)'].notnull().any():
            df_display["Fat Mass"] = (df_display["Body Fat (%)"] / 100) * (
                df_display["Weight (kg)"] if latest_system == "Metric (kg/cm)" else df_display["Weight (lbs)"]
            )
            df_display["Lean Body Mass"] = (
                df_display["Weight (kg)"] if latest_system == "Metric (kg/cm)" else df_display["Weight (lbs)"]
            ) - df_display["Fat Mass"]

            fig_lbm_fm = px.line(
                df_display,
                x="Date",
                y=["Lean Body Mass", "Fat Mass"],
                title="💪 Lean Body Mass and Fat Mass Over Time",
                labels={"value": "Mass", "Date": "Date", "variable": ""},
                markers=True
            )
            st.plotly_chart(fig_lbm_fm, use_container_width=True)
        else:
            st.info(
                "No body fat data available to calculate Lean Body Mass and Fat Mass.")

        # Polar Chart for Body Measurements
        st.write("---")
        st.subheader("🌐 Body Measurements Polar Chart")

        # Select the latest entry with all measurements
        latest_entry = df_display.iloc[-1]

        measurements = {}
        measurement_labels = ["Chest", "Waist", "Hips", "Arms"]
        for measure in measurement_labels:
            if measure in latest_entry:
                val = latest_entry[measure]
                if pd.notnull(val):
                    measurements[measure] = val

        if measurements:
            categories = list(measurements.keys())
            values = list(measurements.values())

            # Close the loop for the polar chart
            categories += [categories[0]]
            values += [values[0]]

            fig_polar = go.Figure(
                data=[
                    go.Scatterpolar(r=values, theta=categories, fill='toself')
                ],
                layout=go.Layout(
                    title=go.layout.Title(text="🔴 Current Body Measurements"),
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(values) * 1.2]
                        )
                    ),
                    showlegend=False
                )
            )
            st.plotly_chart(fig_polar, use_container_width=True)
        else:
            st.info("No body measurements available for polar chart.")

        # Interactive Metrics Selection
        st.write("---")
        st.subheader("🔄 Interactive Metrics Selection")

        available_metrics = ["Weight", "BMI", "Body Fat (%)", "Lean Body Mass", "Fat Mass",
                             "Chest", "Waist", "Hips", "Arms"]

        selected_metrics = st.multiselect(
            "Select metrics to display",
            options=available_metrics,
            default=["Weight", "BMI"]
        )

        if selected_metrics:
            fig_interactive = go.Figure()
            for metric in selected_metrics:
                if metric == "Weight":
                    y_data = df_display["Weight (kg)" if latest_system ==
                                        "Metric (kg/cm)" else "Weight (lbs)"]
                    y_label = weight_label
                elif metric == "BMI":
                    y_data = df_display["BMI"]
                    y_label = "BMI"
                elif metric == "Body Fat (%)":
                    y_data = df_display["Body Fat (%)"]
                    y_label = "Body Fat (%)"
                elif metric == "Lean Body Mass":
                    y_data = df_display["Lean Body Mass"] if "Lean Body Mass" in df_display.columns else None
                    y_label = "Lean Body Mass"
                elif metric == "Fat Mass":
                    y_data = df_display["Fat Mass"] if "Fat Mass" in df_display.columns else None
                    y_label = "Fat Mass"
                else:
                    # Body Measurements
                    measure_col = f"{metric} (cm)" if latest_system == "Metric (kg/cm)" else f"{metric} (in)"
                    y_data = df_display[measure_col] if measure_col in df_display.columns else None
                    y_label = metric

                if y_data is not None and y_data.notnull().any():
                    fig_interactive.add_trace(go.Scatter(
                        x=df_display["Date"],
                        y=y_data,
                        mode='lines+markers',
                        name=metric
                    ))

            fig_interactive.update_layout(
                title="📈 Selected Body Metrics Over Time",
                xaxis_title="Date",
                yaxis_title="Metrics",
                hovermode="x unified"
            )
            st.plotly_chart(fig_interactive, use_container_width=True)
        else:
            st.info("Select at least one metric to display.")

        # Export Data as CSV
        st.write("---")
        st.subheader("📥 Export Data")

        csv = df_display.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📄 Download Body Metrics as CSV",
            data=csv,
            file_name='body_metrics.csv',
            mime='text/csv',
        )

        # Export Visualizations as Images
        st.write("---")
        st.subheader("📤 Export Visualizations")

        visualizations = {
            "Weight Over Time": fig_weight_time,
            "BMI Over Time": fig_bmi,
            "Body Fat Percentage Over Time": fig_body_fat if 'Body Fat (%)' in df_display.columns else None,
            "BMI Categories Over Time": fig_bmi_cat if 'BMI Category' in df_display.columns else None,
            "Correlation Heatmap": fig_corr if 'corr_matrix' in locals() else None,
            "Weight vs. Body Fat Percentage": fig_scatter_wt_bf if 'Body Fat (%)' in df_display.columns else None,
            "Lean Body Mass and Fat Mass Over Time": fig_lbm_fm if 'Lean Body Mass' in df_display.columns else None,
            "Polar Chart": fig_polar if 'fig_polar' in locals() else None,
            "Interactive Metrics": fig_interactive if 'fig_interactive' in locals() else None
        }

        selected_vis = st.selectbox(
            "Select a visualization to download:",
            options=list(visualizations.keys()),
            index=0
        )

        if visualizations[selected_vis] is not None:
            img_bytes = pio.to_image(
                visualizations[selected_vis], format='png')
            st.download_button(
                label=f"📥 Download {selected_vis} as PNG",
                data=img_bytes,
                file_name=f"{selected_vis.replace(' ', '_').lower()}.png",
                mime='image/png',
            )
        else:
            st.info("Selected visualization is not available for download.")


def bmi_category(bmi):
    if bmi is None:
        return "Unknown"
    elif bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def estimate_body_fat(weight, height, age, gender):
    """
    Estimate body fat percentage using the Deurenberg formula.
    Note: This formula requires age and gender for more accurate estimation.
    """
    if height is None or age is None or gender is None:
        return None  # Cannot estimate without these parameters

    # Calculate BMI
    bmi = calculate_bmi(weight, height, gender)

    if bmi is None:
        return None

    # Deurenberg formula: Body Fat (%) = 1.20 × BMI + 0.23 × Age - 10.8 × Gender - 5.4
    # Gender: 1 for male, 0 for female, 0.5 for others
    if gender.lower() == "male":
        gender_numeric = 1
    elif gender.lower() == "female":
        gender_numeric = 0
    else:
        gender_numeric = 0.5  # For 'Other' or unspecified

    body_fat = 1.20 * bmi + 0.23 * age - 10.8 * gender_numeric - 5.4
    return body_fat


def calculate_bmi(weight, height, gender):
    """
    Calculate BMI based on weight and height.
    BMI = weight (kg) / (height (m))^2 for Metric
    BMI = 703 * weight (lbs) / (height (in))^2 for Imperial
    """
    if gender is None:
        return None  # Gender is needed for certain BMI adjustmentss

    # Determine measurement system based on units
    if weight > 500:  # Assuming metric if weight > 500 lbs is unlikely
        # Metric system
        bmi = weight / ((height / 100) ** 2)  # height converted to meters
    else:
        # Imperial system
        bmi = (703 * weight) / (height ** 2)

    return bmi


if __name__ == "__main__":
    main()
