import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from db_utils import create_connection
import datetime

# Page configuration
st.set_page_config(
    page_title="Workout Animations",
    page_icon="🎬",
    layout="wide"
)

# Page title and description
st.title("🎬 Workout Animations")
st.markdown("""
This page features interactive animations that help visualize your fitness journey over time.
Each animation provides unique insights into different aspects of your workout progress and body metrics.
""")

# Load workout data


@st.cache_data(ttl=300)
def load_workouts_data():
    """Load workout data from the database."""
    with create_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM workouts", conn)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["volume"] = df["weight"] * df["sets"] * df["reps"]
    return df

# Load body metrics data


@st.cache_data(ttl=300)
def load_body_metrics_data():
    """Load body metrics data from the database."""
    with create_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM body_metrics ORDER BY entry_date", conn)
        if not df.empty:
            df["entry_date"] = pd.to_datetime(
                df["entry_date"], errors="coerce")
            # Calculate BMI
            df["bmi"] = (df["user_weight"] * 703) / \
                (df["height"] * df["height"])
    return df

# Load exercises data


@st.cache_data(ttl=300)
def load_exercises_data():
    """Load exercises data from the database."""
    with create_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM exercises", conn)
    return df


# Load data
workouts_df = load_workouts_data()
body_metrics_df = load_body_metrics_data()
exercises_df = load_exercises_data()

# Check if data is available
if workouts_df.empty:
    st.warning("No workout data available. Please add some workouts first.")
else:
    # Create tabs for different animations
    tabs = st.tabs([
        "1. Workout Volume Evolution",
        "2. Strength Progression",
        "3. Body Composition Journey",
        "4. Workout Type Distribution",
        "5. Exercise Frequency Heatmap"
    ])

    # 1. Workout Volume Evolution Animation
    with tabs[0]:
        st.header("📈 Workout Volume Evolution")
        st.markdown("""
        This animation shows how your workout volume has evolved over time for different muscle groups.
        The bubble size represents the total volume (weight × sets × reps) for each muscle group on a given day.
        """)

        # Prepare data
        volume_data = workouts_df.copy()
        # Fill missing muscle types
        volume_data["muscle_type"] = volume_data["muscle_type"].fillna("Other")
        # Group by date and muscle type
        daily_volume = volume_data.groupby(["date", "muscle_type"])[
            "volume"].sum().reset_index()

        # Get unique dates for animation
        unique_dates = sorted(daily_volume["date"].unique())

        # Create animation
        fig = px.scatter(
            daily_volume,
            x="date",
            y="muscle_type",
            size="volume",
            color="muscle_type",
            animation_frame="date",
            animation_group="muscle_type",
            range_x=[min(unique_dates), max(unique_dates)],
            size_max=50,
            title="Muscle Group Volume Over Time",
            labels={"date": "Date", "muscle_type": "Muscle Group",
                    "volume": "Total Volume"},
            height=600
        )

        # Update layout
        fig.update_layout(
            xaxis=dict(title="Date", gridcolor="rgba(128, 128, 128, 0.1)"),
            yaxis=dict(title="Muscle Group",
                       gridcolor="rgba(128, 128, 128, 0.1)"),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=12),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [date.strftime("%Y-%m-%d")],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": date.strftime("%Y-%m-%d"),
                        "method": "animate"
                    }
                    for date in unique_dates
                ]
            }]
        )

        # Display the animation
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.subheader("💡 Insights")
        st.markdown("""
        - Track which muscle groups receive the most volume over time
        - Identify patterns in your training focus
        - Spot gaps in your training where certain muscle groups might be neglected
        - See how your training volume distribution has evolved
        """)

    # 2. Strength Progression Animation
    with tabs[1]:
        st.header("💪 Strength Progression")
        st.markdown("""
        This animation shows your strength progression for different exercises over time.
        The line chart displays the maximum weight lifted for each exercise, allowing you to visualize your strength gains.
        """)

        # Prepare data
        strength_data = workouts_df.copy()
        # Group by date and workout to find max weight for each exercise on each day
        daily_max_weight = strength_data.groupby(["date", "workout"])[
            "weight"].max().reset_index()

        # Get top exercises by frequency
        top_exercises = strength_data["workout"].value_counts().head(
            8).index.tolist()
        filtered_data = daily_max_weight[daily_max_weight["workout"].isin(
            top_exercises)]

        # Create a complete date range
        all_dates = pd.date_range(
            start=filtered_data["date"].min(), end=filtered_data["date"].max())

        # Create animation frames
        frames = []
        for i, date in enumerate(all_dates):
            # Get data up to this date
            mask = filtered_data["date"] <= date
            data_to_date = filtered_data[mask]

            if not data_to_date.empty:
                # Get the latest max weight for each exercise up to this date
                latest_weights = data_to_date.sort_values(
                    "date").groupby("workout").last().reset_index()

                # Create frame
                frame = go.Frame(
                    data=[
                        go.Bar(
                            x=latest_weights["workout"],
                            y=latest_weights["weight"],
                            marker_color=px.colors.qualitative.Plotly,
                            text=latest_weights["weight"].round(1),
                            textposition="auto"
                        )
                    ],
                    name=date.strftime("%Y-%m-%d")
                )
                frames.append(frame)

        # Create initial figure
        initial_data = filtered_data[filtered_data["date"]
                                     == filtered_data["date"].min()]
        if not initial_data.empty:
            initial_data = initial_data.groupby("workout").last().reset_index()
        else:
            # If no data for the first date, use the earliest available data
            initial_data = filtered_data.sort_values(
                "date").groupby("workout").first().reset_index()

        fig = go.Figure(
            data=[
                go.Bar(
                    x=initial_data["workout"],
                    y=initial_data["weight"],
                    marker_color=px.colors.qualitative.Plotly,
                    text=initial_data["weight"].round(1),
                    textposition="auto"
                )
            ],
            frames=frames
        )

        # Update layout
        fig.update_layout(
            title="Maximum Weight Progression by Exercise",
            xaxis=dict(title="Exercise", gridcolor="rgba(128, 128, 128, 0.1)"),
            yaxis=dict(title="Max Weight (lbs)",
                       gridcolor="rgba(128, 128, 128, 0.1)"),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=12),
            height=600,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": frame.name,
                        "method": "animate"
                    }
                    for frame in frames
                ]
            }]
        )

        # Display the animation
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.subheader("💡 Insights")
        st.markdown("""
        - Track your strength progression for key exercises
        - Identify which exercises are showing the most improvement
        - Spot plateaus where progress has stalled
        - Compare strength gains across different exercises
        """)

    # 3. Body Composition Journey Animation
    with tabs[2]:
        st.header("⚖️ Body Composition Journey")
        st.markdown("""
        This animation visualizes your body composition changes over time.
        The chart shows the relationship between weight, body fat percentage, and BMI.
        """)

        if body_metrics_df.empty:
            st.warning(
                "No body metrics data available. Please add body measurements first.")
        else:
            # Prepare data
            body_data = body_metrics_df.copy()

            # Create animation
            if "body_fat" in body_data.columns and "bmi" in body_data.columns:
                # Filter out rows with missing data
                body_data = body_data.dropna(
                    subset=["user_weight", "body_fat", "bmi"])

                if not body_data.empty:
                    fig = px.scatter(
                        body_data,
                        x="user_weight",
                        y="body_fat",
                        size="bmi",
                        color="bmi",
                        animation_frame="entry_date",
                        range_x=[body_data["user_weight"].min(
                        ) - 5, body_data["user_weight"].max() + 5],
                        range_y=[body_data["body_fat"].min(
                        ) - 2, body_data["body_fat"].max() + 2],
                        color_continuous_scale="RdYlGn_r",  # Red for higher BMI, green for lower
                        size_max=30,
                        title="Body Composition Changes Over Time",
                        labels={
                            "user_weight": "Weight (lbs)",
                            "body_fat": "Body Fat %",
                            "bmi": "BMI",
                            "entry_date": "Date"
                        },
                        height=600
                    )

                    # Add BMI category regions
                    bmi_categories = [
                        {"name": "Underweight", "min_bmi": 0, "max_bmi": 18.5,
                            "color": "rgba(52, 152, 219, 0.2)"},
                        {"name": "Normal", "min_bmi": 18.5, "max_bmi": 25,
                            "color": "rgba(46, 204, 113, 0.2)"},
                        {"name": "Overweight", "min_bmi": 25, "max_bmi": 30,
                            "color": "rgba(243, 156, 18, 0.2)"},
                        {"name": "Obese", "min_bmi": 30, "max_bmi": 100,
                            "color": "rgba(231, 76, 60, 0.2)"}
                    ]

                    # Calculate average height to estimate weight ranges for BMI categories
                    avg_height = body_data["height"].mean()

                    for category in bmi_categories:
                        min_weight = (category["min_bmi"]
                                      * (avg_height ** 2)) / 703
                        max_weight = (category["max_bmi"]
                                      * (avg_height ** 2)) / 703

                        fig.add_shape(
                            type="rect",
                            x0=min_weight,
                            x1=max_weight,
                            y0=0,
                            y1=100,
                            fillcolor=category["color"],
                            line=dict(width=0),
                            layer="below"
                        )

                        # Add text annotation for BMI category
                        fig.add_annotation(
                            x=(min_weight + max_weight) / 2,
                            y=5,
                            text=category["name"],
                            showarrow=False,
                            font=dict(size=10)
                        )

                    # Update layout
                    fig.update_layout(
                        xaxis=dict(title="Weight (lbs)",
                                   gridcolor="rgba(128, 128, 128, 0.1)"),
                        yaxis=dict(title="Body Fat %",
                                   gridcolor="rgba(128, 128, 128, 0.1)"),
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                        font=dict(size=12),
                        updatemenus=[{
                            "buttons": [
                                {
                                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                                    "label": "Play",
                                    "method": "animate"
                                },
                                {
                                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                                    "label": "Pause",
                                    "method": "animate"
                                }
                            ],
                            "direction": "left",
                            "pad": {"r": 10, "t": 10},
                            "showactive": False,
                            "type": "buttons",
                            "x": 0.1,
                            "y": 0,
                            "xanchor": "right",
                            "yanchor": "top"
                        }]
                    )

                    # Display the animation
                    st.plotly_chart(fig, use_container_width=True)

                    # Insights
                    st.subheader("💡 Insights")
                    st.markdown("""
                    - Visualize the relationship between weight and body fat percentage
                    - Track your progress through different BMI categories
                    - See how your body composition changes over time
                    - Identify trends in your weight and body fat fluctuations
                    """)
                else:
                    st.warning(
                        "Insufficient body metrics data with body fat and BMI values.")
            else:
                st.warning(
                    "Body fat or BMI data not available in your body metrics.")

    # 4. Workout Type Distribution Animation
    with tabs[3]:
        st.header("🔄 Workout Type Distribution")
        st.markdown("""
        This animation shows how your workout type distribution has changed over time.
        The pie chart displays the proportion of different workout types (Push, Pull, Leg, etc.) for each month.
        """)

        # Prepare data
        workout_type_data = workouts_df.copy()
        # Fill missing workout types
        workout_type_data["workout_type"] = workout_type_data["workout_type"].fillna(
            "Other")

        # Add month column
        workout_type_data["month"] = workout_type_data["date"].dt.to_period(
            "M")

        # Group by month and workout type
        monthly_distribution = workout_type_data.groupby(
            ["month", "workout_type"]).size().reset_index(name="count")

        # Get unique months
        unique_months = sorted(monthly_distribution["month"].unique())

        # Create frames for animation
        frames = []
        for month in unique_months:
            month_data = monthly_distribution[monthly_distribution["month"] == month]

            frame = go.Frame(
                data=[
                    go.Pie(
                        labels=month_data["workout_type"],
                        values=month_data["count"],
                        hole=0.3,
                        textinfo="label+percent",
                        marker=dict(colors=px.colors.qualitative.Bold)
                    )
                ],
                name=str(month)
            )
            frames.append(frame)

        # Create initial figure with first month's data
        initial_month = unique_months[0]
        initial_data = monthly_distribution[monthly_distribution["month"]
                                            == initial_month]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=initial_data["workout_type"],
                    values=initial_data["count"],
                    hole=0.3,
                    textinfo="label+percent",
                    marker=dict(colors=px.colors.qualitative.Bold)
                )
            ],
            frames=frames
        )

        # Update layout
        fig.update_layout(
            title="Monthly Workout Type Distribution",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=12),
            height=600,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Month: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(month)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": str(month),
                        "method": "animate"
                    }
                    for month in unique_months
                ]
            }]
        )

        # Display the animation
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.subheader("💡 Insights")
        st.markdown("""
        - Track how your workout focus changes month to month
        - Identify imbalances in your training split
        - See seasonal patterns in your workout preferences
        - Ensure you're maintaining a balanced approach to training
        """)

    # 5. Exercise Frequency Heatmap Animation
    with tabs[4]:
        st.header("🔥 Exercise Frequency Heatmap")
        st.markdown("""
        This animation shows a heatmap of your exercise frequency over time.
        The heatmap displays which days of the week you're most active and how your workout schedule has evolved.
        """)

        # Prepare data
        frequency_data = workouts_df.copy()

        # Extract day of week and week number
        frequency_data["day_of_week"] = frequency_data["date"].dt.day_name()
        frequency_data["week"] = frequency_data["date"].dt.isocalendar().week
        frequency_data["month"] = frequency_data["date"].dt.month
        frequency_data["year_month"] = frequency_data["date"].dt.strftime(
            "%Y-%m")

        # Order days of week
        days_order = ["Monday", "Tuesday", "Wednesday",
                      "Thursday", "Friday", "Saturday", "Sunday"]

        # Group by year-month, day of week and count workouts
        monthly_heatmap = frequency_data.groupby(
            ["year_month", "day_of_week"]).size().reset_index(name="count")

        # Get unique year-months
        unique_year_months = sorted(monthly_heatmap["year_month"].unique())

        # Create frames for animation
        frames = []
        for year_month in unique_year_months:
            month_data = monthly_heatmap[monthly_heatmap["year_month"] == year_month]

            # Create a complete dataset with all days of week
            complete_data = pd.DataFrame({"day_of_week": days_order})
            complete_data = complete_data.merge(
                month_data, on="day_of_week", how="left")
            complete_data["year_month"] = year_month
            complete_data["count"] = complete_data["count"].fillna(0)

            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=complete_data["count"],
                        x=complete_data["day_of_week"],
                        y=[year_month] * len(complete_data),
                        colorscale="Viridis",
                        showscale=True,
                        text=complete_data["count"].astype(
                            int).astype(str) + " workouts",
                        hoverinfo="text"
                    )
                ],
                name=year_month
            )
            frames.append(frame)

        # Create initial figure with first month's data
        initial_year_month = unique_year_months[0]
        initial_data = monthly_heatmap[monthly_heatmap["year_month"]
                                       == initial_year_month]

        # Create a complete dataset with all days of week for initial data
        complete_initial_data = pd.DataFrame({"day_of_week": days_order})
        complete_initial_data = complete_initial_data.merge(
            initial_data, on="day_of_week", how="left")
        complete_initial_data["year_month"] = initial_year_month
        complete_initial_data["count"] = complete_initial_data["count"].fillna(
            0)

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=complete_initial_data["count"],
                    x=complete_initial_data["day_of_week"],
                    y=[initial_year_month] * len(complete_initial_data),
                    colorscale="Viridis",
                    showscale=True,
                    text=complete_initial_data["count"].astype(
                        int).astype(str) + " workouts",
                    hoverinfo="text"
                )
            ],
            frames=frames
        )

        # Update layout
        fig.update_layout(
            title="Workout Frequency by Day of Week (Monthly)",
            xaxis=dict(title="Day of Week", categoryorder="array",
                       categoryarray=days_order),
            yaxis=dict(title="Month"),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=12),
            height=600,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Month: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [year_month],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": year_month,
                        "method": "animate"
                    }
                    for year_month in unique_year_months
                ]
            }]
        )

        # Display the animation
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.subheader("💡 Insights")
        st.markdown("""
        - Identify your most consistent workout days
        - Spot patterns in your weekly workout schedule
        - Track how your workout frequency changes month to month
        - Find opportunities to improve consistency in your training schedule
        """)

# Footer
st.markdown("---")
st.markdown(
    "💪 **Workout Tracker** - Visualize your fitness journey with interactive animations")
