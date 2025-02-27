import streamlit as st
from db_utils import init_db, create_connection
from auth import check_password
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
import time
import os
import sqlite3

# Global styling and configuration
st.set_page_config(
    page_title="Workout & Body Metrics Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown("""
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #1f77b4;
    }
    
    /* Metric container styling */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #dee2e6;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    /* Success message styling */
    div[data-testid="stSuccess"] {
        border-radius: 5px;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
    
    /* Chart styling */
    .js-plotly-plot {
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Form styling */
    div[data-testid="stForm"] {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }

    /* Dashboard card styling */
    .dashboard-card {
        background-color: rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        border-left: 4px solid #1f77b4;
    }

    /* Metric highlight */
    .metric-highlight {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
    }

    /* Columns with margin */
    .column-with-margin {
        margin: 0 10px;
    }
    
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .dashboard-card {
            background-color: rgba(49, 51, 63, 0.2);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        div[data-testid="metric-container"] {
            background-color: rgba(49, 51, 63, 0.2);
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        .js-plotly-plot {
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
    }
    </style>
""", unsafe_allow_html=True)


def calculate_bmi(weight_lbs, height_in):
    """Calculate BMI from weight in pounds and height in inches."""
    if weight_lbs <= 0 or height_in <= 0:
        return 0
    return (weight_lbs * 703) / (height_in * height_in)


def bmi_category(bmi_value):
    """Return the BMI category and color based on the BMI value."""
    if bmi_value < 18.5:
        return "Underweight", "#3498db"
    elif bmi_value < 25:
        return "Normal", "#2ecc71"
    elif bmi_value < 30:
        return "Overweight", "#f39c12"
    else:
        return "Obese", "#e74c3c"


def show_metric_card(title, value, trend=None, suffix="", precision=1):
    """Display a metric with optional trend indicator."""
    if value is None:
        st.markdown(f"**{title}**\nNo data")
        return

    value_str = f"{value:.{precision}f}{suffix}"
    if trend is not None:
        if trend > 0:
            delta_color = "normal" if "weight" not in title.lower() else "inverse"
            st.metric(title, value_str,
                      f"+{trend:.{precision}f}{suffix}", delta_color=delta_color)
        elif trend < 0:
            delta_color = "inverse" if "weight" not in title.lower() else "normal"
            st.metric(title, value_str,
                      f"{trend:.{precision}f}{suffix}", delta_color=delta_color)
        else:
            st.metric(title, value_str, "No change")
    else:
        st.metric(title, value_str)


@st.cache_data
def load_workouts():
    """Load workouts data from the database."""
    try:
        db_path = "workouts.db"
        last_update = os.path.getmtime(db_path)

        # Check if we need to add the modification counter to the session state
        if "db_modification_counter" not in st.session_state:
            st.session_state.db_modification_counter = 0

    except Exception as e:
        st.error(f"❌ Unable to access the database file: {e}")
        last_update = None

    @st.cache_data
    def get_workouts(last_update, modification_counter):
        try:
            with create_connection() as conn:
                df = pd.read_sql_query("SELECT * FROM workouts", conn)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["volume"] = df["weight"] * df["sets"] * df["reps"]
            return df
        except sqlite3.OperationalError as e:
            # Table doesn't exist yet or other SQL error
            st.warning(
                "Workout data table not found. It will be created when you add your first workout.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading workout data: {e}")
            return pd.DataFrame()

    if last_update is not None:
        # Pass both the file modification time and the counter to invalidate cache
        return get_workouts(last_update, st.session_state.db_modification_counter)
    else:
        return pd.DataFrame()


def load_body_metrics_data():
    """Load body metrics data from the database."""
    try:
        with create_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM body_metrics ORDER BY entry_date", conn)
            if not df.empty:
                df["entry_date"] = pd.to_datetime(
                    df["entry_date"], errors="coerce")
        return df
    except sqlite3.OperationalError as e:
        # Table doesn't exist yet or other SQL error
        st.warning(
            "Body metrics table not found. It will be created when you add your first body metrics entry.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading body metrics data: {e}")
        return pd.DataFrame()


def load_progress_photos_count():
    """Count progress photos by type."""
    try:
        with create_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT photo_type, COUNT(*) FROM progress_photos GROUP BY photo_type")
            return dict(cursor.fetchall())
    except sqlite3.OperationalError as e:
        # Table doesn't exist yet or other SQL error
        return {}
    except Exception as e:
        st.error(f"Error loading progress photos: {e}")
        return {}


def load_goals_data():
    """Load user fitness goals."""
    try:
        with create_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM body_metrics_goals WHERE achieved = 0", conn)
            return df
    except sqlite3.OperationalError as e:
        # Table doesn't exist yet or other SQL error
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading goals data: {e}")
        return pd.DataFrame()


def calculate_trend(current, previous):
    """Calculate percentage or absolute change between current and previous values."""
    if current is None or previous is None or pd.isna(current) or pd.isna(previous):
        return None

    if previous == 0:
        return None

    return ((current - previous) / previous) * 100


def unified_dashboard():
    """Create a unified dashboard with both workout and body metrics data."""

    # Add refresh controls
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.title("💪 Fitness Dashboard")
    with col2:
        auto_refresh = st.checkbox("Auto refresh", value=False)
    with col3:
        if st.button("🔄 Refresh"):
            # Increment counter to invalidate cache
            if "db_modification_counter" in st.session_state:
                st.session_state.db_modification_counter += 1
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    # Handle auto-refresh (every 30 seconds)
    if auto_refresh and "last_refresh_time" in st.session_state and time.time() - st.session_state.last_refresh_time > 30:
        if "db_modification_counter" in st.session_state:
            st.session_state.db_modification_counter += 1
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    elif "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()

    # Load all necessary data
    workouts_df = load_workouts()
    body_metrics_df = load_body_metrics_data()
    photo_counts = load_progress_photos_count()
    goals_df = load_goals_data()

    # Create tabs for different dashboard views
    tabs = st.tabs(["Overview", "Workout Stats",
                   "Body Metrics", "Goals & Progress"])

    with tabs[0]:
        st.header("📊 Fitness Overview")

        # Top row with key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Workout summary
            if not workouts_df.empty:
                total_workouts = workouts_df['date'].nunique()
                recent_workouts = workouts_df[workouts_df['date'] >= (
                    pd.Timestamp.now() - pd.Timedelta(days=30))]['date'].nunique()
                st.markdown("""
                <div class="dashboard-card">
                    <h3>🏋️ Workouts</h3>
                    <p class="metric-highlight">{}</p>
                    <p>Total Sessions</p>
                    <p>{} in last 30 days</p>
                </div>
                """.format(total_workouts, recent_workouts), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>🏋️ Workouts</h3>
                    <p>No workout data yet</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Body metrics summary
            if not body_metrics_df.empty:
                latest_weight = body_metrics_df['user_weight'].iloc[-1]
                weight_change = None
                if len(body_metrics_df) > 1:
                    prev_weight = body_metrics_df['user_weight'].iloc[-2]
                    weight_change = latest_weight - prev_weight

                weight_str = f"{latest_weight:.1f} lbs"
                change_str = f" ({weight_change:+.1f})" if weight_change is not None else ""

                st.markdown("""
                <div class="dashboard-card">
                    <h3>⚖️ Current Weight</h3>
                    <p class="metric-highlight">{}</p>
                    <p>{}</p>
                </div>
                """.format(weight_str, change_str), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>⚖️ Current Weight</h3>
                    <p>No weight data yet</p>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            # Body composition
            if not body_metrics_df.empty and 'body_fat' in body_metrics_df.columns and not pd.isna(body_metrics_df['body_fat'].iloc[-1]):
                latest_bf = body_metrics_df['body_fat'].iloc[-1]
                bf_change = None
                if len(body_metrics_df) > 1 and not pd.isna(body_metrics_df['body_fat'].iloc[-2]):
                    prev_bf = body_metrics_df['body_fat'].iloc[-2]
                    bf_change = latest_bf - prev_bf

                bf_str = f"{latest_bf:.1f}%"
                change_str = f" ({bf_change:+.1f}%)" if bf_change is not None else ""

                st.markdown("""
                <div class="dashboard-card">
                    <h3>📊 Body Fat</h3>
                    <p class="metric-highlight">{}</p>
                    <p>{}</p>
                </div>
                """.format(bf_str, change_str), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>📊 Body Fat</h3>
                    <p>No body fat data yet</p>
                </div>
                """, unsafe_allow_html=True)

        with col4:
            # Photos & tracking
            photo_total = sum(photo_counts.values()) if photo_counts else 0

            st.markdown("""
            <div class="dashboard-card">
                <h3>📸 Progress Photos</h3>
                <p class="metric-highlight">{}</p>
                <p>Total photos</p>
            </div>
            """.format(photo_total), unsafe_allow_html=True)

        # Second row - charts and trends
        st.subheader("📈 Recent Trends")

        col1, col2 = st.columns(2)

        with col1:
            # Weight trend chart
            if not body_metrics_df.empty:
                recent_metrics = body_metrics_df.tail(10)  # Last 10 entries
                fig = px.line(recent_metrics, x='entry_date', y='user_weight',
                              title="Recent Weight Trend",
                              labels={'entry_date': 'Date',
                                      'user_weight': 'Weight (lbs)'},
                              markers=True)
                fig.update_layout(height=300)
                # Add animation and improved styling
                fig.update_traces(
                    mode='lines+markers',
                    marker=dict(size=8, opacity=0.8),
                    line=dict(width=3),
                )
                fig.update_layout(
                    transition_duration=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No body metrics data available for trend visualization")

        with col2:
            # Workout volume chart
            if not workouts_df.empty:
                # Group by date and sum volume
                recent_volume = workouts_df.groupby(
                    'date')['volume'].sum().reset_index()
                recent_volume = recent_volume.sort_values(
                    'date').tail(10)  # Last 10 days with workouts

                fig = px.line(recent_volume, x='date', y='volume',
                              title="Recent Workout Volume",
                              labels={'date': 'Date',
                                      'volume': 'Volume (lbs)'},
                              markers=True)
                fig.update_layout(height=300)
                # Add animation and improved styling
                fig.update_traces(
                    mode='lines+markers',
                    marker=dict(size=8, symbol='circle', line=dict(
                        width=2, color='DarkSlateGrey')),
                    line=dict(width=3)
                )
                fig.update_layout(
                    transition_duration=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No workout data available for trend visualization")

        # Add Volume by Workout Type graph
        if not workouts_df.empty:
            st.subheader("🏋️ Volume by Workout Type")

            # Group by workout_type and date to create time series data
            # First ensure we have date data to work with
            if 'date' in workouts_df.columns:
                # Create a date field with a consistent format (month level)
                workouts_df['month'] = workouts_df['date'].dt.strftime('%Y-%m')

                # Group by month and workout type to get volume over time
                workout_type_time_series = workouts_df.groupby(['month', 'workout_type'])[
                    'volume'].sum().reset_index()

                if not workout_type_time_series.empty:
                    # Create animated line chart with nice styling for volume over time by type
                    fig = px.line(
                        workout_type_time_series,
                        x='month',
                        y='volume',
                        color='workout_type',
                        labels={
                            'month': 'Month', 'volume': 'Total Volume (lbs)', 'workout_type': 'Type'},
                        title='Volume by Workout Type Over Time',
                        markers=True,
                        color_discrete_map={
                            'Push': '#1f77b4',
                            'Pull': '#ff7f0e',
                            'Leg': '#2ca02c',
                            'Full-Body': '#d62728',
                            'Core': '#9467bd'
                        }
                    )

                    # Enhance the styling
                    fig.update_traces(
                        mode='lines+markers',
                        marker=dict(size=8, opacity=0.8),
                        line=dict(width=3),
                    )

                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=40, r=40, t=60, b=40),
                        title_font=dict(size=20),
                        legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1),
                        transition_duration=800,
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Add trend analysis (keep the pie chart for distribution)
                    st.markdown("### Workout Type Analysis")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Get overall volume by type for pie chart
                        workout_type_volume = workouts_df.groupby(
                            'workout_type')['volume'].sum().reset_index()

                        # Volume distribution pie chart
                        fig = px.pie(
                            workout_type_volume,
                            values='volume',
                            names='workout_type',
                            title='Volume Distribution',
                            hole=0.4,
                            color='workout_type',
                            color_discrete_map={
                                'Push': '#1f77b4',
                                'Pull': '#ff7f0e',
                                'Leg': '#2ca02c',
                                'Full-Body': '#d62728',
                                'Core': '#9467bd'
                            }
                        )
                        fig.update_traces(textposition='inside',
                                          textinfo='percent+label')
                        fig.update_layout(
                            height=320,
                            margin=dict(l=20, r=20, t=40, b=20),
                            transition_duration=500,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Add a stacked area chart to show composition over time
                        fig = px.area(
                            workout_type_time_series,
                            x='month',
                            y='volume',
                            color='workout_type',
                            title='Volume Composition Over Time',
                            labels={
                                'month': 'Month', 'volume': 'Volume (lbs)', 'workout_type': 'Type'},
                            color_discrete_map={
                                'Push': '#1f77b4',
                                'Pull': '#ff7f0e',
                                'Leg': '#2ca02c',
                                'Full-Body': '#d62728',
                                'Core': '#9467bd'
                            }
                        )
                        fig.update_layout(
                            height=320,
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, xanchor="right", x=1),
                            transition_duration=500,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No workout type data available for time series analysis")
            else:
                # Fall back to simple volume by type if no date data
                workout_type_volume = workouts_df.groupby(
                    'workout_type')['volume'].sum().reset_index()

                if not workout_type_volume.empty:
                    st.info(
                        "Time series data not available. Showing volume distribution by type.")
                    fig = px.pie(
                        workout_type_volume,
                        values='volume',
                        names='workout_type',
                        title='Volume Distribution by Type',
                        hole=0.4,
                        color='workout_type',
                        color_discrete_map={
                            'Push': '#1f77b4',
                            'Pull': '#ff7f0e',
                            'Leg': '#2ca02c',
                            'Full-Body': '#d62728',
                            'Core': '#9467bd'
                        }
                    )
                    fig.update_traces(textposition='inside',
                                      textinfo='percent+label')
                    fig.update_layout(
                        height=400,
                        transition_duration=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No workout type data available for analysis")

        # Third row - Goals and upcoming workouts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Active Goals")
            if not goals_df.empty:
                # Get only the active goals with closest due dates
                active_goals = goals_df.sort_values('target_date').head(5)
                for _, goal in active_goals.iterrows():
                    days_left = (pd.to_datetime(
                        goal['target_date']) - pd.Timestamp.now()).days
                    progress_text = f"{goal['metric_name']}: {goal['target_value']:.1f} ({days_left} days left)"
                    st.progress(min(max(1 - (days_left / 30), 0), 1),
                                text=progress_text)
            else:
                st.info("No active goals set")

        with col2:
            st.subheader("📆 Recent Activity")

            # Combine recent workouts and body metrics in a timeline
            timeline_data = []

            if not workouts_df.empty:
                recent_workouts = workouts_df.sort_values(
                    'date', ascending=False).head(5)
                for _, workout in recent_workouts.iterrows():
                    timeline_data.append({
                        'date': workout['date'],
                        'activity': f"🏋️ {workout['workout']} workout: {workout['weight']} lbs × {workout['sets']} sets × {workout['reps']} reps"
                    })

            if not body_metrics_df.empty:
                recent_metrics = body_metrics_df.sort_values(
                    'entry_date', ascending=False).head(5)
                for _, metric in recent_metrics.iterrows():
                    timeline_data.append({
                        'date': metric['entry_date'],
                        'activity': f"⚖️ Weight recorded: {metric['user_weight']:.1f} lbs"
                    })

            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df = timeline_df.sort_values('date', ascending=False)
                for _, item in timeline_df.head(5).iterrows():
                    st.markdown(
                        f"**{item['date'].strftime('%Y-%m-%d')}**: {item['activity']}")
            else:
                st.info("No recent activity")

    with tabs[1]:
        st.header("🏋️ Workout Statistics")
        if not workouts_df.empty:
            # Key workout metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                total_volume = workouts_df['volume'].sum()
                total_sessions = workouts_df['date'].nunique()
                avg_volume = total_volume / total_sessions if total_sessions > 0 else 0

                st.metric("Total Sessions", f"{total_sessions}")
                st.metric("Total Volume", f"{total_volume:,.0f} lbs")
                st.metric("Avg Volume/Session", f"{avg_volume:,.0f} lbs")

            with col2:
                # Most frequent workouts
                top_workouts = workouts_df['workout'].value_counts().head(5)
                st.subheader("Most Frequent Exercises")
                fig = px.bar(x=top_workouts.index, y=top_workouts.values,
                             labels={'x': 'Exercise', 'y': 'Count'},
                             title="Top 5 Exercises by Frequency")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                # Max weights by exercise
                max_weights = workouts_df.groupby(
                    'workout')['weight'].max().sort_values(ascending=False).head(5)
                st.subheader("Personal Records")
                fig = px.bar(x=max_weights.index, y=max_weights.values,
                             labels={'x': 'Exercise', 'y': 'Weight (lbs)'},
                             title="Top 5 Personal Records")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Volume progression over time
            st.subheader("Volume Progression")
            monthly_volume = workouts_df.groupby(pd.Grouper(key='date', freq='M'))[
                'volume'].sum().reset_index()
            fig = px.line(monthly_volume, x='date', y='volume',
                          labels={'date': 'Month',
                                  'volume': 'Monthly Volume (lbs)'},
                          title="Monthly Training Volume")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Muscle group distribution
            if 'muscle_type' in workouts_df.columns:
                st.subheader("Training Distribution")
                muscle_dist = workouts_df.groupby(
                    'muscle_type')['volume'].sum().reset_index()
                muscle_dist['percentage'] = muscle_dist['volume'] / \
                    muscle_dist['volume'].sum() * 100

                fig = px.pie(muscle_dist, values='percentage', names='muscle_type',
                             title="Volume Distribution by Muscle Group")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No workout data available for analysis")

    with tabs[2]:
        st.header("📏 Body Metrics Analysis")
        if not body_metrics_df.empty:
            # Latest metrics
            latest = body_metrics_df.iloc[-1]
            if len(body_metrics_df) > 1:
                prev = body_metrics_df.iloc[-2]
            else:
                prev = None

            # Overview cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                weight_change = calculate_trend(
                    latest['user_weight'], prev['user_weight']) if prev is not None else None
                show_metric_card("Current Weight",
                                 latest['user_weight'], weight_change, " lbs")

            with col2:
                if not pd.isna(latest['height']) and latest['height'] > 0:
                    bmi = calculate_bmi(
                        latest['user_weight'], latest['height'])
                    bmi_status, _ = bmi_category(bmi)
                    show_metric_card("BMI", bmi, None, f" ({bmi_status})")

            with col3:
                if 'body_fat' in latest and not pd.isna(latest['body_fat']):
                    bf_change = calculate_trend(
                        latest['body_fat'], prev['body_fat']) if prev is not None else None
                    show_metric_card(
                        "Body Fat %", latest['body_fat'], bf_change, "%")

            with col4:
                if not pd.isna(latest['waist']) and not pd.isna(latest['hips']) and latest['hips'] > 0:
                    whr = latest['waist'] / latest['hips']
                    prev_whr = prev['waist'] / prev['hips'] if prev is not None and not pd.isna(
                        prev['waist']) and not pd.isna(prev['hips']) and prev['hips'] > 0 else None
                    whr_change = calculate_trend(
                        whr, prev_whr) if prev_whr is not None else None
                    show_metric_card("Waist-Hip Ratio", whr, whr_change, "", 3)

            # Metrics over time charts
            st.subheader("Body Composition Trends")

            col1, col2 = st.columns(2)

            with col1:
                # Weight trend
                fig = px.line(body_metrics_df, x='entry_date', y='user_weight',
                              title="Weight Over Time",
                              labels={'entry_date': 'Date',
                                      'user_weight': 'Weight (lbs)'},
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Body fat trend
                if 'body_fat' in body_metrics_df.columns and not body_metrics_df['body_fat'].isna().all():
                    fig = px.line(body_metrics_df, x='entry_date', y='body_fat',
                                  title="Body Fat % Over Time",
                                  labels={'entry_date': 'Date',
                                          'body_fat': 'Body Fat %'},
                                  markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No body fat data available")

            # Measurements comparison
            st.subheader("Body Measurements")
            measurements = ['chest', 'waist', 'hips',
                            'arms', 'thigh', 'calf', 'neck']
            available_measurements = [
                m for m in measurements if m in body_metrics_df.columns and not body_metrics_df[m].isna().all()]

            if available_measurements:
                fig = px.line(body_metrics_df, x='entry_date', y=available_measurements,
                              title="Body Measurements Over Time",
                              labels={'entry_date': 'Date',
                                      'value': 'Measurement (inches)'},
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No measurement data available")
        else:
            st.info("No body metrics data available for analysis")

    with tabs[3]:
        st.header("🎯 Goals & Progress")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Active Fitness Goals")
            if not goals_df.empty:
                for _, goal in goals_df.iterrows():
                    target_date = pd.to_datetime(goal['target_date'])
                    days_left = (target_date - pd.Timestamp.now()).days

                    # Get current value from latest body metrics
                    current_value = None
                    metric_name = goal['metric_name'].lower()

                    if not body_metrics_df.empty:
                        latest = body_metrics_df.iloc[-1]
                        if metric_name == 'weight' and not pd.isna(latest['user_weight']):
                            current_value = latest['user_weight']
                        elif metric_name == 'body fat' and 'body_fat' in latest and not pd.isna(latest['body_fat']):
                            current_value = latest['body_fat']
                        elif metric_name.lower() in latest and not pd.isna(latest[metric_name.lower()]):
                            current_value = latest[metric_name.lower()]

                    # Calculate progress percentage
                    if current_value is not None and not pd.isna(goal['target_value']):
                        progress_pct = min(
                            max(current_value / goal['target_value'], 0), 1)
                        progress_str = f"{goal['metric_name']}: {current_value:.1f} / {goal['target_value']:.1f} ({days_left} days left)"
                        st.progress(progress_pct, text=progress_str)
                    else:
                        st.write(
                            f"**{goal['metric_name']}**: Target {goal['target_value']:.1f} by {target_date.strftime('%Y-%m-%d')} ({days_left} days left)")
            else:
                st.info("No active goals set")
                st.write("Visit the Body Metrics page to set new goals")

        with col2:
            st.subheader("Progress Photos")
            if photo_counts:
                # Create a pie chart of photo types
                photo_df = pd.DataFrame({
                    'Type': photo_counts.keys(),
                    'Count': photo_counts.values()
                })

                fig = px.pie(photo_df, values='Count', names='Type',
                             title="Progress Photos by Type")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                st.write(
                    f"Total progress photos: {sum(photo_counts.values())}")
                st.write(
                    "Visit the Body Metrics page to view or upload progress photos")
            else:
                st.info("No progress photos uploaded yet")
                st.write("Visit the Body Metrics page to upload progress photos")

        # Overall progress summary
        st.subheader("Overall Progress Summary")
        if not body_metrics_df.empty and len(body_metrics_df) > 1:
            first_record = body_metrics_df.iloc[0]
            latest_record = body_metrics_df.iloc[-1]
            days_diff = (latest_record['entry_date'] -
                         first_record['entry_date']).days

            if days_diff > 0:
                # Calculate changes
                weight_change = latest_record['user_weight'] - \
                    first_record['user_weight']
                bf_change = (latest_record['body_fat'] - first_record['body_fat']) if 'body_fat' in latest_record and 'body_fat' in first_record and not pd.isna(
                    latest_record['body_fat']) and not pd.isna(first_record['body_fat']) else None

                # Create summary
                summary = pd.DataFrame({
                    'Metric': ['Start Date', 'Current Date', 'Days Tracking', 'Starting Weight', 'Current Weight', 'Weight Change'],
                    'Value': [
                        first_record['entry_date'].strftime('%Y-%m-%d'),
                        latest_record['entry_date'].strftime('%Y-%m-%d'),
                        str(days_diff),
                        f"{first_record['user_weight']:.1f} lbs",
                        f"{latest_record['user_weight']:.1f} lbs",
                        f"{weight_change:+.1f} lbs ({(weight_change/first_record['user_weight']*100):+.1f}%)"
                    ]
                })

                if bf_change is not None:
                    summary = pd.concat([summary, pd.DataFrame({
                        'Metric': ['Starting Body Fat', 'Current Body Fat', 'Body Fat Change'],
                        'Value': [
                            f"{first_record['body_fat']:.1f}%",
                            f"{latest_record['body_fat']:.1f}%",
                            f"{bf_change:+.1f}% ({(bf_change/first_record['body_fat']*100) if first_record['body_fat'] > 0 else 0:+.1f}%)"
                        ]
                    })])

                st.dataframe(summary, use_container_width=True,
                             hide_index=True)
            else:
                st.info("Need more data points for progress summary")
        else:
            st.info("Need more data points for progress summary")


def main():
    # Check password first
    if check_password():
        # Initialize the DB with necessary tables/columns
        init_db()

        # Initialize the database modification counter if it doesn't exist
        if "db_modification_counter" not in st.session_state:
            st.session_state.db_modification_counter = 0

        # Initialize the last refresh time if it doesn't exist
        if "last_refresh_time" not in st.session_state:
            st.session_state.last_refresh_time = time.time()

        # Initialize filters if they don't exist
        if "filters" not in st.session_state:
            st.session_state.filters = {
                "workout": [],
                "workout_type": [],
                "muscle_type": []
            }

        # Display the unified dashboard
        unified_dashboard()


if __name__ == "__main__":
    main()
