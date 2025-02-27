from auth import check_password
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from io import BytesIO
import base64
import os
from scipy.optimize import curve_fit
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from db_utils import create_connection  # adjust if needed
import sys
import time

# Add parent directory to path to import auth module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =======================
# PAGE CONFIGURATION & CUSTOM CSS
# =======================
st.set_page_config(layout="wide", page_title="Workout Progress Dashboard")
st.markdown(
    """
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }
    .css-1d391kg {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# =======================
# DATA LOADING & PREPROCESSING
# =======================


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
        with create_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM workouts", conn)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    if last_update is not None:
        # Pass both the file modification time and the counter to invalidate cache
        return get_workouts(last_update, st.session_state.db_modification_counter)
    else:
        return pd.DataFrame()

# --- 1RM FORMULAS ---


def epley_1rm(weight, reps):
    return weight * (1 + reps / 30) if reps > 0 else None


def brzycki_1rm(weight, reps):
    return weight * (36 / (37 - reps)) if 0 < reps < 37 else None


def lombardi_1rm(weight, reps):
    return weight * (reps ** 0.10) if reps > 0 else None

# =======================
# GLOBAL FILTERING (Sidebar)
# =======================


def filter_data(df):
    # Initialize the filters in session state if it doesn't exist
    if "filters" not in st.session_state:
        st.session_state.filters = {
            "workout": [],
            "workout_type": [],
            "muscle_type": []
        }

    st.sidebar.header("🔧 Global Filter Options")
    min_date = st.sidebar.date_input(
        "Start Date", value=df["date"].min(), key="filter_start_date")
    max_date = st.sidebar.date_input(
        "End Date", value=df["date"].max(), key="filter_end_date")

    all_workouts = sorted(df["workout"].unique().tolist())
    if st.sidebar.checkbox("Filter by specific workouts?", value=False, key="custom_workouts"):
        selected_workouts = st.sidebar.multiselect(
            "Select Workouts", options=all_workouts, default=[],
            help="Select one or more workouts (leave empty to include all)."
        )
        if not selected_workouts:
            selected_workouts = all_workouts
    else:
        selected_workouts = all_workouts

    # Store selected workouts in session state
    st.session_state.filters["workout"] = selected_workouts if selected_workouts != all_workouts else [
    ]

    if "workout_type" in df.columns:
        all_types = sorted(df["workout_type"].dropna().unique().tolist())
        selected_types = st.sidebar.multiselect(
            "Select Workout Types", options=all_types, default=all_types)
        # Store selected workout types in session state
        st.session_state.filters["workout_type"] = selected_types if selected_types != all_types else [
        ]
    else:
        selected_types = []
        st.session_state.filters["workout_type"] = []

    if "muscle_type" in df.columns:
        all_muscles = sorted(df["muscle_type"].dropna().unique().tolist())
        selected_muscles = st.sidebar.multiselect(
            "Select Muscle Types", options=all_muscles, default=all_muscles)
        # Store selected muscle types in session state
        st.session_state.filters["muscle_type"] = selected_muscles if selected_muscles != all_muscles else [
        ]
    else:
        selected_muscles = []
        st.session_state.filters["muscle_type"] = []

    filter_logic = st.sidebar.radio(
        "Combine Workout Filters with:", options=["AND", "OR"], index=0)
    mask = (df["date"] >= pd.to_datetime(min_date)) & (
        df["date"] <= pd.to_datetime(max_date))
    workout_mask = df["workout"].isin(
        selected_workouts) if selected_workouts else pd.Series(True, index=df.index)
    type_mask = df["workout_type"].isin(selected_types) if (
        "workout_type" in df.columns and selected_types) else pd.Series(True, index=df.index)
    muscle_mask = df["muscle_type"].isin(selected_muscles) if (
        "muscle_type" in df.columns and selected_muscles) else pd.Series(True, index=df.index)
    if filter_logic == "AND":
        combined = workout_mask & type_mask & muscle_mask
    else:
        combined = workout_mask | type_mask | muscle_mask
    mask &= combined
    return df.loc[mask].copy()

# =======================
# KPI SECTION
# =======================


def key_performance_indicators(df):
    st.subheader("📊 Key Performance Indicators")
    if df.empty:
        st.info("No data available for KPIs.")
        return

    # Group by date and workout to get proper session counts
    daily_workouts = df.groupby(['date', 'workout']).agg({
        'sets': 'sum',
        'volume': 'sum',
        'weight': ['max', 'mean'],
        'reps': 'sum',
        'Epley_1RM': 'max'
    }).reset_index()

    total_sessions = daily_workouts['date'].nunique()

    # Calculate total volume directly from original dataframe to avoid multi-index issues
    total_volume = df['volume'].sum()
    avg_volume_per_session = total_volume / \
        total_sessions if total_sessions > 0 else 0

    # Calculate max volume by date directly from original dataframe
    daily_volume = df.groupby('date')['volume'].sum()
    max_volume = daily_volume.max() if not daily_volume.empty else 0

    # Most frequent workout based on unique days
    freq_workout = daily_workouts['workout'].value_counts()
    most_freq_workout = freq_workout.idxmax() if not freq_workout.empty else "None"
    freq_count = freq_workout.max() if not freq_workout.empty else 0

    # Handle multi-index columns properly
    max_weight = df['weight'].max()
    avg_sets = df.groupby(['date', 'workout'])[
        'sets'].sum().mean() if not df.empty else 0
    avg_reps = df.groupby(['date', 'workout'])[
        'reps'].mean().mean() if not df.empty else 0
    avg_1rm = df['Epley_1RM'].mean() if 'Epley_1RM' in df.columns else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", f"{total_sessions}")
        st.metric("Total Volume", f"{total_volume:,.0f} lbs")
        st.metric("Avg. Volume/Session", f"{avg_volume_per_session:,.0f} lbs")
    with col2:
        st.metric("Max Volume (Session)", f"{max_volume:,.0f} lbs")
        st.metric("Most Frequent Workout",
                  f"{most_freq_workout} ({freq_count} times)")
    with col3:
        st.metric("Max Weight", f"{max_weight:,.0f} lbs")
        st.metric("Avg. Sets", f"{avg_sets:.1f}")
        st.metric("Avg. Reps", f"{avg_reps:.1f}")
        st.metric("Avg. Best 1RM", f"{avg_1rm:,.0f} lbs")

# =======================
# SUMMARY STATISTICS
# =======================


def summary_statistics(df):
    st.subheader("🔍 Summary Statistics")

    # Group by date and workout first to combine same-day workouts
    daily_stats = df.groupby(['date', 'workout']).agg({
        'sets': 'sum',
        'weight': ['mean', 'max'],
        'volume': 'sum',
        'reps': ['sum', 'mean']
    }).reset_index()

    # Then create summary statistics
    workout_stats = daily_stats.groupby('workout').agg({
        ('sets', 'sum'): 'sum',
        ('weight', 'mean'): 'mean',
        ('weight', 'max'): 'max',
        ('volume', 'sum'): 'sum',
        ('reps', 'mean'): 'mean'
    }).reset_index()

    # Rename columns for clarity
    workout_stats.columns = ['Workout', 'Total Sets',
                             'Avg Weight (lbs)', 'Max Weight (lbs)', 'Total Volume', 'Avg Reps/Set']

    # Format the statistics
    workout_stats['Avg Weight (lbs)'] = workout_stats['Avg Weight (lbs)'].round(
        1)
    workout_stats['Total Volume'] = workout_stats['Total Volume'].round(0)
    workout_stats['Avg Reps/Set'] = workout_stats['Avg Reps/Set'].round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Workout Summary Statistics**")
        st.dataframe(workout_stats, use_container_width=True)

    with col2:
        st.write("**Recent Progress (Last 30 Days)**")
        recent_stats = daily_stats[daily_stats['date'] >= (
            pd.Timestamp.now() - pd.Timedelta(days=30))].copy()
        if not recent_stats.empty:
            recent_progress = recent_stats.groupby('workout').agg({
                ('volume', 'sum'): 'sum',
                ('weight', 'max'): 'max'
            }).reset_index()
            recent_progress.columns = ['Workout',
                                       'Recent Volume', 'Recent Max Weight']
            st.dataframe(recent_progress, use_container_width=True)
        else:
            st.info("No workouts recorded in the last 30 days")

# =======================
# VISUALIZATIONS
# =======================


def visualizations(df):
    st.subheader("📈 Visualizations")
    sets_summary = df.groupby("workout")["sets"].sum(
    ).reset_index().rename(columns={"sets": "Total Sets"})
    weight_summary = df.groupby("workout")["weight"].mean(
    ).reset_index().rename(columns={"weight": "Avg Weight (lbs)"})
    volume_summary = df.groupby("workout")["volume"].sum(
    ).reset_index().rename(columns={"volume": "Total Volume"})
    col1, col2 = st.columns(2)
    with col1:
        fig_sets = px.bar(sets_summary, x="workout", y="Total Sets",
                          title="Total Sets by Workout", text_auto=True)
        st.plotly_chart(fig_sets, use_container_width=True)
        fig_weight = px.bar(weight_summary, x="workout", y="Avg Weight (lbs)",
                            title="Average Weight by Workout", text_auto=True)
        st.plotly_chart(fig_weight, use_container_width=True)
    with col2:
        fig_volume = px.bar(volume_summary, x="workout", y="Total Volume",
                            title="Total Volume by Workout", text_auto=True)
        st.plotly_chart(fig_volume, use_container_width=True)
    by_date = df.groupby("date")["volume"].sum(
    ).reset_index().sort_values("date")
    fig_line = px.line(by_date, x="date", y="volume", title="Total Volume Over Time", markers=True,
                       labels={"volume": "Total Volume", "date": "Date"})
    st.plotly_chart(fig_line, use_container_width=True)

# =======================
# PERSONAL BESTS
# =======================


def personal_bests(df):
    st.subheader("🏅 Personal Bests")
    pb = df.groupby("workout")["weight"].max().reset_index().rename(
        columns={"weight": "Max Weight (lbs)"})
    st.write("**Max Weight by Workout**")
    st.dataframe(pb, use_container_width=True)
    st.write("**Estimated 1RM (Epley, Brzycki, Lombardi)**")
    st.dataframe(
        df[["date", "workout", "sets", "reps", "weight",
            "Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"]]
        .rename(columns={"date": "Date", "workout": "Workout", "sets": "Sets", "reps": "Reps", "weight": "Weight (lbs)"}),
        use_container_width=True
    )
    workouts = sorted(df["workout"].unique().tolist())
    selected_workout = st.selectbox("Select a Workout", workouts)
    df_selected = df[df["workout"] == selected_workout].sort_values("date")
    if not df_selected.empty:
        fig_1rm = px.line(df_selected, x="date", y=["Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"],
                          title=f"Estimated 1RM Over Time - {selected_workout}",
                          labels={"value": "Estimated 1RM",
                                  "date": "Date", "variable": "Formula"},
                          markers=True)
        st.plotly_chart(fig_1rm, use_container_width=True)
    else:
        st.info("ℹ️ No data available for the selected workout.")

# =======================
# ADVANCED ANALYTICS TAB
# =======================


def advanced_analytics_tab(df):
    st.header("Advanced Analytics & Modeling")

    st.subheader("Correlation Analysis")
    corr_cols = ["sets", "reps", "weight", "volume",
                 "Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"]
    available_cols = [
        col for col in corr_cols if col in df.columns and df[col].notnull().any()]
    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        st.dataframe(corr_matrix, use_container_width=True)
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                             title="Correlation Heatmap", color_continuous_scale='RdBu', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric data for correlation analysis.")

    # Add animated visualization of progress over time
    st.subheader("✨ Animated Progress Visualization")
    if len(df) > 0 and 'date' in df.columns and 'workout_type' in df.columns:
        # Prepare data for animation - aggregate by date and workout_type
        anim_data = df.groupby(['date', 'workout_type']).agg({
            'volume': 'sum',
            'weight': 'max'
        }).reset_index()

        # Create animation of volume by workout type over time
        fig_anim = px.scatter(
            anim_data,
            x='weight',
            y='volume',
            size='volume',
            color='workout_type',
            animation_frame=pd.to_datetime(
                anim_data['date']).dt.strftime('%Y-%m-%d'),
            animation_group='workout_type',
            range_x=[0, anim_data['weight'].max() * 1.1],
            range_y=[0, anim_data['volume'].max() * 1.1],
            title="Workout Progress Animation (Weight vs Volume)",
            labels={
                'weight': 'Max Weight (lbs)',
                'volume': 'Total Volume (lbs)',
                'workout_type': 'Workout Type'
            }
        )

        # Customize animation settings
        fig_anim.update_layout(
            height=600,
            xaxis_title="Max Weight (lbs)",
            yaxis_title="Total Volume (lbs)"
        )
        # Improve play button settings
        fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
        fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

        st.plotly_chart(fig_anim, use_container_width=True)
        st.caption(
            "▶️ Play the animation to see how your workout volume and max weight have changed over time")
    else:
        st.info("Not enough data for animated visualization. Make sure your data includes dates and workout types.")

    st.subheader("Weight vs. Volume Scatter Plot")
    if "weight" in df.columns and "volume" in df.columns:
        corr_val = df["weight"].corr(df["volume"])
        st.write(f"Correlation (Weight vs. Volume): {corr_val:.2f}")
        fig_scatter = px.scatter(df, x="weight", y="volume", trendline="ols",
                                 title="Weight vs. Volume", hover_data=["workout", "date"])
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("7-Day Moving Average of Volume")
    try:
        moving_avg = df.groupby("date")["volume"].sum(
        ).reset_index().sort_values("date")
        moving_avg["7d_MA"] = moving_avg["volume"].rolling(window=7).mean()
        fig_ma = px.line(moving_avg, x="date", y=["volume", "7d_MA"],
                         title="Total Volume & 7-Day Moving Average", markers=True,
                         labels={"value": "Volume", "date": "Date", "variable": "Metric"})
        st.plotly_chart(fig_ma, use_container_width=True)
    except Exception as e:
        st.error(f"Error calculating moving average: {e}")

    st.subheader("Derivatives of Volume Over Time")
    vol_daily = df.groupby("date")["volume"].sum(
    ).reset_index().sort_values("date")
    vol_daily["vol_ma"] = vol_daily["volume"].rolling(
        window=3, center=True, min_periods=1).mean()
    vol_daily["dV/dt"] = vol_daily["vol_ma"].diff()
    vol_daily["d2V/dt2"] = vol_daily["dV/dt"].diff()
    fig_deriv = go.Figure()
    fig_deriv.add_trace(go.Scatter(
        x=vol_daily["date"], y=vol_daily["dV/dt"], mode='lines+markers', name='1st Derivative'))
    fig_deriv.add_trace(go.Scatter(
        x=vol_daily["date"], y=vol_daily["d2V/dt2"], mode='lines+markers', name='2nd Derivative'))
    fig_deriv.update_layout(title="Derivatives of Volume Over Time",
                            xaxis_title="Date", yaxis_title="Approx. Derivative")
    st.plotly_chart(fig_deriv, use_container_width=True)

    st.subheader("Differential Equation Modeling")
    daily_df = df.groupby("date")["volume"].sum(
    ).reset_index().sort_values("date")
    if len(daily_df) < 5:
        st.warning("Not enough data points for model fitting (need at least 5).")
    else:
        xdata = daily_df["date"].apply(
            lambda x: x.toordinal()).values.astype(float)
        ydata = daily_df["volume"].values.astype(float)

        def exponential_model(x, a, b, c):
            return a * np.exp(b * (x - x.min())) + c

        def logistic_model(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))
        try:
            initial_exp = [max(ydata), 0.01, min(ydata)]
            popt_exp, _ = curve_fit(
                exponential_model, xdata, ydata, p0=initial_exp, maxfev=10000)
            y_exp_fit = exponential_model(xdata, *popt_exp)
            residuals_exp = ydata - y_exp_fit
            r2_exp = 1 - (np.sum(residuals_exp**2) /
                          np.sum((ydata - np.mean(ydata))**2))
        except Exception as e:
            popt_exp, r2_exp = None, None
        try:
            initial_log = [max(ydata), 0.1, np.median(xdata)]
            popt_log, _ = curve_fit(
                logistic_model, xdata, ydata, p0=initial_log, maxfev=10000)
            y_log_fit = logistic_model(xdata, *popt_log)
            residuals_log = ydata - y_log_fit
            r2_log = 1 - (np.sum(residuals_log**2) /
                          np.sum((ydata - np.mean(ydata))**2))
        except Exception as e:
            popt_log, r2_log = None, None
        fig_model = go.Figure()
        fig_model.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["volume"],
                                       mode='markers', name='Actual Volume'))
        if popt_exp is not None:
            fig_model.add_trace(go.Scatter(x=daily_df["date"], y=y_exp_fit,
                                           mode='lines', name=f'Exponential Fit (R²={r2_exp:.2f})'))
        if popt_log is not None:
            fig_model.add_trace(go.Scatter(x=daily_df["date"], y=y_log_fit,
                                           mode='lines', name=f'Logistic Fit (R²={r2_log:.2f})'))
        fig_model.update_layout(
            title="Modeling Total Volume Over Time", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_model, use_container_width=True)
        if popt_exp is not None:
            st.write(
                f"Exponential Model: a={popt_exp[0]:.2f}, b={popt_exp[1]:.4f}, c={popt_exp[2]:.2f}, R²={r2_exp:.2f}")
        if popt_log is not None:
            st.write(
                f"Logistic Model: L={popt_log[0]:.2f}, k={popt_log[1]:.4f}, x₀={popt_log[2]:.2f}, R²={r2_log:.2f}")

    st.subheader("Scatter Matrix of Key Variables")
    scatter_matrix_fig = px.scatter_matrix(df, dimensions=["sets", "reps", "weight", "volume", "Epley_1RM"],
                                           title="Scatter Matrix of Key Variables")
    st.plotly_chart(scatter_matrix_fig, use_container_width=True)

    st.subheader("Box & Violin Plots")
    col_a, col_b = st.columns(2)
    with col_a:
        if "workout_type" in df.columns:
            fig_box = px.box(df, x="workout_type", y="weight",
                             title="Box Plot: Weight by Workout Type")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Workout type data not available.")
    with col_b:
        fig_violin = px.violin(df, x="workout", y="volume", box=True, points="all",
                               title="Violin Plot: Volume by Workout")
        st.plotly_chart(fig_violin, use_container_width=True)

    st.subheader("7-Day Rolling Volatility of Daily Volume")
    vol_daily = df.groupby("date")["volume"].sum(
    ).reset_index().sort_values("date")
    vol_daily["rolling_std"] = vol_daily["volume"].rolling(window=7).std()
    fig_volatility = px.line(vol_daily, x="date", y="rolling_std",
                             title="7-Day Rolling Volatility", labels={"rolling_std": "Std Dev", "date": "Date"})
    st.plotly_chart(fig_volatility, use_container_width=True)

    st.subheader("Polynomial Regression: Volume vs Weight")
    X = df["weight"].values.reshape(-1, 1)
    y = df["volume"].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    df["predicted_volume"] = model.predict(X_poly)
    fig_poly = px.scatter(df, x="weight", y="volume", title="Volume vs Weight with Polynomial Regression",
                          labels={"volume": "Volume", "weight": "Weight"})
    fig_poly.add_traces(px.line(df, x="weight", y="predicted_volume").data)
    st.plotly_chart(fig_poly, use_container_width=True)

    st.subheader("Cluster Analysis of Workouts")
    features = df[["weight", "sets", "reps", "volume", "Epley_1RM"]].dropna()
    if len(features) > 10:
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            df_cluster = pd.DataFrame(components, columns=["PC1", "PC2"])
            df_cluster["Cluster"] = clusters.astype(str)
            fig_cluster = px.scatter(
                df_cluster, x="PC1", y="PC2", color="Cluster",
                title="KMeans Clustering (3 Clusters)",
                labels={"PC1": "Principal Component 1",
                        "PC2": "Principal Component 2"}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        except Exception as e:
            st.error(f"Error in cluster analysis: {e}")
    else:
        st.info("Not enough data for clustering analysis.")

    st.subheader("Density Heatmap: Weight vs Reps")
    fig_density = px.density_heatmap(df, x="weight", y="reps", nbinsx=20, nbinsy=20,
                                     title="Density Heatmap: Weight vs Reps")
    st.plotly_chart(fig_density, use_container_width=True)

    st.subheader("Regression Analysis for 1RM Prediction")
    features_reg = df[["weight", "reps", "sets"]].dropna()
    target = df.loc[features_reg.index, "Epley_1RM"]
    reg = LinearRegression()
    reg.fit(features_reg, target)
    df["predicted_1RM"] = reg.predict(df[["weight", "reps", "sets"]])
    fig_reg = px.scatter(df, x="Epley_1RM", y="predicted_1RM",
                         title="Actual vs Predicted Epley 1RM",
                         labels={"Epley_1RM": "Actual Epley 1RM", "predicted_1RM": "Predicted Epley 1RM"})
    fig_reg.add_trace(go.Scatter(x=df["Epley_1RM"], y=df["Epley_1RM"],
                                 mode="lines", name="Ideal Fit"))
    st.plotly_chart(fig_reg, use_container_width=True)

    st.subheader("Parallel Coordinates Plot")
    # Create a numeric mapping for workout types
    workout_type_map = {'Push': 0, 'Pull': 1,
                        'Leg': 2, 'Core': 3, 'Full-Body': 4}
    parallel_data = df.copy()
    parallel_data['workout_type_num'] = parallel_data['workout_type'].map(
        workout_type_map)

    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        parallel_data,
        color='workout_type_num',  # Use numeric values for color instead of categorical
        dimensions=['volume', 'weight',
                    'sets', 'reps'],  # Removed 'exercise_count' as it doesn't exist in the DataFrame
        title='Multi-dimensional Workout Analysis',
        color_continuous_scale=px.colors.qualitative.G10,  # Use a qualitative color scale
        labels={
            'volume': 'Volume (lbs)',
            'weight': 'Max Weight (lbs)',
            'sets': 'Total Sets',
            'reps': 'Total Reps',
            'workout_type_num': 'Workout Type'
        }
    )

    # Add proper color axis configuration
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Workout Type",
            tickvals=list(workout_type_map.values()),
            ticktext=list(workout_type_map.keys())
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# =======================
# FUTURE PLANNING TAB
# =======================


def future_planning_tab(df):
    st.header("🎯 Future Workout Planning")

    if df.empty:
        st.info("No workout data available for planning.")
        return

    # Get all unique exercises from the database
    all_exercises = sorted(df['workout'].unique())

    # If workout_type exists in the database, use it for categorization
    if 'workout_type' in df.columns:
        workout_categories = {}
        for workout_type in df['workout_type'].unique():
            if pd.notna(workout_type):
                exercises = df[df['workout_type'] ==
                               workout_type]['workout'].unique().tolist()
                workout_categories[workout_type] = {
                    'Primary': exercises,
                    'Secondary': []
                }
    else:
        # If no workout_type column, try to infer categories based on exercise names
        workout_categories = {
            'Push': {'Primary': [], 'Secondary': []},
            'Pull': {'Primary': [], 'Secondary': []},
            'Leg': {'Primary': [], 'Secondary': []}
        }

        # Basic categorization based on common exercise names
        for exercise in all_exercises:
            ex_lower = exercise.lower()
            if any(term in ex_lower for term in ['bench', 'press', 'shoulder', 'chest', 'tricep']):
                workout_categories['Push']['Primary'].append(exercise)
            elif any(term in ex_lower for term in ['row', 'pull', 'curl', 'back', 'lat']):
                workout_categories['Pull']['Primary'].append(exercise)
            elif any(term in ex_lower for term in ['squat', 'dead', 'leg']):
                workout_categories['Leg']['Primary'].append(exercise)

    # Get the last workout date
    last_workout_date = df['date'].max()
    st.write(f"Last workout was on: {last_workout_date.strftime('%Y-%m-%d')}")

    # Create tabs for different sections
    main_tabs = st.tabs(
        ['Workout Recommendations', 'Progress Tracking', 'Long-term Planning'])

    with main_tabs[0]:
        st.subheader("📋 Workout Recommendations")

        # Create tabs for Push/Pull/Leg
        workout_tabs = st.tabs(['Push', 'Pull', 'Leg'])

        for tab_idx, workout_type in enumerate(['Push', 'Pull', 'Leg']):
            with workout_tabs[tab_idx]:
                if workout_type in workout_categories:
                    exercises = workout_categories[workout_type]

                    # Create recommendations table
                    recommendations = []

                    for exercise in exercises['Primary']:
                        exercise_history = df[df['workout']
                                              == exercise].sort_values('date')
                        if not exercise_history.empty:
                            # Group by date to handle multiple sets in the same workout
                            daily_exercise = exercise_history.groupby('date').agg({
                                'weight': ['max', 'mean'],
                                'sets': 'sum',
                                'reps': ['sum', 'mean'],
                                'volume': 'sum'
                            }).reset_index()

                            # Get the last workout day
                            last_workout = daily_exercise.iloc[-1]

                            # Get personal best (by weight)
                            best_weight = daily_exercise['weight']['max'].max()
                            best_workout = daily_exercise[daily_exercise['weight']
                                                          ['max'] == best_weight].iloc[0]

                            # Calculate recommendations based on last workout performance
                            last_max_weight = last_workout['weight']['max']
                            last_avg_weight = last_workout['weight']['mean']
                            total_sets = last_workout['sets']['sum']
                            total_reps = last_workout['reps']['sum']
                            avg_reps_per_set = last_workout['reps']['mean']

                            # Get the full breakdown of the last workout
                            last_workout_detail = exercise_history[exercise_history['date']
                                                                   == last_workout.name]
                            sets_breakdown = [
                                f"{row['weight']}x{row['reps']}" for _, row in last_workout_detail.iterrows()]
                            last_workout_str = " | ".join(sets_breakdown)

                            # Progressive overload strategy based on average reps per set
                            if avg_reps_per_set >= 12:
                                weight_recommendation = last_max_weight + 5
                                reps_recommendation = "6-8"
                                sets_recommendation = total_sets
                                note = "Increase weight, reset reps lower"
                            elif avg_reps_per_set < 6:
                                weight_recommendation = max(
                                    last_max_weight - 5, 0)
                                reps_recommendation = "8-10"
                                sets_recommendation = total_sets
                                note = "Decrease weight to focus on form"
                            else:
                                weight_recommendation = last_max_weight
                                reps_recommendation = f"{int(avg_reps_per_set) + 1}-{int(avg_reps_per_set) + 2}"
                                sets_recommendation = min(total_sets + 1, 5)
                                note = "Increase reps/sets for volume progression"

                            recommendations.append({
                                'Exercise': exercise,
                                'Last Workout': last_workout_str,
                                'Total Sets': total_sets,
                                'Avg Reps/Set': f"{avg_reps_per_set:.1f}",
                                'Max Weight': f"{last_max_weight:.1f}",
                                'Total Volume': f"{last_workout['volume']['sum']:,.0f}",
                                'Recommended Weight': f"{weight_recommendation:.1f}",
                                'Target Sets': sets_recommendation,
                                'Target Reps': reps_recommendation,
                                'Personal Best': f"{best_weight:.1f}",
                                'Notes': note
                            })

                    if recommendations:
                        # Convert to DataFrame for better display
                        df_recommendations = pd.DataFrame(recommendations)

                        # Display last workout details
                        st.write("#### Last Workout Details")
                        last_workout_df = df_recommendations[[
                            'Exercise', 'Last Workout', 'Total Sets', 'Avg Reps/Set', 'Max Weight', 'Total Volume']]
                        st.dataframe(last_workout_df, use_container_width=True)

                        # Display recommendations
                        st.write("#### Recommendations for Next Workout")
                        next_workout_df = df_recommendations[[
                            'Exercise', 'Recommended Weight', 'Target Sets', 'Target Reps', 'Notes']]
                        st.dataframe(next_workout_df, use_container_width=True)

                        # Display personal bests
                        st.write("#### Personal Bests")
                        pb_df = df_recommendations[[
                            'Exercise', 'Personal Best']]
                        st.dataframe(pb_df, use_container_width=True)
                    else:
                        st.info(
                            f"No previous {workout_type} workout data available.")

    # Rest of the code for Progress Tracking and Long-term Planning tabs remains the same...

# =======================
# DASHBOARD TAB
# =======================


def dashboard_tab(df):
    st.header("📊 Workout Dashboard")

    if df.empty:
        st.info("No workout data available.")
        return

    # Create date filters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", value=df['date'].min(), key="dashboard_start_date")
    with col2:
        end_date = st.date_input(
            "End Date", value=df['date'].max(), key="dashboard_end_date")

    # Filter data based on date range
    mask = (df['date'] >= pd.to_datetime(start_date)) & (
        df['date'] <= pd.to_datetime(end_date))
    filtered_df = df[mask]

    # Top row metrics
    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics
    total_workouts = filtered_df['date'].nunique()
    total_volume = filtered_df['volume'].sum()
    avg_workouts_per_week = total_workouts / \
        max(1, (filtered_df['date'].max() -
            filtered_df['date'].min()).days / 7)
    max_weights = filtered_df.groupby('workout')['weight'].max()

    with col1:
        st.metric("Total Workouts", f"{total_workouts}")
    with col2:
        st.metric("Total Volume", f"{total_volume:,.0f} lbs")
    with col3:
        st.metric("Workouts/Week", f"{avg_workouts_per_week:.1f}")
    with col4:
        if not max_weights.empty:
            strongest_lift = max_weights.idxmax()
            max_weight = max_weights.max()
            st.metric("Strongest Lift",
                      f"{strongest_lift}: {max_weight:.0f} lbs")

    # Daily Volume Line Graph
    st.subheader("📈 Daily Training Volume")
    daily_volume = filtered_df.groupby('date')['volume'].sum().reset_index()

    # Create line graph with hover data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_volume['date'],
        y=daily_volume['volume'],
        mode='lines+markers',
        name='Daily Volume',
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br>" +
        "<b>Volume:</b> %{y:,.0f} lbs<br>" +
        "<extra></extra>"
    ))

    # Customize layout
    fig.update_layout(
        title="Daily Training Volume Over Time",
        xaxis_title="Date",
        yaxis_title="Volume (lbs)",
        hovermode='x unified',
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add animated performance trends visualization
    st.subheader("🎬 Performance Trends Animation")
    if not filtered_df.empty and 'workout' in filtered_df.columns:
        # Get top 5 most frequent exercises
        top_exercises = filtered_df['workout'].value_counts().head(
            5).index.tolist()

        # Filter data for these exercises
        top_exercise_data = filtered_df[filtered_df['workout'].isin(
            top_exercises)]

        if not top_exercise_data.empty:
            # Prepare data for the animation - find max weight by date for each exercise
            performance_data = top_exercise_data.groupby(['date', 'workout']).agg({
                'weight': 'max',
                'reps': 'max',
                'volume': 'sum'
            }).reset_index()

            # Create the animated line chart
            fig_anim = px.line(
                performance_data,
                x='date',
                y='weight',
                color='workout',
                animation_frame=pd.to_datetime(
                    performance_data['date']).dt.strftime('%Y-%m-%d'),
                range_y=[0, performance_data['weight'].max() * 1.1],
                title="Exercise Performance Over Time",
                labels={
                    'date': 'Date',
                    'weight': 'Max Weight (lbs)',
                    'workout': 'Exercise'
                },
                markers=True
            )

            # Customize animation settings
            fig_anim.update_layout(
                height=500,
                xaxis_title="Date",
                yaxis_title="Max Weight (lbs)"
            )

            # Improve animation playback settings
            fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 600
            fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300

            st.plotly_chart(fig_anim, use_container_width=True)
            st.caption(
                "▶️ Press play to watch how your strength has progressed for your top exercises over time")
        else:
            st.info("Not enough exercise data to create animation.")
    else:
        st.info("Not enough data for animation visualization.")

    # Create two rows with two columns each
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Recent Progress (Last 30 days vs Previous 30 days)
    with row1_col1:
        st.subheader("📈 30-Day Progress")
        last_30_days = filtered_df[filtered_df['date'] >= (
            filtered_df['date'].max() - pd.Timedelta(days=30))]
        prev_30_days = filtered_df[
            (filtered_df['date'] < (filtered_df['date'].max() - pd.Timedelta(days=30))) &
            (filtered_df['date'] >=
             (filtered_df['date'].max() - pd.Timedelta(days=60)))
        ]

        progress_metrics = []
        for period, data in [("Last 30 Days", last_30_days), ("Previous 30 Days", prev_30_days)]:
            if not data.empty:
                metrics = {
                    "Period": period,
                    "Total Volume": data['volume'].sum(),
                    "Max Weight": data['weight'].max(),
                    "Workouts": data['date'].nunique()
                }
                progress_metrics.append(metrics)

        if len(progress_metrics) == 2:
            progress_df = pd.DataFrame(progress_metrics)
            st.dataframe(progress_df, use_container_width=True)

            # Calculate and show percentage changes
            volume_change = (
                (progress_metrics[0]["Total Volume"] / progress_metrics[1]["Total Volume"]) - 1) * 100
            weight_change = (
                (progress_metrics[0]["Max Weight"] / progress_metrics[1]["Max Weight"]) - 1) * 100
            workouts_change = progress_metrics[0]["Workouts"] - \
                progress_metrics[1]["Workouts"]

            st.write(f"Volume Change: {volume_change:+.1f}%")
            st.write(f"Max Weight Change: {weight_change:+.1f}%")
            st.write(
                f"Workout Frequency Change: {workouts_change:+.0f} sessions")

    # Weekly Volume Trend
    with row1_col2:
        st.subheader("📊 Weekly Volume")
        weekly_volume = filtered_df.groupby(pd.Grouper(key='date', freq='W'))[
            'volume'].sum().reset_index()
        fig = px.line(weekly_volume, x='date', y='volume',
                      title='Weekly Training Volume',
                      labels={'volume': 'Volume (lbs)', 'date': 'Week'})
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Exercise Distribution
    with row2_col1:
        st.subheader("🎯 Exercise Focus")
        exercise_volume = filtered_df.groupby(
            'workout')['volume'].sum().reset_index()
        exercise_volume['percentage'] = exercise_volume['volume'] / \
            exercise_volume['volume'].sum() * 100

        fig = px.pie(exercise_volume, values='percentage', names='workout',
                     title='Volume Distribution by Exercise')
        fig.update_layout(showlegend=True, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Personal Records Table
    with row2_col2:
        st.subheader("🏆 Personal Records")
        pr_data = []
        for workout in filtered_df['workout'].unique():
            workout_data = filtered_df[filtered_df['workout'] == workout]
            pr_data.append({
                'Exercise': workout,
                'Max Weight': workout_data['weight'].max(),
                'Max Volume': workout_data['volume'].max(),
                'Date Achieved': workout_data.loc[workout_data['weight'].idxmax(), 'date'].strftime('%Y-%m-%d')
            })

        pr_df = pd.DataFrame(pr_data)
        st.dataframe(pr_df.sort_values(
            'Max Weight', ascending=False), use_container_width=True)

    # Workout Calendar Heatmap
    st.subheader("📅 Workout Calendar")
    calendar_data = filtered_df.groupby('date')['volume'].sum().reset_index()
    calendar_data['week'] = calendar_data['date'].dt.strftime('%Y-%U')
    calendar_data['weekday'] = calendar_data['date'].dt.day_name()

    weekday_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
    calendar_data['weekday'] = pd.Categorical(
        calendar_data['weekday'], categories=weekday_order, ordered=True)

    pivot_table = calendar_data.pivot_table(
        index='weekday',
        columns='week',
        values='volume',
        aggfunc='sum',
        fill_value=0
    )

    fig = px.imshow(pivot_table,
                    labels=dict(x='Week', y='Day', color='Volume'),
                    aspect='auto',
                    title='Training Volume Heatmap')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# =======================
# DOWNLOAD & RAW DATA TAB
# =======================


def download_reports_tab(df):
    st.header("Downloads")
    st.markdown("""
    **Download Options:**
    - Download the **raw data** as CSV.
    - Download a comprehensive **Excel report** with multiple sheets.
    """)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📄 Download Raw Data as CSV", data=csv,
                       file_name='workout_data.csv', mime='text/csv')
    if st.button("📊 Download Analytics Report as Excel"):
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                summary = {
                    "Total Sets": df.groupby("workout")["sets"].sum(),
                    "Average Weight": df.groupby("workout")["weight"].mean(),
                    "Total Volume": df.groupby("workout")["volume"].sum(),
                    "Personal Bests": df.groupby("workout")["weight"].max(),
                    "Max Epley 1RM": df.groupby("workout")["Epley_1RM"].max(),
                    "Max Brzycki 1RM": df.groupby("workout")["Brzycki_1RM"].max(),
                    "Max Lombardi 1RM": df.groupby("workout")["Lombardi_1RM"].max()
                }
                for sheet_name, data in summary.items():
                    data.to_frame().to_excel(writer, sheet_name=sheet_name)
                df.to_excel(writer, sheet_name="FullData", index=False)
                corr_cols = ["sets", "reps", "weight", "volume",
                             "Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"]
                available_corr_cols = [
                    col for col in corr_cols if col in df.columns and df[col].notnull().any()]
                if len(available_corr_cols) > 1:
                    df[available_corr_cols].corr().to_excel(
                        writer, sheet_name="Correlation")
            processed_data = output.getvalue()
            b64 = base64.b64encode(processed_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="analytics_report.xlsx">📥 Click here to download the Excel report</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error generating the Excel report: {e}")


def raw_data_tab(df):
    st.header("Raw Data")
    st.dataframe(df, use_container_width=True)

# =======================
# MAIN APPLICATION
# =======================


def main():
    # Check password first
    if not check_password():
        st.stop()

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

    # Add refresh controls in the top right corner
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        st.title("📊 Workout Progress Dashboard")
    with col2:
        auto_refresh = st.checkbox("Auto refresh", value=False)
    with col3:
        if st.button("🔄 Refresh"):
            # Increment counter to invalidate cache
            st.session_state.db_modification_counter += 1
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    # Handle auto-refresh (every 30 seconds)
    if auto_refresh and time.time() - st.session_state.last_refresh_time > 30:
        st.session_state.db_modification_counter += 1
        st.session_state.last_refresh_time = time.time()
        st.rerun()

    # Load the data with proper caching that invalidates when database changes
    df = load_workouts()
    if df.empty:
        st.warning("⚠️ No workouts logged yet. Please add or upload data.")
        return

    # Preprocessing
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce")
    df["volume"] = df["weight"] * \
        df["sets"] * df["reps"]
    df["Epley_1RM"] = df.apply(
        lambda row: epley_1rm(row["weight"], row["reps"]), axis=1)
    df["Brzycki_1RM"] = df.apply(
        lambda row: brzycki_1rm(row["weight"], row["reps"]), axis=1)
    df["Lombardi_1RM"] = df.apply(
        lambda row: lombardi_1rm(row["weight"], row["reps"]), axis=1)

    filtered_df = filter_data(df)
    st.sidebar.markdown("### Filter Summary")
    if not filtered_df.empty:
        st.sidebar.markdown(
            f"👁️ Showing {len(filtered_df)} workout entries")
        if "workout" in st.session_state.filters and st.session_state.filters["workout"]:
            workouts = ", ".join(st.session_state.filters["workout"])
            st.sidebar.markdown(f"🏋️ **Workouts**: {workouts}")
        if "workout_type" in st.session_state.filters and st.session_state.filters["workout_type"]:
            workout_types = ", ".join(
                st.session_state.filters["workout_type"])
            st.sidebar.markdown(
                f"💪 **Workout Types**: {workout_types}")
        if "muscle_type" in st.session_state.filters and st.session_state.filters["muscle_type"]:
            muscle_types = ", ".join(
                st.session_state.filters["muscle_type"])
            st.sidebar.markdown(
                f"🔬 **Muscle Groups**: {muscle_types}")
    else:
        st.sidebar.markdown(f"⚠️ No workouts match your filters")

    # Create tabs
    tabs = st.tabs(["Dashboard", "Workout Analytics", "Advanced Analytics",
                   "Future Planning", "Raw Data", "Reports"])

    with tabs[0]:
        dashboard_tab(filtered_df)
    with tabs[1]:
        # KPIs
        key_performance_indicators(filtered_df)
        # Summary statistics
        summary_statistics(filtered_df)
        # Visualizations
        visualizations(filtered_df)
        # Personal Bests
        personal_bests(filtered_df)
    with tabs[2]:
        advanced_analytics_tab(filtered_df)
    with tabs[3]:
        future_planning_tab(filtered_df)
    with tabs[4]:
        raw_data_tab(filtered_df)
    with tabs[5]:
        download_reports_tab(filtered_df)


if __name__ == "__main__":
    main()
