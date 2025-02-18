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
    except Exception as e:
        st.error(f"❌ Unable to access the database file: {e}")
        last_update = None

    @st.cache_data
    def get_workouts(last_update):
        with create_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM workouts", conn)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    if last_update is not None:
        return get_workouts(last_update)
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
    st.sidebar.header("🔧 Global Filter Options")
    min_date = st.sidebar.date_input("Start Date", value=df["date"].min())
    max_date = st.sidebar.date_input("End Date", value=df["date"].max())

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

    if "workout_type" in df.columns:
        all_types = sorted(df["workout_type"].dropna().unique().tolist())
        selected_types = st.sidebar.multiselect(
            "Select Workout Types", options=all_types, default=all_types)
    else:
        selected_types = []

    if "muscle_type" in df.columns:
        all_muscles = sorted(df["muscle_type"].dropna().unique().tolist())
        selected_muscles = st.sidebar.multiselect(
            "Select Muscle Types", options=all_muscles, default=all_muscles)
    else:
        selected_muscles = []

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
    total_sessions = df['date'].nunique()  # Count unique dates instead of rows
    total_volume = df["volume"].sum()
    avg_volume = df["volume"].mean()
    max_volume = df["volume"].max()
    freq_workout = df["workout"].value_counts()
    most_freq_workout = freq_workout.idxmax()
    freq_count = freq_workout.max()
    max_weight = df["weight"].max()
    avg_sets = df["sets"].mean()
    avg_reps = df["reps"].mean()
    avg_1rm = df["Epley_1RM"].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", f"{total_sessions}")
        st.metric("Total Volume", f"{total_volume:.0f} lbs")
        st.metric("Avg. Volume/Session", f"{avg_volume:.0f} lbs")
    with col2:
        st.metric("Max Volume (Session)", f"{max_volume:.0f} lbs")
        st.metric("Most Frequent Workout",
                  f"{most_freq_workout} ({freq_count} sets)")  # Changed to clarify these are sets
        st.metric("Max Weight Lifted", f"{max_weight:.0f} lbs")
    with col3:
        st.metric("Avg. Sets", f"{avg_sets:.1f}")
        st.metric("Avg. Reps", f"{avg_reps:.1f}")
        st.metric("Avg. Epley 1RM", f"{avg_1rm:.0f} lbs")

# =======================
# SUMMARY STATISTICS
# =======================


def summary_statistics(df):
    st.subheader("🔍 Summary Statistics")
    total_sets = df.groupby("workout")["sets"].sum(
    ).reset_index().rename(columns={"sets": "Total Sets"})
    avg_weight = df.groupby("workout")["weight"].mean().reset_index().rename(
        columns={"weight": "Avg Weight (lbs)"})
    total_volume = df.groupby("workout")["volume"].sum(
    ).reset_index().rename(columns={"volume": "Total Volume"})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Total Sets by Workout**")
        st.dataframe(total_sets, use_container_width=True)
    with col2:
        st.write("**Average Weight by Workout**")
        st.dataframe(avg_weight, use_container_width=True)
    with col3:
        st.write("**Total Volume by Workout**")
        st.dataframe(total_volume, use_container_width=True)

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

# =======================
# FUTURE PLANNING TAB
# =======================


def future_planning_tab(df):
    st.header("Future Workout Planning")

    st.subheader("Weekly Volume Trend")
    df_weekly = df.groupby(pd.Grouper(key="date", freq="W")).agg(
        {"volume": "sum"}).reset_index()
    fig_weekly = px.bar(df_weekly, x="date", y="volume", title="Weekly Total Volume",
                        labels={"volume": "Volume", "date": "Week Ending"})
    st.plotly_chart(fig_weekly, use_container_width=True)

    st.subheader("Volume by Muscle Group")
    if "muscle_type" in df.columns:
        muscle_volume = df.groupby("muscle_type")["volume"].sum().reset_index()
        fig_muscle = px.pie(muscle_volume, names="muscle_type", values="volume",
                            title="Volume Distribution by Muscle Group")
        st.plotly_chart(fig_muscle, use_container_width=True)
    else:
        st.info("Muscle type data not available.")

    st.subheader("Workout Calendar Heatmap")
    df_calendar = df.copy()
    df_calendar["week"] = df_calendar["date"].dt.strftime("%Y-%U")
    df_calendar["weekday"] = df_calendar["date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
    df_calendar["weekday"] = pd.Categorical(
        df_calendar["weekday"], categories=weekday_order, ordered=True)
    pivot = df_calendar.pivot_table(
        index="weekday", columns="week", values="volume", aggfunc="sum", fill_value=0)
    fig_calendar = px.imshow(pivot, labels=dict(x="Week", y="Day of Week", color="Volume"),
                             x=pivot.columns, y=pivot.index, aspect="auto",
                             title="Calendar Heatmap of Volume")
    st.plotly_chart(fig_calendar, use_container_width=True)

    st.subheader("Days Since Last Workout by Muscle Group")
    if "muscle_type" in df.columns:
        today = pd.Timestamp.today().normalize()
        last_dates = df.groupby("muscle_type")["date"].max().reset_index()
        last_dates["days_since"] = (today - last_dates["date"]).dt.days
        fig_days = px.bar(last_dates, x="muscle_type", y="days_since",
                          title="Days Since Last Workout by Muscle Group",
                          labels={"muscle_type": "Muscle Group", "days_since": "Days Since Last Session"})
        st.plotly_chart(fig_days, use_container_width=True)
    else:
        st.info("Muscle type data not available.")

    st.subheader("Forecast Future Weekly Volume")
    df_weekly = df.groupby(pd.Grouper(key="date", freq="W")).agg(
        {"volume": "sum"}).reset_index()
    if len(df_weekly) >= 10:
        forecast_df = df_weekly.rename(columns={"date": "ds", "volume": "y"})
        m = Prophet(weekly_seasonality=True)
        try:
            m.fit(forecast_df)
            future = m.make_future_dataframe(periods=4, freq="W")
            forecast = m.predict(future)
            fig_forecast = m.plot(forecast)
            st.pyplot(fig_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Error forecasting weekly volume: {e}")
    else:
        st.info("Not enough data points for forecasting future weekly volume.")

    st.subheader("Session Frequency Over Time")
    sessions = df.groupby(pd.Grouper(key="date", freq="W")
                          ).size().reset_index(name="sessions")
    fig_sessions = px.line(sessions, x="date", y="sessions",
                           title="Weekly Session Frequency",
                           labels={"sessions": "Number of Sessions", "date": "Week Ending"})
    st.plotly_chart(fig_sessions, use_container_width=True)

    st.subheader("Rolling Average of Epley 1RM")
    df_sorted = df.sort_values("date")
    df_sorted["rolling_1RM"] = df_sorted["Epley_1RM"].rolling(
        window=7, min_periods=1).mean()
    fig_rolling_1rm = px.line(df_sorted, x="date", y="rolling_1RM",
                              title="7-Day Rolling Average of Epley 1RM",
                              labels={"rolling_1RM": "Rolling Average Epley 1RM", "date": "Date"})
    st.plotly_chart(fig_rolling_1rm, use_container_width=True)

    st.subheader("Monthly Volume Trend")
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
    monthly_volume = df_monthly.groupby("month")["volume"].sum().reset_index()
    fig_monthly = px.bar(monthly_volume, x="month", y="volume",
                         title="Monthly Total Volume",
                         labels={"volume": "Volume", "month": "Month"})
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.subheader("1RM Improvement Over Time")
    improvement = []
    for workout in df["workout"].unique():
        df_work = df[df["workout"] == workout].sort_values("date")
        if len(df_work) > 1:
            first = df_work.iloc[0]["Epley_1RM"]
            last = df_work.iloc[-1]["Epley_1RM"]
            pct_improve = ((last - first) / first) * 100 if first != 0 else 0
            improvement.append(
                {"workout": workout, "pct_improve": pct_improve})
    if improvement:
        df_improve = pd.DataFrame(improvement)
        fig_improve_1rm = px.bar(df_improve, x="workout", y="pct_improve",
                                 title="Percentage Improvement in Epley 1RM (First vs. Latest)",
                                 labels={"pct_improve": "Improvement (%)", "workout": "Workout"})
        st.plotly_chart(fig_improve_1rm, use_container_width=True)
    else:
        st.info("Not enough data for 1RM improvement analysis.")

# =======================
# DASHBOARD TAB
# =======================


def dashboard_tab(df):
    st.header("Dashboard")
    key_performance_indicators(df)
    st.markdown("---")
    st.subheader("Summary Statistics")
    summary_statistics(df)
    st.markdown("---")
    st.subheader("Visualizations")
    visualizations(df)
    st.markdown("---")
    st.subheader("Personal Bests")
    personal_bests(df)

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
    st.title("📊 Workout Progress Dashboard")
    df_workouts = load_workouts()
    if df_workouts.empty:
        st.warning("⚠️ No workouts logged yet. Please add or upload data.")
        return

    # Preprocessing
    df_workouts["date"] = pd.to_datetime(df_workouts["date"], errors="coerce")
    df_workouts["volume"] = df_workouts["weight"] * \
        df_workouts["sets"] * df_workouts["reps"]
    df_workouts["Epley_1RM"] = df_workouts.apply(
        lambda row: epley_1rm(row["weight"], row["reps"]), axis=1)
    df_workouts["Brzycki_1RM"] = df_workouts.apply(
        lambda row: brzycki_1rm(row["weight"], row["reps"]), axis=1)
    df_workouts["Lombardi_1RM"] = df_workouts.apply(
        lambda row: lombardi_1rm(row["weight"], row["reps"]), axis=1)

    filtered_df = filter_data(df_workouts)
    st.sidebar.markdown("### Filter Summary")
    if not filtered_df.empty:
        st.sidebar.write(
            f"**Date Range:** {filtered_df['date'].min().date()} to {filtered_df['date'].max().date()}")
        st.sidebar.write(
            f"**Workouts:** {', '.join(sorted(filtered_df['workout'].unique()))}")
        if "workout_type" in filtered_df.columns:
            st.sidebar.write(
                f"**Workout Types:** {', '.join(sorted(filtered_df['workout_type'].dropna().unique()))}")
        if "muscle_type" in filtered_df.columns:
            st.sidebar.write(
                f"**Muscle Types:** {', '.join(sorted(filtered_df['muscle_type'].dropna().unique()))}")
    else:
        st.sidebar.info("No data matches the current filters.")

    # Define Tabs
    tabs = st.tabs(["Dashboard", "Advanced Analytics",
                   "Future Planning", "Downloads", "Raw Data"])
    with tabs[0]:
        dashboard_tab(filtered_df)
    with tabs[1]:
        advanced_analytics_tab(filtered_df)
    with tabs[2]:
        future_planning_tab(filtered_df)
    with tabs[3]:
        download_reports_tab(filtered_df)
    with tabs[4]:
        raw_data_tab(filtered_df)


if __name__ == "__main__":
    main()
