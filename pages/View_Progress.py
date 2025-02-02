import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.optimize import curve_fit
from prophet import Prophet
from io import BytesIO
import base64
from db_utils import create_connection
import datetime
import os  # Added import for handling file modification time

# Enable wide mode for better layout
st.set_page_config(layout="wide")


def main():
    st.header("📊 View Progress Dashboard - Enhanced Analytics")

    # Initialize tabs
    tabs = st.tabs([
        "📋 Summary Statistics",
        "📈 Visualizations",
        "🏅 Personal Bests",
        "🔬 Advanced Analytics & Modeling",
        "📥 Download Reports"
    ])

    # Load data
    df_workouts = load_workouts()

    if df_workouts.empty:
        st.warning("⚠️ No workouts logged yet. Please add or upload data.")
        return

    # Calculate Volume
    df_workouts["volume"] = df_workouts["weight"] * \
        df_workouts["reps"] * df_workouts["sets"]

    # Add multiple 1RM formulas for later use
    df_workouts["Epley_1RM"] = df_workouts.apply(
        lambda row: epley_1rm(row["weight"], row["reps"]), axis=1
    )
    df_workouts["Brzycki_1RM"] = df_workouts.apply(
        lambda row: brzycki_1rm(row["weight"], row["reps"]), axis=1
    )
    df_workouts["Lombardi_1RM"] = df_workouts.apply(
        lambda row: lombardi_1rm(row["weight"], row["reps"]), axis=1
    )

    # Tab 1: Summary Statistics
    with tabs[0]:
        summary_statistics(df_workouts)

    # Tab 2: Visualizations
    with tabs[1]:
        visualizations(df_workouts)

    # Tab 3: Personal Bests
    with tabs[2]:
        personal_bests(df_workouts)

    # Tab 4: Advanced Analytics & Modeling
    with tabs[3]:
        advanced_analytics(df_workouts)

    # Tab 5: Download Reports
    with tabs[4]:
        download_reports(df_workouts)


def load_workouts():
    """Load workouts data from the database."""
    @st.cache_data
    def get_workouts(last_update):
        with create_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM workouts", conn)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    db_path = "workouts.db"  # Update this path if your database is located elsewhere
    try:
        last_update = os.path.getmtime(db_path)
    except Exception as e:
        st.error(f"❌ Unable to access the database file: {e}")
        last_update = None

    if last_update is not None:
        return get_workouts(last_update)
    else:
        return pd.DataFrame()  # Return empty DataFrame if database access fails


# ------------------------------------------------
# 1RM Formulas
# ------------------------------------------------
def epley_1rm(weight, reps):
    """Epley formula: 1RM = weight * (1 + reps/30)."""
    if reps > 0:
        return weight * (1 + (reps / 30))
    return None


def brzycki_1rm(weight, reps):
    """Brzycki formula: 1RM = weight * (36 / (37 - reps)), valid for reps <= ~10-12."""
    if 0 < reps < 37:  # to avoid division by zero or negative
        return weight * (36 / (37 - reps))
    return None


def lombardi_1rm(weight, reps):
    """Lombardi formula: 1RM = weight * (reps ^ 0.10)."""
    if reps > 0:
        return weight * (reps ** 0.10)
    return None


# ------------------------------------------------
# Summary Statistics
# ------------------------------------------------
def summary_statistics(df):
    """Display summary statistics."""
    st.subheader("🔍 Summary Statistics")

    with st.expander("ℹ️ About This Section", expanded=False):
        st.write("""
        This section provides high-level metrics:
        - **Total Sets**: Sum of all sets performed per workout.
        - **Average Weight**: Mean weight lifted per workout.
        - **Total Volume**: Total volume calculated as weight × reps × sets.
        """)

    # Calculate summary metrics
    total_sets = df.groupby("workout")["sets"].sum(
    ).reset_index().rename(columns={"sets": "Total Sets"})
    avg_weight = df.groupby("workout")["weight"].mean().reset_index().rename(
        columns={"weight": "Avg Weight (lbs)"})
    total_volume = df.groupby("workout")["volume"].sum(
    ).reset_index().rename(columns={"volume": "Total Volume"})

    # Display in columns
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


# ------------------------------------------------
# Visualization
# ------------------------------------------------
def visualizations(df):
    """Display various visualizations with togglable explanations."""
    st.subheader("📈 Visualizations")

    with st.expander("ℹ️ About These Charts", expanded=False):
        st.markdown("""
        These charts provide visual insights into your workout data:
        - **Total Sets by Workout**: Total number of sets performed for each workout.
        - **Average Weight by Workout**: Mean weight lifted in each workout.
        - **Total Volume by Workout**: Total volume lifted per workout.
        - **Volume Over Time**: Trend of your total daily volume.
        """)

    # Aggregate summaries
    sets_summary = df.groupby("workout")["sets"].sum(
    ).reset_index().rename(columns={"sets": "Total Sets"})
    weight_summary = df.groupby("workout")["weight"].mean(
    ).reset_index().rename(columns={"weight": "Avg Weight (lbs)"})
    volume_summary = df.groupby("workout")["volume"].sum(
    ).reset_index().rename(columns={"volume": "Total Volume"})

    # Layout for visualizations
    col1, col2 = st.columns(2)

    with col1:
        fig_sets = px.bar(
            sets_summary,
            x="workout",
            y="Total Sets",
            title="Total Sets by Workout",
            labels={"workout": "Workout", "Total Sets": "Sets"},
            text_auto=True
        )
        st.plotly_chart(fig_sets, use_container_width=True)

        fig_weight = px.bar(
            weight_summary,
            x="workout",
            y="Avg Weight (lbs)",
            title="Average Weight by Workout",
            labels={"workout": "Workout",
                    "Avg Weight (lbs)": "Average Weight (lbs)"},
            text_auto=True
        )
        st.plotly_chart(fig_weight, use_container_width=True)

    with col2:
        fig_volume = px.bar(
            volume_summary,
            x="workout",
            y="Total Volume",
            title="Total Volume by Workout",
            labels={"workout": "Workout",
                    "Total Volume": "Volume (lbs x reps x sets)"},
            text_auto=True
        )
        st.plotly_chart(fig_volume, use_container_width=True)

    # Additional Visualization: Volume Over Time
    st.write("**Volume Over Time**")
    by_date = df.groupby("date")["volume"].sum(
    ).reset_index().sort_values("date")
    fig_line = px.line(
        by_date,
        x="date",
        y="volume",
        title="Total Volume Over Time",
        labels={"volume": "Total Volume (lbs x reps x sets)", "date": "Date"},
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)


# ------------------------------------------------
# Personal Bests
# ------------------------------------------------
def personal_bests(df):
    """Display personal bests (max weight) and 1RM estimates."""
    st.subheader("🏅 Personal Bests")

    with st.expander("ℹ️ About These Results", expanded=False):
        st.markdown("""
        - **Max Weight by Workout**: The heaviest weight you've lifted for each workout.
        - **Estimated 1RM**: Various formulas (Epley, Brzycki, Lombardi) used to estimate your one-rep max.
        """)

    # Personal best (max weight) per workout
    pb = df.groupby("workout")["weight"].max().reset_index().rename(
        columns={"weight": "Max Weight (lbs)"})
    st.write("**Max Weight by Workout**")
    st.dataframe(pb, use_container_width=True)

    st.write("**Estimated 1RM (Epley, Brzycki, Lombardi) for Each Recorded Set**")
    st.dataframe(
        df[["date", "workout", "sets", "reps", "weight",
            "Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"]]
        .rename(columns={
            "date": "Date",
            "workout": "Workout",
            "sets": "Sets",
            "reps": "Reps",
            "weight": "Weight (lbs)"
        }),
        use_container_width=True
    )

    # 1RM Over Time - User Selection
    st.write("**Compare 1RM Formulas Over Time**")
    workouts = df["workout"].unique().tolist()
    selected_workout = st.selectbox(
        "Select a Workout for 1RM Comparison", workouts)

    df_selected = df[df["workout"] ==
                     selected_workout].copy().sort_values("date")

    if not df_selected.empty:
        fig_1rm = px.line(
            df_selected,
            x="date",
            y=["Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"],
            title=f"📈 Estimated 1RM Over Time - {selected_workout}",
            labels={
                "value": "Estimated 1RM (lbs)", "date": "Date", "variable": "Formula"},
            markers=True
        )
        st.plotly_chart(fig_1rm, use_container_width=True)
    else:
        st.info("ℹ️ No data available for the selected workout.")


# ------------------------------------------------
# Advanced Analytics & Modeling
# ------------------------------------------------
def advanced_analytics(df):
    """Provide advanced analytics and interactive filtering, including calculus & differential equations."""
    st.subheader("🔬 Advanced Analytics & Modeling")

    # ------------------------------
    # Filter Data
    # ------------------------------
    with st.expander("🔧 Filter Options"):
        col1, col2 = st.columns(2)
        with col1:
            min_date = st.date_input("Start Date", df["date"].min())
        with col2:
            max_date = st.date_input("End Date", df["date"].max())

        # Workout selection
        all_workouts = df["workout"].unique().tolist()
        selected_workouts = st.multiselect(
            "Select Workouts", options=all_workouts, default=all_workouts)

        # Apply filters
        mask = (
            (df["date"] >= pd.to_datetime(min_date)) &
            (df["date"] <= pd.to_datetime(max_date)) &
            (df["workout"].isin(selected_workouts))
        )
        filtered_df = df.loc[mask].copy()

    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # Initialize a container for all advanced analytics
    advanced_container = st.container()

    with advanced_container:
        # ---------------------------------------------------
        # 1) Correlation Matrix & Heatmap
        # ---------------------------------------------------
        with st.expander("1) Correlation Analysis"):
            st.write("""
            Analyze the relationships between different workout variables.
            A **correlation coefficient** close to +1 indicates a strong positive relationship, 
            while a coefficient close to -1 indicates a strong negative relationship.
            """)
            corr_cols = ["sets", "reps", "weight", "volume",
                         "Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"]
            existing_corr_cols = [
                col for col in corr_cols if col in filtered_df.columns and filtered_df[col].notnull().any()]

            if len(existing_corr_cols) > 1:
                corr_matrix = filtered_df[existing_corr_cols].corr()
                st.write("**Correlation Matrix**")
                st.dataframe(corr_matrix, use_container_width=True)

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
                st.info("Not enough numeric columns for correlation analysis.")

        # ---------------------------------------------------
        # 2) Scatter + Regression: Weight vs. Volume
        # ---------------------------------------------------
        with st.expander("2) Weight vs. Volume with Trendline"):
            st.write("""
            Explore the relationship between the **Weight** you lift and the **Volume** you achieve.
            A trendline helps visualize the direction and strength of this relationship.
            """)
            corr_val = filtered_df["weight"].corr(filtered_df["volume"])
            st.write(
                f"**Correlation Coefficient (Weight vs. Volume):** {corr_val:.2f}")

            fig_scatter = px.scatter(
                filtered_df,
                x="weight",
                y="volume",
                trendline="ols",
                title="Weight vs. Volume (with OLS Trendline)",
                labels={
                    "weight": "Weight (lbs)", "volume": "Volume (lbs x reps x sets)"},
                hover_data=["workout", "date"]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ---------------------------------------------------
        # 3) Workout Frequency Over Time
        # ---------------------------------------------------
        with st.expander("3) Workout Frequency Over Time"):
            st.write("""
            Visualize how often each workout is performed over time.  
            This helps identify patterns, such as consistently performing certain workouts or varying your routine.
            """)
            freq = filtered_df.groupby(
                ["date", "workout"]).size().reset_index(name='count')
            fig_freq = px.bar(
                freq,
                x="date",
                y="count",
                color="workout",
                title="📊 Workout Frequency Over Time",
                labels={"count": "Number of Sessions",
                        "date": "Date", "workout": "Workout"},
                barmode="group"
            )
            st.plotly_chart(fig_freq, use_container_width=True)

        # ---------------------------------------------------
        # 4) Volume Distribution by Workout
        # ---------------------------------------------------
        with st.expander("4) Volume Distribution by Workout"):
            st.write("""
            A box plot illustrating the distribution of **Volume** for each workout.  
            It highlights the median, quartiles, and potential outliers.
            """)
            fig_volume_dist = px.box(
                filtered_df,
                x="workout",
                y="volume",
                title="📦 Volume Distribution by Workout",
                labels={
                    "volume": "Volume (lbs x reps x sets)", "workout": "Workout"}
            )
            st.plotly_chart(fig_volume_dist, use_container_width=True)

        # ---------------------------------------------------
        # 5) Top 5 Workouts by Volume
        # ---------------------------------------------------
        with st.expander("5) Top 5 Workouts by Total Volume"):
            st.write("""
            Identify which workouts contribute the most to your total volume.  
            Focus on these for potential progression or optimization.
            """)
            top5 = (
                filtered_df.groupby("workout")["volume"]
                .sum()
                .reset_index()
                .sort_values("volume", ascending=False)
                .head(5)
            )
            st.dataframe(
                top5.rename(columns={"workout": "Workout",
                            "volume": "Total Volume"}),
                use_container_width=True
            )

        # ---------------------------------------------------
        # 6) Moving Average of Volume (7-Day)
        # ---------------------------------------------------
        with st.expander("6) 7-Day Moving Average of Total Volume"):
            st.write("""
            The **7-Day Moving Average** smooths out daily fluctuations to reveal longer-term trends in your total volume.
            """)
            try:
                moving_avg = filtered_df.groupby(
                    "date")["volume"].sum().reset_index().sort_values("date")
                moving_avg["7d_MA"] = moving_avg["volume"].rolling(
                    window=7).mean()

                fig_moving_avg = px.line(
                    moving_avg,
                    x="date",
                    y=["volume", "7d_MA"],
                    title="📈 Total Volume and 7-Day Moving Average Over Time",
                    labels={"date": "Date", "value": "Total Volume (lbs x reps x sets)",
                            "variable": "Metric"},
                    markers=True
                )
                st.plotly_chart(fig_moving_avg, use_container_width=True)
            except Exception as e:
                st.error(
                    f"❌ An error occurred while calculating moving average: {e}")

        # ---------------------------------------------------
        # 7) Day-of-Week Analysis
        # ---------------------------------------------------
        with st.expander("7) Day-of-Week Analysis"):
            st.write("""
            Analyze on which days of the week you train the most.  
            This can help in scheduling and ensuring balanced training across the week.
            """)
            filtered_df["day_of_week"] = filtered_df["date"].dt.day_name()
            day_volume = (
                filtered_df.groupby("day_of_week")["volume"]
                .sum()
                .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            )
            fig_day = px.bar(
                day_volume.reset_index(),
                x="day_of_week",
                y="volume",
                title="Total Volume by Day of Week",
                labels={"day_of_week": "Day of Week", "volume": "Total Volume"}
            )
            st.plotly_chart(fig_day, use_container_width=True)

        # ---------------------------------------------------
        # 8) Derivatives & Calculus
        # ---------------------------------------------------
        with st.expander("8) Calculus: Derivatives of Volume Over Time"):
            st.write("""
            Using finite differences to approximate the **1st derivative** (rate of change) and **2nd derivative** (acceleration) of your total volume over time.
            - **1st derivative (dV/dt)**: Indicates whether your volume is increasing or decreasing.
            - **2nd derivative (d²V/dt²)**: Shows the acceleration of volume changes.
            """)

            # For derivative approximation, we need daily volumes in chronological order
            vol_daily = filtered_df.groupby(
                "date")["volume"].sum().reset_index().sort_values("date")
            vol_daily["volume_ma"] = vol_daily["volume"].rolling(
                window=3, center=True, min_periods=1).mean()

            # Approximate 1st derivative via finite difference
            vol_daily["dV/dt"] = vol_daily["volume_ma"].diff()

            # Approximate 2nd derivative
            vol_daily["d2V/dt2"] = vol_daily["dV/dt"].diff()

            # Plot 1st derivative and 2nd derivative
            fig_deriv = go.Figure()
            fig_deriv.add_trace(go.Scatter(
                x=vol_daily["date"], y=vol_daily["dV/dt"], mode='lines+markers',
                name='dV/dt (1st derivative)'
            ))
            fig_deriv.add_trace(go.Scatter(
                x=vol_daily["date"], y=vol_daily["d2V/dt2"], mode='lines+markers',
                name='d²V/dt² (2nd derivative)'
            ))
            fig_deriv.update_layout(
                title="Rate of Change (1st & 2nd Derivative) of Volume Over Time",
                xaxis_title="Date",
                yaxis_title="Approx. Derivative of Volume"
            )
            st.plotly_chart(fig_deriv, use_container_width=True)

            st.info("""
            - **Positive 1st derivative**: Volume is increasing day-to-day.
            - **Negative 1st derivative**: Volume is decreasing day-to-day.
            - **Positive 2nd derivative**: The rate of increase is accelerating.
            - **Negative 2nd derivative**: The rate of increase is decelerating or rate of decrease is accelerating.
            """)

        # ---------------------------------------------------
        # 9) Differential Equation Modeling (Exponential & Logistic)
        # ---------------------------------------------------
        with st.expander("9) Differential Equation Modeling: Exponential & Logistic Fits"):
            st.write("""
            Fit your total volume over time to **Exponential** and **Logistic** growth models to identify underlying patterns.
            - **Exponential Model**: Assumes continuous growth.
            - **Logistic Model**: Accounts for growth saturation.
            """)

            daily_df = filtered_df.groupby(
                "date")["volume"].sum().reset_index().sort_values("date")
            if len(daily_df) < 5:
                st.warning(
                    "Not enough data points to fit advanced models (need at least 5).")
            else:
                # Convert date to ordinal
                xdata = daily_df["date"].apply(
                    lambda x: x.toordinal()).values.astype(float)
                ydata = daily_df["volume"].values.astype(float)

                # Exponential Fit
                def exponential_model(x, a, b, c):
                    return a * np.exp(b * (x - x.min())) + c

                # Logistic Fit
                def logistic_model(x, L, k, x0):
                    return L / (1 + np.exp(-k * (x - x0)))

                # Try Exponential
                try:
                    initial_exp = [max(ydata), 0.01, min(ydata)]
                    popt_exp, pcov_exp = curve_fit(
                        exponential_model, xdata, ydata, p0=initial_exp, maxfev=10000)
                    a_fit, b_fit, c_fit = popt_exp
                    y_exp_fit = exponential_model(xdata, a_fit, b_fit, c_fit)
                    # Calculate R^2
                    residuals_exp = ydata - y_exp_fit
                    ss_res_exp = np.sum(residuals_exp**2)
                    ss_tot_exp = np.sum((ydata - np.mean(ydata))**2)
                    r2_exp = 1 - (ss_res_exp / ss_tot_exp)
                except Exception as e:
                    popt_exp, r2_exp = None, None

                # Try Logistic
                try:
                    initial_log = [max(ydata), 0.1, np.median(xdata)]
                    popt_log, pcov_log = curve_fit(
                        logistic_model, xdata, ydata, p0=initial_log, maxfev=10000)
                    L_fit, k_fit, x0_fit = popt_log
                    y_log_fit = logistic_model(xdata, L_fit, k_fit, x0_fit)
                    # Calculate R^2
                    residuals_log = ydata - y_log_fit
                    ss_res_log = np.sum(residuals_log**2)
                    ss_tot_log = np.sum((ydata - np.mean(ydata))**2)
                    r2_log = 1 - (ss_res_log / ss_tot_log)
                except Exception as e:
                    popt_log, r2_log = None, None

                # Plot results
                fig_model = go.Figure()
                # Actual data
                fig_model.add_trace(go.Scatter(
                    x=daily_df["date"], y=daily_df["volume"], mode='markers',
                    name='Actual Volume'
                ))

                # Exponential best fit
                if popt_exp is not None:
                    fig_model.add_trace(go.Scatter(
                        x=daily_df["date"], y=y_exp_fit, mode='lines',
                        name=f'Exponential Fit (R²={r2_exp:.2f})'
                    ))

                # Logistic best fit
                if popt_log is not None:
                    fig_model.add_trace(go.Scatter(
                        x=daily_df["date"], y=y_log_fit, mode='lines',
                        name=f'Logistic Fit (R²={r2_log:.2f})'
                    ))

                fig_model.update_layout(
                    title="Exponential & Logistic Modeling of Volume Over Time",
                    xaxis_title="Date",
                    yaxis_title="Total Daily Volume"
                )
                st.plotly_chart(fig_model, use_container_width=True)

                if popt_exp is not None:
                    st.write(
                        f"**Exponential Model Parameters**: a={popt_exp[0]:.2f}, b={popt_exp[1]:.4f}, c={popt_exp[2]:.2f}, R²={r2_exp:.2f}")
                if popt_log is not None:
                    st.write(
                        f"**Logistic Model Parameters**: L={popt_log[0]:.2f}, k={popt_log[1]:.4f}, x₀={popt_log[2]:.2f}, R²={r2_log:.2f}")

        # ---------------------------------------------------
        # 10) Time Series Forecasting
        # ---------------------------------------------------
        with st.expander("10) Time Series Forecasting with Prophet"):
            st.write("""
            Forecast your total daily volume for the next 30 days using **Facebook Prophet**.  
            This helps in planning and understanding future training loads.
            """)

            daily_df = filtered_df.groupby(
                "date")["volume"].sum().reset_index().sort_values("date")

            if len(daily_df) < 10:
                st.warning(
                    "Not enough data points to perform forecasting (need at least 10).")
            else:
                # Prepare data for Prophet
                prophet_df = daily_df.rename(
                    columns={"date": "ds", "volume": "y"})
                m = Prophet(daily_seasonality=True)
                try:
                    m.fit(prophet_df)
                    future = m.make_future_dataframe(periods=30)
                    forecast = m.predict(future)

                    # Plot forecast
                    fig_prophet = m.plot(forecast)
                    st.pyplot(fig_prophet, use_container_width=True)

                    # Plot forecast components
                    fig_components = m.plot_components(forecast)
                    st.pyplot(fig_components, use_container_width=True)

                    # Display forecast metrics
                    st.write("**Forecast Summary**")
                    forecast_summary = forecast[[
                        'ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
                    st.dataframe(forecast_summary.rename(columns={
                        "ds": "Date",
                        "yhat": "Forecasted Volume",
                        "yhat_lower": "Lower Confidence Interval",
                        "yhat_upper": "Upper Confidence Interval"
                    }), use_container_width=True)
                except Exception as e:
                    st.error(
                        f"❌ An error occurred during forecasting: {e}")


# ------------------------------------------------
# Download Reports
# ------------------------------------------------
def download_reports(df):
    """Allow users to download analytics reports and charts."""
    st.subheader("📥 Download Reports")

    with st.expander("ℹ️ About Download Options", expanded=False):
        st.markdown("""
        - **Raw Data**: Download your workout data as a CSV file.
        - **Analytics Report**: Download a comprehensive Excel report containing various analytics and models.
        """)

    # Prepare data for download
    csv = df.to_csv(index=False).encode('utf-8')

    # Download button for raw data
    st.download_button(
        label="📄 Download Raw Data as CSV",
        data=csv,
        file_name='workout_data.csv',
        mime='text/csv',
    )

    # Generate reports in Excel with multiple sheets
    if st.button("📊 Download Analytics Report as Excel"):
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Summary Statistics
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

                # Full Data
                df.to_excel(writer, sheet_name="FullData", index=False)

                # Correlation
                corr_cols = ["sets", "reps", "weight", "volume",
                             "Epley_1RM", "Brzycki_1RM", "Lombardi_1RM"]
                existing_corr_cols = [
                    col for col in corr_cols if col in df.columns and df[col].notnull().any()]
                if len(existing_corr_cols) > 1:
                    correlation = df[existing_corr_cols].corr()
                    correlation.to_excel(writer, sheet_name="Correlation")

                # PCA Results
                # Assuming PCA was run and stored, else skip
                # Similarly for other analyses

                # Additional sheets can be added here

            processed_data = output.getvalue()
            b64 = base64.b64encode(processed_data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="analytics_report.xlsx">📥 Click here to download the Excel report</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ An error occurred while generating the report: {e}")


# ------------------------------------------------
# Polynomial Regression Class
# ------------------------------------------------
class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = None
        self.model = LinearRegression()

    def fit(self, X, y):
        self.poly_features = self._polynomial_features(X)
        self.model.fit(self.poly_features, y)

    def predict(self, X):
        poly_X = self._polynomial_features(X)
        return self.model.predict(poly_X)

    def _polynomial_features(self, X):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=self.degree)
        return poly.fit_transform(X)


if __name__ == "__main__":
    main()
