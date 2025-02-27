import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Shared styling constants
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb00',
    'background': '#f8f9fa',
    'border': '#dee2e6'
}

FONTS = {
    'header': 'sans-serif',
    'body': 'sans-serif'
}

MEASUREMENT_UNITS = {
    'imperial': {
        'weight': 'lbs',
        'height': 'in',
        'length': 'in',
    },
    'metric': {
        'weight': 'kg',
        'height': 'cm',
        'length': 'cm',
    }
}

# Conversion factors
CONVERSION_FACTORS = {
    'kg_to_lbs': 2.20462,
    'lbs_to_kg': 0.453592,
    'cm_to_in': 0.393701,
    'in_to_cm': 2.54
}

# Shared utility functions


def format_metric(value, precision=2):
    """Format numeric values for display"""
    if pd.isna(value) or value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.{precision}f}"


def format_date(date):
    """Format date for display."""
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date.strftime("%Y-%m-%d")


def format_percentage(value, precision=1):
    """Format percentage values for display."""
    if pd.isna(value) or value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:+.{precision}f}%"


def calculate_trend(current, previous):
    """Calculate trend percentage between two values"""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return None
    return ((current - previous) / previous) * 100


def get_date_range_options():
    """
    Get common date range options for filtering.

    Returns:
        dict: Dictionary of date range options with pandas datetime values
    """
    today = pd.to_datetime(datetime.now().date())
    return {
        '1W': today - pd.Timedelta(days=7),
        '1M': today - pd.Timedelta(days=30),
        '3M': today - pd.Timedelta(days=90),
        '6M': today - pd.Timedelta(days=180),
        '1Y': today - pd.Timedelta(days=365),
        'All': None
    }


def apply_date_filter(df, start_date, date_column='date'):
    """
    Filter DataFrame by date range.

    Args:
        df (pd.DataFrame): DataFrame to filter
        start_date (datetime.date or str): Start date for filtering
        date_column (str): Name of the date column to filter on

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if start_date is None:
        return df

    # Convert start_date to pandas datetime if it's a string or date object
    if isinstance(start_date, (str, type(datetime.date))):
        start_date = pd.to_datetime(start_date)

    # Ensure the DataFrame's date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    return df[df[date_column] >= start_date]


def bmi_category(bmi):
    """Get BMI category with color coding"""
    if bmi is None:
        return "Unknown", COLORS['warning']
    elif bmi < 18.5:
        return "Underweight", COLORS['warning']
    elif 18.5 <= bmi < 25:
        return "Normal weight", COLORS['success']
    elif 25 <= bmi < 30:
        return "Overweight", COLORS['warning']
    else:
        return "Obese", COLORS['danger']


def estimate_body_fat(weight, height, age, gender, neck=None, waist=None, hips=None):
    """
    Estimate body fat percentage using the US Navy method when measurements are available,
    otherwise falls back to the BMI method.
    """
    if weight is None or height is None:
        return None

    # If we have all the measurements needed for US Navy method
    if neck is not None and waist is not None and (gender.lower() != 'female' or hips is not None):
        # US Navy Method
        if gender.lower() == 'male':
            body_fat = 86.010 * \
                np.log10(waist - neck) - 70.041 * np.log10(height) + 36.76
        elif gender.lower() == 'female':
            if hips is not None:
                body_fat = 163.205 * \
                    np.log10(waist + hips - neck) - 97.684 * \
                    np.log10(height) - 78.387
            else:
                return None
        else:
            # For non-binary gender, take average of male and female calculations
            male_bf = 86.010 * np.log10(waist - neck) - \
                70.041 * np.log10(height) + 36.76
            if hips is not None:
                female_bf = 163.205 * \
                    np.log10(waist + hips - neck) - 97.684 * \
                    np.log10(height) - 78.387
                body_fat = (male_bf + female_bf) / 2
            else:
                body_fat = male_bf

        return max(0, min(body_fat, 100))  # Clamp between 0 and 100

    # If we don't have all measurements, use BMI method
    elif age is not None and gender is not None:
        bmi = calculate_bmi(weight, height)
        if bmi is None:
            return None

        # Deurenberg formula
        if gender.lower() == 'male':
            gender_factor = 1
        elif gender.lower() == 'female':
            gender_factor = 0
        else:
            gender_factor = 0.5

        body_fat = (1.20 * bmi) + (0.23 * age) - (10.8 * gender_factor) - 5.4
        return max(0, min(body_fat, 100))  # Clamp between 0 and 100

    return None


def calculate_bmi(weight, height):
    """Calculate BMI based on weight and height"""
    if weight is None or height is None:
        return None

    # Determine measurement system based on units
    if weight > 500:  # Assuming metric if weight > 500 lbs is unlikely
        # Metric system
        bmi = weight / ((height / 100) ** 2)  # height converted to meters
    else:
        # Imperial system
        bmi = (703 * weight) / (height ** 2)

    return bmi

# Shared UI components


def show_metric_card(title, value, delta=None, precision=2, prefix="", suffix=""):
    """Display a metric card with optional trend indicator"""
    formatted_value = f"{prefix}{format_metric(value, precision)}{suffix}"
    if delta is not None:
        st.metric(title, formatted_value, format_percentage(delta))
    else:
        st.metric(title, formatted_value)


def show_date_filter():
    """Show date range filter widget"""
    ranges = get_date_range_options()
    selected = st.selectbox("📅 Time Range", list(
        ranges.keys()), key="date_filter")
    return ranges[selected]


def show_loading_spinner():
    """Show a loading spinner"""
    return st.spinner("Loading...")


def show_error_message(message):
    """Show an error message."""
    st.error(f"❌ {message}")


def show_success_message(message):
    """Show a success message."""
    st.success(f"✅ {message}")


def show_info_message(message):
    """Show an info message."""
    st.info(f"ℹ️ {message}")


def show_warning_message(message):
    """Show a warning message."""
    st.warning(f"⚠️ {message}")

# Unit conversion functions


def convert_weight(value, from_unit='kg', to_unit='lbs'):
    """Convert weight between units."""
    if pd.isna(value):
        return None
    if from_unit == to_unit:
        return value
    if from_unit == 'kg' and to_unit == 'lbs':
        return value * CONVERSION_FACTORS['kg_to_lbs']
    if from_unit == 'lbs' and to_unit == 'kg':
        return value * CONVERSION_FACTORS['lbs_to_kg']
    raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")


def convert_length(value, from_unit='cm', to_unit='in'):
    """Convert length measurements between units."""
    if pd.isna(value):
        return None
    if from_unit == to_unit:
        return value
    if from_unit == 'cm' and to_unit == 'in':
        return value * CONVERSION_FACTORS['cm_to_in']
    if from_unit == 'in' and to_unit == 'cm':
        return value * CONVERSION_FACTORS['in_to_cm']
    raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")

# Data validation functions


def validate_weight(weight, unit='lbs'):
    """Validate weight input."""
    if pd.isna(weight):
        return False, "Weight is required"
    if unit == 'lbs' and (weight < 50 or weight > 1000):
        return False, "Weight must be between 50 and 1000 lbs"
    if unit == 'kg' and (weight < 23 or weight > 453):
        return False, "Weight must be between 23 and 453 kg"
    return True, None


def validate_height(height, unit='in'):
    """Validate height input."""
    if pd.isna(height):
        return True, None  # Height is optional
    if unit == 'in' and (height < 36 or height > 120):
        return False, "Height must be between 36 and 120 inches"
    if unit == 'cm' and (height < 91 or height > 305):
        return False, "Height must be between 91 and 305 cm"
    return True, None


def validate_measurement(value, name, unit='in'):
    """Validate body measurement input."""
    if pd.isna(value):
        return True, None  # Measurements are optional
    if unit == 'in' and (value < 0 or value > 100):
        return False, f"{name} must be between 0 and 100 inches"
    if unit == 'cm' and (value < 0 or value > 254):
        return False, f"{name} must be between 0 and 254 cm"
    return True, None
