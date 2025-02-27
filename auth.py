import streamlit as st
import hmac
import os


def check_password():
    """Returns True if the user entered the correct password."""
    # Username and password
    CORRECT_USERNAME = "ben"
    CORRECT_PASSWORD = "admin"

    # Initialize session state for auth
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If already authenticated, return True
    if st.session_state.authenticated:
        return True

    # Show login form in a centered container with styling
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <style>
        .auth-container {
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 20px;
            background-color: #f9f9f9;
            margin-top: 50px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        <div class="auth-container">
        <h2 style="text-align: center;">💪 Workout Tracker Login</h2>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", key="username_input")
        password = st.text_input(
            "Password", type="password", key="password_input")
        login_button = st.button(
            "Login", key="login_button", use_container_width=True)

    # Check credentials when button is pressed
    if login_button:
        if username.lower() == CORRECT_USERNAME.lower() and password == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            with col2:
                st.error("Incorrect username or password")

    # Important: Block everything else if not authenticated
    if not st.session_state.authenticated:
        # Hide all other elements
        st.stop()

    return st.session_state.authenticated
