#!/usr/bin/env python3
import os
import sys
import subprocess
import streamlit.web.cli as stcli

# Get the directory where the executable is located
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_dir = os.path.dirname(sys.executable)
else:
    # Running as a script
    base_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the base directory
os.chdir(base_dir)

# Run the Streamlit app
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "main.py",
                "--browser.serverAddress=localhost", "--server.headless=true"]
    sys.exit(stcli.main())
