# launch.py
import os
import webbrowser
import time
import subprocess


def main():
    # Optionally choose a port, or let Streamlit pick one
    port = 8501

    # Start streamlit as a subprocess
    # If you have a specific port in mind, do: f"streamlit run main.py --server.port={port}"
    process = subprocess.Popen(
        ["streamlit", "run", "main.py", f"--server.port={port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a moment for the server to start
    time.sleep(2)

    # Open the local URL in the default browser
    webbrowser.open(f"http://localhost:{port}")

    # Wait for the process to complete (or keep running until user stops)
    process.wait()


if __name__ == "__main__":
    main()
