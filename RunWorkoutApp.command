#!/bin/bash

# Cleanup function that will be called when the script exits
cleanup() {
    echo ""
    echo "🛑 Stopping Workout Tracker..."
    pkill -f "streamlit run main.py"
    echo "✅ Workout Tracker has been stopped."
    exit 0
}

# Register the cleanup function to be called on script exit
trap cleanup EXIT INT TERM

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to that directory
cd "$DIR"

# Function to generate a simple ASCII QR code
generate_qr_code() {
    local url="$1"
    
    # Check if qrencode is installed
    if command -v qrencode &> /dev/null; then
        echo "Scan this QR code with your phone:"
        qrencode -t ANSI "$url"
    else
        echo "For QR code generation, install qrencode with: brew install qrencode"
    fi
}

# Get local IP address
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

# Get external IP address 
EXTERNAL_IP=$(curl -s https://api.ipify.org)

# App URLs
LOCAL_URL="http://$LOCAL_IP:7231"
EXTERNAL_URL="http://$EXTERNAL_IP:7231"

# Clear any previous instances
pkill -f "streamlit run main.py" 2>/dev/null

# Run the Streamlit app in the background but keep it connected to this terminal
echo "Starting Workout Tracker..."
streamlit run main.py --server.port=7231 --server.address=0.0.0.0 > /dev/null 2>&1 &

# Store the PID to kill it when script exits
STREAMLIT_PID=$!

# Open browser to the app
sleep 2


# Display access information
echo ""
echo "✅ Workout Tracker is running!"
echo ""
echo "📱 Access from your local network:"
echo "$LOCAL_URL"
echo ""
echo "🌐 Access from anywhere (requires router port forwarding):"
echo "$EXTERNAL_URL"
echo ""
echo "🔐 Login credentials:"
echo "Username: ben"
echo "Password: admin"
echo ""

# Generate QR code for local access
echo "Local Access QR Code:"
generate_qr_code "$LOCAL_URL"

echo ""
echo "External Access QR Code:"
generate_qr_code "$EXTERNAL_URL"

echo ""
echo "ℹ️ The app will automatically stop when you close this terminal window."
echo "⚠️ Press Ctrl+C to stop the app and exit."
echo ""

# Wait for the streamlit process to finish
# This will keep the script running until streamlit exits or is killed
wait $STREAMLIT_PID 