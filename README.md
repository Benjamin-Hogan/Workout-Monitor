# Workout Tracker App

A simple application to track your workouts and fitness progress.

## Quick Start

1. **Run the app directly**:

   ```
   ./RunWorkoutApp.command
   ```

   This will open the Workout Tracker in your default browser. The app will automatically stop when you close the terminal window.

2. **Access from your phone at the gym**:

   When you run the app, it will display a URL (like http://192.168.0.141:7231) that you can use to access the app from any device on your local network, including your phone.

   If you have qrencode installed (`brew install qrencode`), the app will also display a QR code you can scan with your phone.

3. **Access from anywhere**:

   The app will also show you an external URL that you can use to access your workout tracker from anywhere with internet access.

   This requires that you've configured your router to forward port 7231 to your computer (which you've already done).

   ```
   http://your-external-ip:7231
   ```

4. **Login credentials**:

   The app is protected with authentication. You must log in to access any page:

   - Username: ben (case-insensitive, so "Ben" or "BEN" will also work)
   - Password: admin (case-sensitive)

5. **Add an icon** (optional):
   ```
   pip install pillow
   ./add_icon.py
   ```
   Then follow the on-screen instructions to set the icon.

## Requirements

- Python 3.6 or higher
- Streamlit
- Other dependencies as specified in `requirements.txt`

## Installation

If you haven't installed the required packages:

```
pip install -r requirements.txt
```

## Features

- Track your workouts
- Monitor your progress
- Set fitness goals
- Visualize your achievements
- Access from your mobile device at the gym
- Access from anywhere with internet
- Protected with secure authentication
- Automatically terminates when terminal is closed

## Troubleshooting

- **App won't start**: Make sure you have all dependencies installed
- **Permission denied**: Run `chmod +x RunWorkoutApp.command` to make the file executable
- **Security warning**: On macOS, you might need to right-click the file and select "Open" the first time you run it
- **Can't access from phone**: Make sure your phone is on the same WiFi network as your computer
- **Can't access remotely**: Verify your router port forwarding settings (port 7231)
- **Stop the app**: Either close the terminal window or press Ctrl+C in the terminal window

## Security Considerations

- The authentication is basic and meant for personal use
- For increased security, consider using stronger passwords
- Your workout data is stored locally on your computer
- Always close the terminal window when you're done using the app

## Development

To modify the app, edit the `main.py` file and any associated Python modules.
