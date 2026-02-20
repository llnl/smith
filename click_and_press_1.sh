#!/bin/bash
# Utility to click mouse at current position and press '1' every 3 seconds

echo "Starting auto-clicker. Press Ctrl+C to stop."

while true; do
    # Click at current mouse position (. means current position)
    cliclick c:.

    # Brief delay
    sleep 0.1

    # Press the '1' key
    osascript -e 'tell application "System Events" to keystroke "1"'

    # Wait 10 seconds before next iteration
    sleep 10
done

