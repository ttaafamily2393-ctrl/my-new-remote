#!/bin/bash

# Configuration
LOG_FILE="/home/tony/intruder_alerts.log"
ALERT_TITLE="⚠️ SECURITY ALERT"

echo "Intruder monitoring service started..."

while true; do
    # 1. Detect unauthorized remote interactive shells (SSH/Netcat/Reverse Shells)
    INTRUDER_PID=$(ps aux | grep -E '(nc -e|bash -i|sh -i|reverse_shell|ncat)' | grep -v grep | awk '{print $2}')
    
    # 2. Detect if an unknown user logs into an active SSH terminal session
    SSH_USERS=$(who | grep -v 'tony' | wc -l)

    if [ ! -z "$INTRUDER_PID" ] || [ "$SSH_USERS" -gt 0 ]; then
        MESSAGE="Suspicious terminal activity or unauthorized user session detected!"
        
        # Write to local forensic log path
        echo "$(date): $MESSAGE" >> "$LOG_FILE"
        
        # Send a graphical pop-up alert to your Chromebook desktop screen
        export DISPLAY=:0
        notify-send "$ALERT_TITLE" "$MESSAGE" --urgency=critical
        
        # Play an audible warning beep through the system speaker
        echo -e "\a"
    fi
    
    # Sleep for 5 seconds before scanning process metrics again to preserve RAM
    sleep 5
done
