#!/bin/sh
echo "Monitoring for update signal..."
while true; do
    if [ -f "/app/scripts/update_signal" ]; then
        echo "Update signal detected. Restarting chatgpt-next-web..."
        cd /app
        docker-compose -p llm_api_for_next_chat up chatgpt-next-web -d
        rm /app/scripts/update_signal
    fi
    sleep 5
done
