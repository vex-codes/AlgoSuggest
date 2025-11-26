#!/bin/bash
echo "Starting AlgoSuggest Production Server..."

# Check if Gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "Gunicorn could not be found. Installing..."
    pip3 install gunicorn
fi

# Kill any process running on port 5000
PID=$(lsof -t -i:5000)
if [ -n "$PID" ]; then
    echo "Port 5000 is in use by PID $PID. Killing it..."
    kill -9 $PID
fi

# Run Gunicorn
# -w 4: Use 4 worker processes for concurrency
# -b 0.0.0.0:5000: Bind to all interfaces on port 5000
# --timeout 120: Increase timeout for initial model loading
echo "Running Gunicorn with 4 workers..."
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 recommendation_engine:app
