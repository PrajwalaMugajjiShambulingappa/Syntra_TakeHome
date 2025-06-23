#!/bin/bash

# Utility to kill process on a port
kill_port() {
    port=$1
    pid=$(lsof -ti tcp:$port)
    if [ ! -z "$pid" ]; then
        echo "Port $port is in use by PID $pid. Killing..."
        kill -9 $pid
    else
        echo "Port $port is free."
    fi
}

echo "Installing Python backend dependencies..."
pip install -r requirements.txt

echo "Navigating to frontend folder and installing dependencies..."
cd frontend || exit
npm install

# Move back to root for launching backend
cd ..

# Kill any process using backend port 8000 and frontend port 3000
echo "Checking and freeing up required ports..."
kill_port 8000
kill_port 3000

# Launch backend in background
echo "Launching backend on port 8000..."
uvicorn src.main:app --reload --port 8000 &

# Launch frontend in background
echo "Launching frontend on port 3000..."
cd frontend || exit
npm start &

echo "All systems are running. Visit http://localhost:3000"
