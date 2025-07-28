#!/bin/bash
# run_demo.sh - Script to run the complete demo system for video recording

# Display banner
echo "========================================================"
echo "  SymbioFlows Demo System Launcher"
echo "========================================================"
echo "This script will prepare and start all necessary services"
echo "for recording a demo video of the complete user journey."
echo ""

# Check for required commands
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required but not installed. Aborting."; exit 1; }

# Set up environment
echo "Setting up environment..."
export NODE_ENV=development
export DEMO_MODE=true

# Create data directory if it doesn't exist
mkdir -p data

# Prepare demo data
echo "Preparing demo data..."
python3 scripts/demo_video_preparation.py

# Check if preparation was successful
if [ $? -ne 0 ]; then
    echo "Error: Demo preparation failed. Please check logs."
    exit 1
fi

# Start backend services
echo "Starting backend services..."
cd backend
npm start &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 10

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Display success message
echo ""
echo "========================================================"
echo "  Demo System Started Successfully!"
echo "========================================================"
echo "Backend running with PID: $BACKEND_PID"
echo "Frontend running with PID: $FRONTEND_PID"
echo ""
echo "Access the application at: http://localhost:3000"
echo ""
echo "Follow the demo guide at: docs/DEMO_VIDEO_GUIDE.md"
echo ""
echo "Press Ctrl+C to stop all services when done."
echo "========================================================"

# Function to handle script termination
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Demo system stopped."
    exit 0
}

# Register cleanup function for script termination
trap cleanup SIGINT SIGTERM

# Keep script running
wait