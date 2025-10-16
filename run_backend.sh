#!/bin/bash
# Script to run the Hearts Game Backend

echo "Starting Hearts Game Backend..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "Please edit .env and set your CHECKPOINT_PATH before running again."
    exit 1
fi

# Run FastAPI server
cd backend
python main.py


