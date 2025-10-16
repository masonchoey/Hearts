#!/bin/bash
# Script to run the Hearts Game Frontend

echo "Starting Hearts Game Frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Run development server
npm run dev


