#!/bin/bash

# Run the Hearts Game backend
# This script sets up the Python path correctly for the backend module

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/backend:$PYTHONPATH"
python -m backend.main