#!/bin/bash
# run.sh -- Quick launcher for Linux / macOS
# Usage: chmod +x run.sh && ./run.sh

set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Starting Retail Shelf Intelligence System..."
python src/main.py "$@"
