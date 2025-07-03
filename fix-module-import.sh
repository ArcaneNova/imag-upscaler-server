#!/bin/bash
# Fix script for module import issue

# Determine where app.py is located
if [ -d "/app/app" ]; then
    APP_DIR="/app/app"
elif [ -d "./app" ]; then
    APP_DIR="./app"
else
    echo "Cannot find app directory!"
    exit 1
fi

echo "App directory found at: $APP_DIR"

# Create an __init__.py file if it doesn't exist
if [ ! -f "$APP_DIR/__init__.py" ]; then
    echo "Creating __init__.py in app directory"
    echo "# Make app directory a proper Python package" > "$APP_DIR/__init__.py"
fi

# Check if we can import the app module
python -c "import app; print('Successfully imported app module')"
python -c "from app import main; print('Successfully imported app.main module')"

echo "Fix completed!"
