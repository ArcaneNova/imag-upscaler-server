#!/usr/bin/env python
"""
Simple test to check if the app.main module can be imported
"""
try:
    from app.main import app
    print("✅ Successfully imported app.main.app")
    print(f"App info: {app.title}, version {app.version}")
except ImportError as e:
    print(f"❌ Failed to import app.main: {e}")
    
    # Additional diagnostics
    import sys
    print(f"Python path: {sys.path}")
    import os
    print(f"Working directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    if os.path.exists('app'):
        print(f"App directory contents: {os.listdir('app')}")
    else:
        print("App directory not found!")
