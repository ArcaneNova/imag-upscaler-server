#!/usr/bin/env python
"""
Diagnose module import issues in the Real-ESRGAN API server
"""
import sys
import os
import importlib

def check_module(module_name):
    """Check if a module can be imported and print its path"""
    print(f"Checking module: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        print(f"   Path: {getattr(module, '__file__', 'Built-in module')}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    """Main diagnostic function"""
    print("\n" + "=" * 60)
    print("REAL-ESRGAN API SERVER DIAGNOSTIC TOOL")
    print("=" * 60 + "\n")
    
    print("Python version:", sys.version)
    print("Python executable:", sys.executable)
    print("Python path:", sys.path)
    print("Working directory:", os.getcwd())
    print("Directory contents:", os.listdir("."))
    
    if os.path.exists("app"):
        print("\nApp directory contents:", os.listdir("app"))
    else:
        print("\nApp directory not found!")
    
    print("\nChecking key modules:")
    check_module("app")
    check_module("app.main")
    check_module("fastapi")
    check_module("uvicorn")
    check_module("torch")
    check_module("realesrgan")
    
    print("\nChecking environment variables:")
    for var in ["PYTHONPATH", "PATH", "DOCKER_CONTAINER", "REDIS_HOST"]:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

if __name__ == "__main__":
    main()
