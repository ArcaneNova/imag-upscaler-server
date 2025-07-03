"""
Main application router - redirects to the actual application in app/main.py

This file exists to handle direct imports of 'main' without the 'app.' prefix
It simply re-exports the application instance from app.main
"""
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import the actual application from app.main
    from app.main import app
    logger.info("Successfully imported app from app.main")
except ImportError as e:
    # If import fails, create a minimal diagnostic app
    logger.error(f"Failed to import app.main: {e}")
    from fastapi import FastAPI
    
    app = FastAPI(title="Real-ESRGAN Diagnostic App")
    
    @app.get("/")
    async def diagnostic_root():
        return {
            "error": "Failed to load the main application",
            "message": str(e),
            "hint": "This is a diagnostic fallback. Please check server logs."
        }
    
    @app.get("/health")
    async def health_check():
        import os
        import sys
        return {
            "status": "error",
            "error": str(e),
            "python_version": sys.version,
            "paths": sys.path,
            "working_directory": os.getcwd(),
            "app_exists": os.path.exists("app"),
            "main_exists": os.path.exists("app/main.py") if os.path.exists("app") else False
        }
