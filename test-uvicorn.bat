@echo off
echo Testing Uvicorn import and command...

REM Test if uvicorn can be run and find the app module
echo.
echo Testing uvicorn command:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log --no-server-header --app-dir . --check-only

REM Test if the app module can be imported directly
echo.
echo Testing direct import:
python test-imports.py

echo.
echo Done!
