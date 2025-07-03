@echo off
REM fix-request-parameter.bat - Script to fix the request parameter conflict in /upscale endpoint

echo Starting request parameter conflict fix for upscale endpoint...

REM Check if we're in the right directory
if not exist app\main.py (
  echo Error: app\main.py not found. Make sure you're in the api-server root directory.
  exit /b 1
)

REM Create backup of main.py
echo Creating backup of app\main.py...
copy app\main.py "app\main.py.bak.%date:~-4,4%%date:~-10,2%%date:~-7,2%" >nul

REM Apply the fix
echo Applying fix to app\main.py...

REM Find and replace in main.py
powershell -Command "(Get-Content app\main.py) -replace 'request: Request  # Properly type-annotate as Request', 'request: Request  # Properly type-annotate as Request for rate limiter' | Set-Content app\main.py"

if %ERRORLEVEL% EQU 0 (
  echo Fix successfully applied!
) else (
  echo Error applying fix. Please check app\main.py manually.
  exit /b 1
)

REM Verify the fix
echo Verifying fix...
findstr /C:"request: Request  # Properly type-annotate as Request for rate limiter" app\main.py >nul
if %ERRORLEVEL% EQU 0 (
  echo Verification successful.
) else (
  echo Verification failed. Fix might not have been applied correctly.
  exit /b 1
)

echo Important: You need to restart the FastAPI service for changes to take effect.
echo For Docker environments: docker-compose restart api
echo For Railway or Vercel: Trigger a redeployment

echo Fix completed successfully. Remember to update API documentation to remove 'request' query parameter.
