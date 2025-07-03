@echo off
REM Apply fixes for the API issues

echo Applying API fixes...

REM Rebuild and restart the containers
docker-compose build
docker-compose down
docker-compose up -d

echo Waiting for services to start...
timeout /t 10

REM Test the health endpoint
echo Testing health endpoint...
curl -X "GET" ^
  "http://localhost:8000/health" ^
  -H "accept: application/json"

echo.
echo.
echo API fixes applied and services restarted successfully!
echo You can now test the upscale endpoint with:
echo python test-api.py --image ^<path-to-image^> [--direct]

pause
