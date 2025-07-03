# start-redis.ps1 - PowerShell script to start Redis locally for Windows users

Write-Host "Starting Redis for testing the Real-ESRGAN API server..." -ForegroundColor Cyan

# Function to check if Docker is available
function Test-DockerAvailable {
    try {
        $null = docker --version
        return $true
    } catch {
        return $false
    }
}

# Function to check if WSL is available
function Test-WSLAvailable {
    try {
        $null = wsl --status
        return $true
    } catch {
        return $false
    }
}

# Function to start Redis in Docker
function Start-RedisDocker {
    Write-Host "Starting Redis using Docker..." -ForegroundColor Cyan
    docker run --name realesrgan-redis -p 6379:6379 -d redis:7-alpine
    
    # Check if Redis started in Docker
    Start-Sleep -Seconds 2
    try {
        $null = docker exec realesrgan-redis redis-cli ping
        Write-Host "Redis is now running on localhost:6379 via Docker" -ForegroundColor Green
        Write-Host "To stop Redis: docker stop realesrgan-redis" -ForegroundColor Yellow
        Write-Host "To remove container: docker rm realesrgan-redis" -ForegroundColor Yellow
        return $true
    } catch {
        Write-Host "Failed to start Redis in Docker" -ForegroundColor Red
        return $false
    }
}

# Main script logic
if (Test-DockerAvailable) {
    # Check if the Redis container is already running
    $redisRunning = docker ps | Select-String -Pattern "realesrgan-redis"
    
    if ($redisRunning) {
        Write-Host "Redis is already running in Docker" -ForegroundColor Green
    } else {
        # Check if container exists but is stopped
        $redisExists = docker ps -a | Select-String -Pattern "realesrgan-redis"
        
        if ($redisExists) {
            Write-Host "Found existing Redis container, starting it..." -ForegroundColor Cyan
            docker start realesrgan-redis
            Write-Host "Redis is now running on localhost:6379" -ForegroundColor Green
        } else {
            # Start new Redis container
            Start-RedisDocker
        }
    }
} elseif (Test-WSLAvailable) {
    Write-Host "Docker not found, but WSL is available." -ForegroundColor Yellow
    Write-Host "You can install Redis in WSL with: wsl sudo apt install redis-server" -ForegroundColor Yellow
    Write-Host "And then start it with: wsl sudo service redis-server start" -ForegroundColor Yellow
} else {
    Write-Host "Neither Docker nor WSL is available." -ForegroundColor Red
    Write-Host "Please install Redis for Windows from: https://github.com/microsoftarchive/redis/releases" -ForegroundColor Yellow
    Write-Host "Alternatively, you can set REDIS_DISABLE=true in .env to run without Redis" -ForegroundColor Yellow
}

Write-Host "`nNOTE: After starting Redis, the API server should automatically connect to it." -ForegroundColor Cyan
