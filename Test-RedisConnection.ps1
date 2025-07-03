# Test-RedisConnection.ps1 - Script to test Redis connection for Real-ESRGAN API server

param (
    [string]$Url,
    [string]$Host,
    [int]$Port = 6379,
    [string]$Password,
    [switch]$UseEnvironment,
    [int]$Wait = 0,
    [int]$Retries = 1
)

# Add colors for better readability
$Green = @{ForegroundColor = "Green"}
$Yellow = @{ForegroundColor = "Yellow"}
$Red = @{ForegroundColor = "Red"}

Write-Host "Testing Redis Connection..." @Yellow
Write-Host

# Check if we should use environment variables
if ($UseEnvironment) {
    $EnvUrl = $env:REDIS_URL -or $env:REDISURL
    $EnvHost = $env:REDIS_HOST -or $env:REDISHOST
    $EnvPort = $env:REDIS_PORT -or $env:REDISPORT -or 6379
    $EnvPass = $env:REDIS_PASSWORD -or $env:REDISPASSWORD
    
    if ($EnvUrl) { $Url = $EnvUrl }
    if ($EnvHost) { $Host = $EnvHost }
    if ($EnvPort) { $Port = [int]$EnvPort }
    if ($EnvPass) { $Password = $EnvPass }
    
    Write-Host "Using Redis configuration from environment variables:" @Yellow
    if ($Url) { Write-Host "URL: $Url" }
    if ($Host) { Write-Host "Host: $Host" }
    if ($Port) { Write-Host "Port: $Port" }
    if ($Password) { Write-Host "Password: [MASKED]" }
    Write-Host
}

# Check if we have required parameters
if (-not $Url -and -not $Host) {
    Write-Host "No Redis connection parameters provided!" @Red
    Write-Host "Please provide either a Redis URL or a host/port combination." @Red
    Write-Host "Example: .\Test-RedisConnection.ps1 -Host localhost -Port 6379" @Yellow
    Write-Host "Example: .\Test-RedisConnection.ps1 -Url redis://localhost:6379" @Yellow
    Write-Host "Example: .\Test-RedisConnection.ps1 -UseEnvironment" @Yellow
    exit 1
}

# Use Python's redis-py library for the test
$Success = $false

for ($attempt = 1; $attempt -le $Retries; $attempt++) {
    if ($attempt -gt 1) {
        Write-Host "Retrying in $Wait seconds... (Attempt $attempt/$Retries)" @Yellow
        Start-Sleep -Seconds $Wait
    }
    
    $PythonScript = @"
import redis
import sys

try:
    # Test Redis connection
    client = None
    
    if '$Url':
        print(f"Connecting using URL: $Url")
        client = redis.Redis.from_url('$Url', decode_responses=True, socket_timeout=3)
    else:
        print(f"Connecting to $Host:$Port")
        connect_args = {
            'host': '$Host',
            'port': $Port,
            'decode_responses': True,
            'socket_timeout': 3
        }
        
        if '$Password':
            connect_args['password'] = '$Password'
            print("Using password authentication")
            
        client = redis.Redis(**connect_args)
    
    # Test connection with PING
    result = client.ping()
    print(f"PING result: {result}")
    
    # Get Redis info
    info = client.info()
    print(f"Connected to Redis {info.get('redis_version')}")
    print(f"Memory used: {info.get('used_memory_human')}")
    print(f"Connected clients: {info.get('connected_clients')}")
    
    client.close()
    sys.exit(0)
except Exception as e:
    print(f"Error connecting to Redis: {str(e)}")
    sys.exit(1)
"@

    # Save script to temp file
    $TempFile = [System.IO.Path]::GetTempFileName() + ".py"
    $PythonScript | Out-File -FilePath $TempFile -Encoding utf8
    
    try {
        # Execute the Python script
        $Output = python $TempFile 2>&1
        $LastExitCode = $LASTEXITCODE
        
        # Display output
        $Output | ForEach-Object {
            if ($LastExitCode -eq 0) {
                Write-Host $_ @Green
            } else {
                Write-Host $_ @Red
            }
        }
        
        # Check if successful
        if ($LastExitCode -eq 0) {
            $Success = $true
            break
        }
    }
    catch {
        Write-Host "Error executing Python script: $_" @Red
    }
    finally {
        # Clean up temp file
        if (Test-Path $TempFile) {
            Remove-Item $TempFile -Force
        }
    }
}

Write-Host
if ($Success) {
    Write-Host "✓ SUCCESS: Redis connection test passed!" @Green
    exit 0
} else {
    Write-Host "✗ ERROR: Redis connection test failed!" @Red
    exit 1
}
