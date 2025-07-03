# Redis Connection Fix for Real-ESRGAN API Server

## Problem
The API server was failing to connect to Redis with the error:
```
Error -2 connecting to redis:6379. Name or service not known.
```

This occurred because:
1. The server was trying to connect to a host named "redis" which is only resolvable in Docker environments
2. The fallback mechanism for non-Docker environments was not robust enough
3. There was no automatic reconnection logic if Redis temporarily disconnected

## Solution
We've implemented several fixes to make the Redis connection more robust:

1. **Enhanced Fallback Mechanism**:
   - The server now tries multiple potential Redis hosts in sequence
   - Includes common hostnames: redis, localhost, 127.0.0.1
   - Adds Railway-specific environment detection for Railway deployments

2. **Automatic Reconnection Logic**:
   - The `get_redis_client()` dependency now tests the connection on each call
   - If the connection is lost, it automatically attempts to reconnect
   - Ensures operations that depend on Redis can recover from temporary disconnections

3. **Better Error Handling and Logging**:
   - Improved error messages with specific connection details
   - Logs all attempted hosts and specific errors for better debugging
   - Provides clear notification when operating without Redis

4. **Local Development Support**:
   - Added scripts for starting Redis locally for testing (start-redis.sh, start-redis.ps1)
   - Created a test script to verify Redis connection (test-redis-connection.py)
   - Provided .env file with appropriate local development settings

## Usage

### Starting Redis Locally
For Linux/macOS:
```bash
./start-redis.sh
```

For Windows:
```powershell
.\start-redis.ps1
```

### Testing Redis Connection
```bash
python test-redis-connection.py
```

### Operating Without Redis
If you don't need Redis functionality, you can disable it by setting:
```
REDIS_DISABLE=true
```
in your `.env` file.

## Notes
- The API server will function without Redis, but job status tracking will be limited
- For Docker Compose deployments, ensure the Redis service is started before the API service
- For Railway deployments, ensure the Redis service is properly configured
- The health endpoint will report Redis status as "connected" or "disconnected"
