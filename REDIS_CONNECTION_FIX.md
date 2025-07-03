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
4. Railway-specific Redis configuration wasn't properly detected

## Solution
We've implemented several fixes to make the Redis connection more robust:

1. **Multiple Connection Methods**:
   - Added support for Redis URL format via `REDIS_URL` environment variable
   - Added detection of Railway-specific Redis variables (`REDISURL`, `REDISHOST`, etc.)
   - Kept traditional host/port configuration as fallback
   - Added support for password authentication

2. **Enhanced Fallback Mechanism**:
   - The server now tries multiple potential Redis hosts in sequence
   - Includes common hostnames: redis, localhost, 127.0.0.1, and platform-specific names
   - Prioritizes platform-specific hostnames when detected

3. **Automatic Reconnection Logic**:
   - The `get_redis_client()` dependency now tests the connection on each call
   - If the connection is lost, it automatically attempts to reconnect
   - Ensures operations that depend on Redis can recover from temporary disconnections

4. **Better Error Handling and Logging**:
   - Improved error messages with specific connection details
   - Logs all attempted hosts and specific errors for better debugging
   - Provides clear notification when operating without Redis

5. **Railway Deployment Support**:
   - Added detection for Railway environment and Redis service variables
   - Created `fix-railway-redis.sh` script for diagnosing Redis issues in Railway
   - Updated environment variable handling to be compatible with Railway's Redis service

## Usage

### Configuration Options

The Redis connection can now be configured in multiple ways:

1. **Using Redis URL** (recommended for managed environments):
   ```
   REDIS_URL=redis://[username]:[password]@host:port/db
   ```

2. **Using Traditional Configuration**:
   ```
   REDIS_HOST=your-redis-host
   REDIS_PORT=6379
   REDIS_PASSWORD=optional-password
   ```

3. **Disabling Redis**:
   ```
   REDIS_DISABLE=true
   ```

### Testing Redis Connection

Use the provided test script to verify your Redis connection:

```bash
python test-redis.py --url redis://localhost:6379
```

Or using host/port:
```bash
python test-redis.py --host localhost --port 6379 --password yourpassword
```

### Fixing Redis in Railway

If you're using Railway, run the included diagnostic script:

```bash
./fix-railway-redis.sh
```

The script will:
1. Detect available Redis environment variables
2. Generate a recommended Redis URL configuration
3. Provide instructions for configuring Redis in Railway
4. Test the Redis connection if possible
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
