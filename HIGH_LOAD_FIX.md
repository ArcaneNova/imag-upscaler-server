# Load Balancing and High-Load Fix for Real-ESRGAN API

This script helps manage high server load observed in the health check.

## Issues Observed:
- High CPU load average (12+)
- High memory usage (72.2%)
- Redis disconnected

## Optimizations Applied:

### 1. Memory Management
- Added dynamic device selection based on available memory
- Added periodic model cache clearing to prevent memory buildup
- Implemented aggressive garbage collection
- Added system load monitoring

### 2. Dynamic Resource Adjustment
- Added tile size adjustment based on available memory
- Dynamic max dimension adjustment based on system load
- Added load-aware processing to scale back resolution in high-load situations

### 3. Performance Monitoring
- Added detailed memory and GPU monitoring
- Improved logging for resource usage

### 4. Cache Management
- Added a counter-based and time-based cache clearing mechanism
- Scheduled periodic cleanup to prevent memory leaks

## Implementation Details:

1. **get_optimal_device()**: Now checks GPU memory before selecting it
2. **get_model()**: Added memory checks and dynamic tile sizing
3. **run_upscale()**: Added system load monitoring with dynamic parameters
4. **clear_model_cache()**: Enhanced with multiple GC passes and better resource tracking
5. **schedule_periodic_cache_clearing()**: Added background thread for automatic cache maintenance

These changes should significantly improve stability under heavy load and prevent memory-related crashes.
