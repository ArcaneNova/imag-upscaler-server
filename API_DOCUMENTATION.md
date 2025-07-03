# Real-ESRGAN API Documentation

## Base URL

The API is available at: `https://imag-upscaler-server-production.up.railway.app`

## Authentication

Currently, the API uses rate limiting but does not require authentication.

## Endpoints

### Root Endpoint

```
GET /
```

Redirects to the API documentation.

### Health Check

```
GET /health
```

Returns the current health status of the API server.

**Response Example:**

```json
{
  "status": "healthy",
  "timestamp": 1717186354.753,
  "version": "2.0.0",
  "redis": "connected",
  "redis_info": {
    "used_memory": "1.2M",
    "connected_clients": 2,
    "total_connections_received": 45
  },
  "system": {
    "cpu_usage": "12.5%",
    "memory_usage": "34.7%",
    "disk_usage": "58.2%",
    "load_average": [0.12, 0.25, 0.15]
  }
}
```

### Submit Image for Upscaling

```
POST /upscale
```

Upload an image for upscaling with Real-ESRGAN.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | The image file to upscale (max 15MB) |
| scale | Integer | No | Upscaling factor (2 or 4, default: 2) |
| face_enhance | Boolean | No | Whether to enhance face regions (default: false) |

**Response Example:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Image submitted for upscaling",
  "estimated_time": "30-120 seconds",
  "parameters": {
    "scale": 2,
    "face_enhance": false
  }
}
```

### Get Job Status

```
GET /status/{job_id}
```

Check the status of a previously submitted upscaling job.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| job_id | String | Yes | The job ID returned from the upscale endpoint |

**Response Example:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "result_url": "https://res.cloudinary.com/demo/image/upload/v1722345678/upscaled_image.jpg",
  "original_filename": "photo.jpg",
  "processing_time": 45.2,
  "completed_at": 1717186400.123
}
```

Possible status values:
- `queued`: Job is waiting to be processed
- `processing`: Job is actively being processed
- `completed`: Job has finished successfully
- `failed`: Job encountered an error
- `cancelled`: Job was cancelled by the user

### Cancel Job

```
DELETE /job/{job_id}
```

Cancel a queued or processing job.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| job_id | String | Yes | The job ID to cancel |

**Response Example:**

```json
{
  "message": "Job cancelled successfully"
}
```

### Get API Metrics

```
GET /metrics
```

Get detailed API metrics including system status, job statistics, and performance metrics.

**Response Example:**

```json
{
  "timestamp": 1717186354.753,
  "system": {
    "cpu_usage_percent": 15.2,
    "memory_usage_percent": 65.4,
    "memory_available_gb": 1.25,
    "disk_usage_percent": 42.8,
    "disk_free_gb": 10.5
  },
  "redis": {
    "connected_clients": 3,
    "used_memory_mb": 128.45,
    "operations_per_second": 42
  },
  "jobs": {
    "total_processed": 1250,
    "current_queue_length": 5,
    "average_processing_time": 35.2
  }
}
```

## Rate Limits

- `/upscale`: 20 requests per minute
- `/status/{job_id}`: 60 requests per minute
- Other endpoints: 30 requests per minute

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (invalid parameters) |
| 404 | Job not found |
| 429 | Too many requests (rate limit exceeded) |
| 500 | Server error |

## Sample Usage (curl)

```bash
# Upload image for upscaling
curl -X POST "https://imag-upscaler-server-production.up.railway.app/upscale" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "scale=2" \
  -F "face_enhance=false"

# Check job status
curl -X GET "https://imag-upscaler-server-production.up.railway.app/status/550e8400-e29b-41d4-a716-446655440000"
```

## Sample Usage (JavaScript)

```javascript
// Upload image for upscaling
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('scale', '2');
formData.append('face_enhance', 'false');

const response = await fetch('https://imag-upscaler-server-production.up.railway.app/upscale', {
  method: 'POST',
  body: formData
});

const result = await response.json();
const jobId = result.job_id;

// Check job status
const statusResponse = await fetch(`https://imag-upscaler-server-production.up.railway.app/status/${jobId}`);
const statusResult = await statusResponse.json();
```
