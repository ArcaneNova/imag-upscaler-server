# Real-ESRGAN Upscaler API Documentation

## Overview
This API provides image upscaling capabilities using Real-ESRGAN models, with support for both synchronous and asynchronous processing.

## Base URL
```
https://your-api-url.com
```

## Authentication
No authentication is required for current endpoints. Rate limiting is applied to prevent abuse.

## Endpoints

### Health Check
Verify the API is functioning properly and check system status.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1751575411.1202464,
  "version": "2.0.0",
  "redis": "connected",
  "system": {
    "cpu_usage": "20.8%",
    "memory_usage": "74.4%",
    "disk_usage": "68.3%",
    "load_average": [12.14, 14.57, 15.27]
  }
}
```

### Submit Upscale
Submit an image for upscaling with Real-ESRGAN.

```
POST /upscale
```

**Parameters:**
- `file` (form-data, required): The image file to upscale
- `scale` (query, optional): Upscaling factor [2, 4], default: 2
- `face_enhance` (query, optional): Whether to enhance faces [true, false], default: false
- `direct_process` (query, optional): Process immediately instead of queueing [true, false], default: false

**Example Request:**
```bash
curl -X 'POST' \
  'https://your-api-url.com/upscale?scale=2&face_enhance=false&direct_process=false' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.jpg;type=image/jpeg'
```

**Response (asynchronous):**
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "queued",
  "message": "Image submitted for upscaling",
  "estimated_time": "30-120 seconds",
  "parameters": {
    "scale": 2,
    "face_enhance": false
  }
}
```

**Response (direct processing):**
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "completed",
  "processing_time": "8.45 seconds",
  "result_url": "https://res.cloudinary.com/your-cloud/image/upload/v1234567890/realesrgan_upscales/abcdef.png",
  "message": "Image upscaled successfully",
  "parameters": {
    "scale": 2,
    "face_enhance": false
  }
}
```

### Check Status
Check the status of an upscaling job.

```
GET /status/{job_id}
```

**Parameters:**
- `job_id` (path, required): The ID of the upscale job

**Example Request:**
```bash
curl -X 'GET' \
  'https://your-api-url.com/status/f47ac10b-58cc-4372-a567-0e02b2c3d479' \
  -H 'accept: application/json'
```

**Response:**
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "completed",
  "created_at": 1751575411.1202464,
  "completed_at": 1751575420.4567123,
  "processing_time": "9.34 seconds",
  "result_url": "https://res.cloudinary.com/your-cloud/image/upload/v1234567890/realesrgan_upscales/abcdef.png",
  "parameters": {
    "scale": 2,
    "face_enhance": false,
    "filename": "original.jpg",
    "content_type": "image/jpeg",
    "file_size": 245678
  }
}
```

## Rate Limiting
- `/upscale`: 20 requests per minute
- `/status/{job_id}`: 60 requests per minute
- `/health`: 120 requests per minute

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid file type: text/plain. Only images are supported."
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded: 20 per 1 minute"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to submit job: Internal server error"
}
```

## Notes
- Maximum file size: 15MB
- Supported image formats: JPEG, PNG, WebP
- For best results, submit images with dimensions between 512x512 and 2048x2048 pixels
- Use `direct_process=true` for immediate processing, or omit for background processing
- Background processing requires a second request to `/status/{job_id}` to get the results
