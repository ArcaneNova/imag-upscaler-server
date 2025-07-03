# Fix for Request Parameter Conflict in Upscale Endpoint

## Problem
The `/upscale` endpoint has a conflict between two uses of the name "request":

1. `request: Request` - Required by SlowAPI's rate limiter to get client IP address
2. Query parameter also named `request` - Being passed as a query parameter in API calls

This is causing the error:
```
Exception: parameter `request` must be an instance of starlette.requests.Request
```

## Solution
We need to rename or remove the query parameter named "request" as it's conflicting with the properly typed `request: Request` parameter that the rate limiter needs.

Changes made:
1. Modified `main.py` to remove any reliance on a query parameter named "request"
2. Updated API documentation and examples to remove references to this parameter
3. Added clearer type annotation to the `request` parameter in the function signature

## Testing
After applying this fix, you should test the upscale endpoint with a curl command like:
```bash
curl -X 'POST' \
  'https://your-api-url.com/upscale?scale=2&face_enhance=false&direct_process=false' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_image.jpg;type=image/jpeg'
```

## Notes
- The parameter named "request" in the query string (`?request=renfrejvbr`) appears to be unnecessary and is causing the conflict
- The SlowAPI rate limiter expects the first parameter of rate-limited functions to be typed as `request: Request`
