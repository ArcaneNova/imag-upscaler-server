#!/usr/bin/env python3
"""
Test script for the upscale endpoint after applying the request parameter fix.
"""

import requests
import os
import argparse
from pprint import pprint
import sys

def test_upscale_endpoint(api_url, image_path):
    """Test the upscale endpoint with an image file."""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return False
    
    # Get file extension and size
    _, ext = os.path.splitext(image_path)
    file_size = os.path.getsize(image_path)
    print(f"Testing with: {image_path} ({file_size/1024:.1f} KB)")
    
    # Prepare the file upload
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, f'image/{ext[1:]}' if ext else 'image/jpeg')}
        
        # Test parameters
        params = {
            'scale': 2,
            'face_enhance': 'false',
            'direct_process': 'true'  # Use direct_process=true for synchronous testing
        }
        
        print(f"Sending request to {api_url}/upscale...")
        try:
            response = requests.post(f"{api_url}/upscale", files=files, params=params)
            
            # Print response details
            print(f"Status code: {response.status_code}")
            if response.status_code == 200:
                print("SUCCESS! Endpoint is working correctly.")
                print("Response data:")
                pprint(response.json())
                return True
            else:
                print("ERROR! Endpoint returned an error:")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"Error occurred: {e}")
            return False

def test_health_endpoint(api_url):
    """Test the health endpoint to verify API is running."""
    
    try:
        print(f"Checking API health at {api_url}/health...")
        response = requests.get(f"{api_url}/health")
        
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS! Health endpoint is working.")
            print("API health data:")
            pprint(response.json())
            return True
        else:
            print("ERROR! Health endpoint returned an error:")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the upscale endpoint after fixing the request parameter issue.')
    parser.add_argument('--url', default='http://localhost:8000', help='API URL (default: http://localhost:8000)')
    parser.add_argument('--image', default='test.jpg', help='Path to test image file (default: test.jpg)')
    parser.add_argument('--health-only', action='store_true', help='Only test the health endpoint')
    
    args = parser.parse_args()
    
    # First test health endpoint
    health_ok = test_health_endpoint(args.url)
    
    if not health_ok:
        print("Health check failed. API may not be running correctly.")
        sys.exit(1)
    
    if not args.health_only:
        # Then test upscale endpoint if health check passed
        upscale_ok = test_upscale_endpoint(args.url, args.image)
        
        if not upscale_ok:
            print("Upscale test failed.")
            sys.exit(1)
    
    print("All tests passed!")
    sys.exit(0)
