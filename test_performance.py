#!/usr/bin/env python3
"""
Quick performance validation script for the optimized Real-ESRGAN API
"""

import requests
import time
import json
import sys
from pathlib import Path

def test_api_performance():
    """Test the API performance with a sample image"""
    
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        response = requests.get(f"{api_url}/ping", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding correctly")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API at {api_url}: {e}")
        return False
    
    print("✅ API is responding")
    
    # Create a test image file (small test image)
    test_image_path = Path("test_image.jpg")
    if not test_image_path.exists():
        print("⚠️  No test image found, please upload an image manually")
        return True
    
    # Test upscaling performance
    print("🚀 Testing upscaling performance...")
    
    start_time = time.time()
    
    with open(test_image_path, 'rb') as f:
        files = {'file': f}
        params = {
            'scale': 2,
            'face_enhance': False,
            'direct_process': False
        }
        
        try:
            response = requests.post(
                f"{api_url}/upscale",
                files=files,
                params=params,
                timeout=120  # 2 minute timeout
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Upscaling successful in {processing_time:.2f} seconds")
                print(f"📊 Result: {json.dumps(result, indent=2)}")
                
                # Performance analysis
                if processing_time < 15:
                    print("🔥 Excellent performance!")
                elif processing_time < 30:
                    print("✅ Good performance")
                elif processing_time < 60:
                    print("⚠️  Acceptable performance")
                else:
                    print("⚠️  Slow performance - may need optimization")
                
                return True
            else:
                print(f"❌ API error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False

if __name__ == "__main__":
    print("🧪 Real-ESRGAN API Performance Test")
    print("=" * 50)
    
    success = test_api_performance()
    sys.exit(0 if success else 1)
