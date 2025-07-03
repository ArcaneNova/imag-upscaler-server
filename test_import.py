#!/usr/bin/env python3
# Test script to check if RealESRGANer can be imported correctly

import sys
print(f"Python version: {sys.version}")

try:
    # Try to import RealESRGANer
    from realesrgan import RealESRGANer
    print('SUCCESS: RealESRGANer imported correctly!')
    
    # Check for specific attributes
    attributes = dir(RealESRGANer)
    print(f"RealESRGANer has {len(attributes)} attributes")
    print(f"Key methods: {'enhance' in attributes}")
    
except ImportError as ie:
    print(f'IMPORT ERROR: {ie}')
    print("Please install the realesrgan package: pip install realesrgan")
    
except Exception as e:
    print(f'FAILED: {e}')
