#!/usr/bin/env python3
"""
Test script for Redis connection in the Real-ESRGAN API server.
"""

import os
import redis
import argparse
import sys
import time

def test_redis_connection(url=None, host=None, port=None, password=None):
    """Test connecting to Redis with the given parameters"""
    
    print(f"Testing Redis connection...")
    
    # Track connection attempts
    attempts = []
    
    # Try connection method 1: URL
    if url:
        try:
            print(f"Attempting connection using Redis URL: {url}")
            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3
            )
            client.ping()
            print(f"✓ SUCCESS: Connected to Redis using URL!")
            
            # Get Redis info
            info = client.info()
            print(f"Redis version: {info.get('redis_version')}")
            print(f"Connected clients: {info.get('connected_clients')}")
            print(f"Memory used: {info.get('used_memory_human')}")
            
            client.close()
            return True
        except Exception as e:
            print(f"✗ ERROR: Failed to connect using Redis URL: {str(e)}")
            attempts.append(f"URL method failed: {str(e)}")
    
    # Try connection method 2: Host/Port
    if host:
        try:
            port = int(port) if port else 6379
            print(f"Attempting connection to {host}:{port}")
            
            connect_args = {
                "host": host,
                "port": port,
                "decode_responses": True,
                "socket_timeout": 3,
                "socket_connect_timeout": 3
            }
            
            if password:
                connect_args["password"] = password
                print(f"Using password authentication")
            
            client = redis.Redis(**connect_args)
            client.ping()
            print(f"✓ SUCCESS: Connected to Redis at {host}:{port}!")
            
            # Get Redis info
            info = client.info()
            print(f"Redis version: {info.get('redis_version')}")
            print(f"Connected clients: {info.get('connected_clients')}")
            print(f"Memory used: {info.get('used_memory_human')}")
            
            client.close()
            return True
        except Exception as e:
            print(f"✗ ERROR: Failed to connect to {host}:{port}: {str(e)}")
            attempts.append(f"Host/port method failed: {str(e)}")
    
    # Try connection method 3: Environment variables
    print("Checking environment variables for Redis configuration...")
    
    env_url = os.getenv("REDIS_URL") or os.getenv("REDISURL")
    if env_url:
        try:
            print(f"Attempting connection using REDIS_URL from environment: {env_url}")
            client = redis.Redis.from_url(
                env_url,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3
            )
            client.ping()
            print(f"✓ SUCCESS: Connected to Redis using environment URL!")
            
            # Get Redis info
            info = client.info()
            print(f"Redis version: {info.get('redis_version')}")
            print(f"Connected clients: {info.get('connected_clients')}")
            print(f"Memory used: {info.get('used_memory_human')}")
            
            client.close()
            return True
        except Exception as e:
            print(f"✗ ERROR: Failed to connect using environment REDIS_URL: {str(e)}")
            attempts.append(f"Environment URL method failed: {str(e)}")
    
    env_host = os.getenv("REDIS_HOST") or os.getenv("REDISHOST")
    env_port = os.getenv("REDIS_PORT") or os.getenv("REDISPORT") or "6379"
    env_password = os.getenv("REDIS_PASSWORD") or os.getenv("REDISPASSWORD")
    
    if env_host:
        try:
            print(f"Attempting connection using Redis host/port from environment: {env_host}:{env_port}")
            
            connect_args = {
                "host": env_host,
                "port": int(env_port),
                "decode_responses": True,
                "socket_timeout": 3,
                "socket_connect_timeout": 3
            }
            
            if env_password:
                connect_args["password"] = env_password
                print(f"Using password authentication from environment")
            
            client = redis.Redis(**connect_args)
            client.ping()
            print(f"✓ SUCCESS: Connected to Redis using environment host/port!")
            
            # Get Redis info
            info = client.info()
            print(f"Redis version: {info.get('redis_version')}")
            print(f"Connected clients: {info.get('connected_clients')}")
            print(f"Memory used: {info.get('used_memory_human')}")
            
            client.close()
            return True
        except Exception as e:
            print(f"✗ ERROR: Failed to connect using environment host/port: {str(e)}")
            attempts.append(f"Environment host/port method failed: {str(e)}")
    
    print("\n=== All connection attempts failed ===")
    for i, attempt in enumerate(attempts, 1):
        print(f"{i}. {attempt}")
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Redis connection for Real-ESRGAN API server')
    parser.add_argument('--url', help='Redis URL in format redis://[user]:[password]@host:port/db')
    parser.add_argument('--host', help='Redis host')
    parser.add_argument('--port', help='Redis port', default="6379")
    parser.add_argument('--password', help='Redis password')
    parser.add_argument('--wait', help='Wait time in seconds to retry connection', type=int, default=0)
    parser.add_argument('--retries', help='Number of retries', type=int, default=1)
    
    args = parser.parse_args()
    
    # No arguments provided, check environment
    if not (args.url or args.host or os.getenv("REDIS_URL") or os.getenv("REDISURL") or os.getenv("REDIS_HOST") or os.getenv("REDISHOST")):
        print("No Redis connection parameters provided via arguments or environment variables.")
        print("Please provide Redis URL or host/port information.")
        print("For help, use: python test-redis.py --help")
        sys.exit(1)
    
    success = False
    
    for attempt in range(args.retries):
        if attempt > 0:
            print(f"\nRetrying in {args.wait} seconds... (Attempt {attempt+1}/{args.retries})")
            time.sleep(args.wait)
        
        if test_redis_connection(args.url, args.host, args.port, args.password):
            success = True
            break
    
    if not success:
        print("\nAll Redis connection attempts failed.")
        sys.exit(1)
    else:
        print("\nRedis connection test passed successfully!")
        sys.exit(0)
