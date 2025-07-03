#!/bin/bash

# fix-railway-redis.sh - Script to fix Redis configuration in Railway environment

# Print colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}Railway Redis Connection Fix${NC}"
echo -e "${YELLOW}=========================================${NC}"
echo

# Check if running in Railway environment
if [ -z "$RAILWAY_ENVIRONMENT" ]; then
  echo -e "${YELLOW}This script is designed to be run in a Railway environment.${NC}"
  echo -e "${YELLOW}If you're running this locally, it will only generate instructions.${NC}"
  echo
fi

echo -e "${GREEN}Current Redis configuration:${NC}"
echo -e "REDIS_HOST: ${YELLOW}$REDIS_HOST${NC}"
echo -e "REDIS_PORT: ${YELLOW}$REDIS_PORT${NC}"
echo -e "REDIS_URL:  ${YELLOW}$REDIS_URL${NC}"
echo

# Check for Railway Redis service variables
if [ -n "$REDISHOST" ] || [ -n "$REDISPORT" ] || [ -n "$REDISUSER" ] || [ -n "$REDISPASSWORD" ] || [ -n "$REDISURL" ]; then
  echo -e "${GREEN}Detected Railway Redis environment variables:${NC}"
  [ -n "$REDISHOST" ] && echo -e "REDISHOST: ${YELLOW}$REDISHOST${NC}"
  [ -n "$REDISPORT" ] && echo -e "REDISPORT: ${YELLOW}$REDISPORT${NC}"
  [ -n "$REDISUSER" ] && echo -e "REDISUSER: ${YELLOW}$REDISUSER${NC}"
  [ -n "$REDISPASSWORD" ] && echo -e "REDISPASSWORD: ${YELLOW}[MASKED]${NC}"
  [ -n "$REDISURL" ] && echo -e "REDISURL: ${YELLOW}$REDISURL${NC}"
  echo
else
  echo -e "${RED}No Railway Redis environment variables detected.${NC}"
  echo -e "${YELLOW}You may need to add a Redis service to your Railway project.${NC}"
  echo
fi

echo -e "${GREEN}Recommended configuration:${NC}"

# Generate recommended Redis URL based on available environment variables
if [ -n "$REDISURL" ]; then
  echo -e "REDIS_URL: ${YELLOW}$REDISURL${NC} (use this value)"
elif [ -n "$REDISHOST" ]; then
  PORT=${REDISPORT:-6379}
  if [ -n "$REDISPASSWORD" ]; then
    if [ -n "$REDISUSER" ]; then
      RECOMMENDED_URL="redis://$REDISUSER:$REDISPASSWORD@$REDISHOST:$PORT/0"
    else
      RECOMMENDED_URL="redis://:$REDISPASSWORD@$REDISHOST:$PORT/0"
    fi
  else
    RECOMMENDED_URL="redis://$REDISHOST:$PORT/0"
  fi
  echo -e "REDIS_URL: ${YELLOW}$RECOMMENDED_URL${NC} (recommended)"
else
  echo -e "${RED}Cannot generate a recommended Redis URL from available environment variables.${NC}"
  echo -e "${YELLOW}You need to manually configure Redis in your Railway project.${NC}"
fi

echo
echo -e "${GREEN}Instructions for fixing Redis connection in Railway:${NC}"
echo -e "1. Go to your Railway project dashboard"
echo -e "2. Add the following environment variables to your API service:"
echo -e "   - REDIS_URL = [The recommended URL above]"
echo -e "   - REDIS_DISABLE = false"
echo -e "3. If you prefer to use host/port configuration, set:"
echo -e "   - REDIS_HOST = [Your Redis hostname]"
echo -e "   - REDIS_PORT = [Your Redis port]"
echo -e "   - REDIS_PASSWORD = [Your Redis password, if any]"
echo -e "4. Redeploy your service for changes to take effect"
echo

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}Connection Test${NC}"
echo -e "${YELLOW}=========================================${NC}"

# Try to run a simple Redis connection test if redis-cli is available
if command -v redis-cli &> /dev/null; then
  echo -e "${GREEN}Testing Redis connection with redis-cli...${NC}"
  
  if [ -n "$REDISURL" ]; then
    echo -e "${YELLOW}Testing connection to $REDISURL${NC}"
    # Extract host, port, password from REDISURL
    REDIS_CLI_ARGS=""
    
    # Parse REDIS_URL for host and port
    HOST=$(echo $REDISURL | sed -r 's/redis:\/\/(.*@)?([^:]+):.*/\2/')
    PORT=$(echo $REDISURL | sed -r 's/.*:([0-9]+).*/\1/')
    
    # Check for password in URL
    if [[ "$REDISURL" == *"@"* ]]; then
      PASSWORD=$(echo $REDISURL | sed -r 's/redis:\/\/(.*):(.*)@.*/\2/')
      REDIS_CLI_ARGS="-h $HOST -p $PORT -a $PASSWORD"
    else
      REDIS_CLI_ARGS="-h $HOST -p $PORT"
    fi
    
    # Test connection
    redis-cli $REDIS_CLI_ARGS PING
    
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}Successfully connected to Redis!${NC}"
    else
      echo -e "${RED}Failed to connect to Redis.${NC}"
    fi
  elif [ -n "$REDISHOST" ]; then
    PORT=${REDISPORT:-6379}
    AUTH_ARGS=""
    [ -n "$REDISPASSWORD" ] && AUTH_ARGS="-a $REDISPASSWORD"
    
    echo -e "${YELLOW}Testing connection to $REDISHOST:$PORT${NC}"
    redis-cli -h $REDISHOST -p $PORT $AUTH_ARGS PING
    
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}Successfully connected to Redis!${NC}"
    else
      echo -e "${RED}Failed to connect to Redis.${NC}"
    fi
  else
    echo -e "${RED}Insufficient information to test Redis connection.${NC}"
  fi
else
  echo -e "${YELLOW}redis-cli not available - cannot perform connection test.${NC}"
  echo -e "${YELLOW}The application should attempt to connect to Redis on startup.${NC}"
fi

echo
echo -e "${GREEN}Script completed. Check application logs for Redis connection status.${NC}"
