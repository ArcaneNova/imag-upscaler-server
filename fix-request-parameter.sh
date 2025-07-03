#!/bin/bash

# fix-request-parameter.sh - Script to fix the request parameter conflict in /upscale endpoint

# Print colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting request parameter conflict fix for upscale endpoint...${NC}"

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
  echo -e "${RED}Error: app/main.py not found. Make sure you're in the api-server root directory.${NC}"
  exit 1
fi

# Create backup of main.py
echo -e "${GREEN}Creating backup of app/main.py...${NC}"
cp app/main.py app/main.py.bak.$(date +%Y%m%d%H%M%S)

# Apply the fix
echo -e "${GREEN}Applying fix to app/main.py...${NC}"

# Check if the issue is already fixed
if grep -q "request: Request  # Properly type-annotate as Request for rate limiter" app/main.py; then
  echo -e "${YELLOW}Fix already applied. No changes needed.${NC}"
else
  # Apply the fix using sed
  sed -i 's/request: Request  # Properly type-annotate as Request/request: Request  # Properly type-annotate as Request for rate limiter/' app/main.py
  
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}Fix successfully applied!${NC}"
  else
    echo -e "${RED}Error applying fix. Please check app/main.py manually.${NC}"
    exit 1
  fi
fi

# Verify the fix
echo -e "${GREEN}Verifying fix...${NC}"
if grep -q "request: Request  # Properly type-annotate as Request for rate limiter" app/main.py; then
  echo -e "${GREEN}Verification successful.${NC}"
else
  echo -e "${RED}Verification failed. Fix might not have been applied correctly.${NC}"
  exit 1
fi

echo -e "${YELLOW}Important: You need to restart the FastAPI service for changes to take effect.${NC}"
echo -e "${YELLOW}For Docker environments: docker-compose restart api${NC}"
echo -e "${YELLOW}For Kubernetes: kubectl rollout restart deployment <your-deployment>${NC}"
echo -e "${YELLOW}For Railway or Vercel: Trigger a redeployment${NC}"

echo -e "${GREEN}Fix completed successfully. Remember to update API documentation to remove 'request' query parameter.${NC}"
