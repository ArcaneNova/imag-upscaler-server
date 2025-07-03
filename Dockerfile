# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies including Redis
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    redis-server \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Create directories and set permissions
RUN mkdir -p temp output weights logs /var/lib/redis /var/log/redis \
    && chown -R appuser:appuser /app /var/lib/redis /var/log/redis \
    && chmod 755 /var/lib/redis /var/log/redis \
    && chmod 755 /usr/bin/redis-server /usr/bin/redis-cli

# Copy application code including startup script
COPY . .
RUN chmod +x start-api.sh entrypoint.sh

# Set ownership
RUN chown -R appuser:appuser /app /var/lib/redis /var/log/redis

# Ensure app directory is recognized as a Python package
RUN test -f app/__init__.py || echo "# Python package" > app/__init__.py

# Ensure main.py exists and is properly set up (skipping if it already exists)
RUN test -f main.py || echo '"""Import redirector for app.main"""\nfrom app.main import app' > main.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV DOCKER_CONTAINER=1

# Debug the Python path
RUN python -c "import sys; print('Python path:', sys.path)"
RUN python -c "import os; print('App files:', os.listdir('/app/app'))"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 6379

# Startup script already copied and made executable above

# Default command for API server
CMD ["./entrypoint.sh"]
