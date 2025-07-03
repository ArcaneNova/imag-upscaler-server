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

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Create directories and set permissions
RUN mkdir -p temp output weights logs \
    && chown -R appuser:appuser /app

# Copy application code and startup script
COPY . .
COPY start-api.sh .
RUN chmod +x start-api.sh

# Set ownership
RUN chown -R appuser:appuser /app

# Ensure app directory is recognized as a Python package
RUN test -f app/__init__.py || echo "# Python package" > app/__init__.py

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

# Expose port
EXPOSE 8000

# Startup script already copied and made executable above

# Default command for API server
CMD ["./start-api.sh"]
