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
RUN mkdir -p temp output weights logs \
    && chown -R appuser:appuser /app \
    && chmod 755 /usr/bin/redis-server /usr/bin/redis-cli

# Copy application code including startup script
COPY . .
RUN chmod +x start-api.sh entrypoint.sh && \
    ls -la entrypoint.sh && \
    head -5 entrypoint.sh

# Set ownership
RUN chown -R appuser:appuser /app

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

# Debug the Python path and files
RUN python -c "import sys; print('Python path:', sys.path)" || echo "Python path check failed"
RUN ls -la /app/ || echo "App directory listing failed"
RUN ls -la /app/app/ || echo "App/app directory listing failed"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ping || exit 1

# Expose ports
EXPOSE 8000 6379

# Startup script already copied and made executable above

# Default command for API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
