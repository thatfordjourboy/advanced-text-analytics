# Use Python 3.11 slim image for Render compatibility
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy startup script and entrypoint
COPY startup.py .
COPY entrypoint.sh .

# Copy requirements
COPY requirements.txt .

# Copy application code
COPY app/ ./app/

# Create data and models directories
RUN mkdir -p data models

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application using the entrypoint script
CMD ["./entrypoint.sh"]
