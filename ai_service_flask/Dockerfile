# Multi-stage build for production optimization
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r aiuser && useradd -r -g aiuser aiuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/aiuser/.local

# Copy application code
COPY --chown=aiuser:aiuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/cache && \
    chown -R aiuser:aiuser /app

# Switch to non-root user
USER aiuser

# Add local bin to PATH
ENV PATH=/home/aiuser/.local/bin:$PATH

# Expose ports for all AI services
EXPOSE 5000 5001 5002 5003 5004

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start the AI gateway (main entry point)
CMD ["python", "ai_gateway.py"] 