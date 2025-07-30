FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Emulate e2-micro resource constraints
ENV MEMORY_LIMIT=1024m
ENV CPU_LIMIT=1.0

WORKDIR /app

# Install system dependencies (minimal like e2-micro)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --disable-pip-version-check \
    --no-warn-script-location -r requirements.txt

# Download NLTK data BEFORE switching to non-root user
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader stopwords  
RUN python -m nltk.downloader wordnet

# Copy application code
COPY . .

# Create non-root user and fix NLTK data permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
RUN mkdir -p /home/appuser/nltk_data && \
    cp -r /root/nltk_data/* /home/appuser/nltk_data/ 2>/dev/null || \
    cp -r /usr/local/nltk_data/* /home/appuser/nltk_data/ 2>/dev/null || \
    true
RUN chown -R appuser:appuser /home/appuser/nltk_data
USER appuser

# Set NLTK data path for the non-root user
ENV NLTK_DATA=/home/appuser/nltk_data

# Expose port
EXPOSE 8080

# Run with resource constraints similar to e2-micro
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "60", "--worker-class", "gthread", "--max-requests", "100", "--max-requests-jitter", "10", "run:app"]