FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_PYTHON_REFRESH=quiet \
    NLTK_DATA=/usr/share/nltk_data \
    PYTHONPATH=/app

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /usr/share/nltk_data artifacts/models uploads

# Copy project files
COPY pipeline/ ./pipeline/
COPY data/ ./data/
COPY templates/ ./templates/
COPY app.py .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]