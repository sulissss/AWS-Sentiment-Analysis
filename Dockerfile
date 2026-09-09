FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/usr/share/nltk_data

WORKDIR /app

# 1. Install build tools for C extensions (scipy, wordcloud)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install dependencies (ensure 'requests' is listed in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Create persistent directory for NLTK datasets downloaded at runtime
RUN mkdir -p /usr/share/nltk_data

# 4. Copy project files and entrypoint
COPY pipeline/ ./pipeline/
COPY data/ ./data/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]