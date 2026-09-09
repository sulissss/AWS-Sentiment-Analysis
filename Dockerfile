FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_PYTHON_REFRESH=quiet \
    NLTK_DATA=/usr/share/nltk_data

WORKDIR /app

# 1. Install build tools for C extensions (scipy, wordcloud)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install dependencies (ensure flask is in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Create persistent directories
RUN mkdir -p /usr/share/nltk_data artifacts/models uploads

# 4. Copy project source code and UI templates
COPY pipeline/ ./pipeline/
COPY data/ ./data/
COPY templates/ ./templates/
COPY app.py .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]