FROM python:3.11-slim
WORKDIR /app

# System deps for geopandas/fiona (if used) can be heavy; keep minimal for demo
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY dev-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.headless=true"]
