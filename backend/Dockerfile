FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured + psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev gcc libmagic1 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create upload directory
RUN mkdir -p uploads chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
