FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index from whatever is mounted at /app/data/reports at build time.
# For a real deployment, mount reports as a volume and run ingest at container start instead.
RUN python cli.py ingest --reports-dir data/sample_reports --index-dir index

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
