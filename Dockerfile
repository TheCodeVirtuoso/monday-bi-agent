FROM python:3.13-slim

WORKDIR /app

# Dependencies first so a code change does not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Railway/Fly/Cloud Run inject $PORT; default to 8000 for a plain docker run.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
