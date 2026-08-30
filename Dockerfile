FROM python:3.13-slim

# Hugging Face Spaces runs containers as a non-root user, so create one and
# own the app directory. Harmless everywhere else (Render, Railway, Fly, a
# plain `docker run`) and better practice regardless.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Dependencies first so a code change does not invalidate the pip layer.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser
ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1

# HF Spaces reads app_port from README front matter (8000). Render, Railway
# and Fly inject $PORT instead, so honour that when present.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
