FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install production Python dependencies.
COPY requirements-prod.txt /app/requirements-prod.txt

RUN pip install --upgrade pip \
    && pip install -r /app/requirements-prod.txt \
    && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 \
        torchvision==0.29.1

# Copy application code.
COPY src /app/src
COPY api /app/api
COPY auth /app/auth
COPY config /app/config
COPY database /app/database
COPY services /app/services
COPY web /app/web
COPY templates /app/templates
COPY static /app/static

# Copy Alembic configuration/migrations.
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic

# Copy the frozen production model.
COPY artifacts/mlflow_baseline/best_model.pt \
     /app/artifacts/mlflow_baseline/best_model.pt

# Create an unprivileged application user.
RUN useradd \
        --create-home \
        --shell /usr/sbin/nologin \
        appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]