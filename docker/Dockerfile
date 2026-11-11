# Build after `bsmguard study` so models/detector.pt exists.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg BSMGUARD_MODEL=/app/models/detector.pt
WORKDIR /app
COPY requirements.txt .
# CPU-only torch wheel (the default PyPI wheel pulls ~2 GB of CUDA libraries)
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY models ./models
RUN pip install --no-deps . && useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "bsmguard.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
