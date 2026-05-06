# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first so subsequent `pip install -r requirements.txt`
# sees torch is already satisfied (the requirements pin is `torch>=2.0.0`).
# CPU wheel is ~200 MB vs ~2.5 GB for the default CUDA wheel.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy project sources after deps so code changes don't bust the pip cache layer.
COPY pytest.ini ./
COPY tests/ ./tests/
COPY code/ ./code/
COPY models/ ./models/

# Default command runs the test suite. Override at `docker run` time to do anything else.
CMD ["pytest", "-v"]
