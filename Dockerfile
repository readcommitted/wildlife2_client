FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# Create non-root user
RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --gid 1001 --disabled-password --gecos "" appuser

COPY --chown=appuser:appgroup . .

# Pre-compile bytecode
RUN python -m compileall -q . && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]