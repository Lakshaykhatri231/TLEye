# TLEye inference service.
#   docker build -t tleye .                          # CPU LightGBM verdict model (small image)
#   docker build -t tleye --build-arg WITH_LORA=1 .  # + CPU PyTorch for the LoRA model; mount the weights at run time:
#   docker run -p 8000:8000 -v "$PWD/models/lora:/app/models/lora:ro" tleye
#   --build-arg WITH_DEMO=1 and -e TLEYE_DEMO=1 also serve the web demo at / (API stays at /health and /check)
FROM python:3.12-slim

ARG WITH_LORA=0
ARG WITH_DEMO=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TLEYE_MODEL=auto
# libgomp: OpenMP runtime LightGBM needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY tleye ./tleye
COPY kaggle/lib ./kaggle/lib
COPY models ./models
RUN pip install -e ".[serve]" && \
    if [ "$WITH_LORA" = "1" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[train]"; \
    fi && \
    if [ "$WITH_DEMO" = "1" ]; then pip install -e ".[demo]"; fi

RUN useradd --create-home --uid 10001 tleye && chown -R tleye /app
USER tleye
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "tleye.serve:app", "--host", "0.0.0.0", "--port", "8000"]
