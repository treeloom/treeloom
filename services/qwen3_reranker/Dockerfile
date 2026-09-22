# GPU reranker service for Qwen3-Reranker models (TEI can't host them).
# torch + CUDA ship in the base image so we don't recompile them.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# torch.compile (inductor/triton) JIT-compiles CUDA kernels at runtime and needs
# a host C compiler — the -runtime base image ships without one. Without this,
# the first compiled forward raises "Failed to find C compiler".
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HF cache on the mounted volume so the model is downloaded once.
ENV HF_HOME=/data
ENV MODEL_ID=tomaarsen/Qwen3-Reranker-0.6B-seq-cls

EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
