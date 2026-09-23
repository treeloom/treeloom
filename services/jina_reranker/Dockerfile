# GPU reranker service for models TEI can't host (jina-reranker-v2-*).
# torch + CUDA ship in the base image so we don't recompile them.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HF cache on the mounted volume so the model is downloaded once.
ENV HF_HOME=/data
ENV MODEL_ID=jinaai/jina-reranker-v2-base-multilingual

EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
