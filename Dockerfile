# CUDA 12.1 + cuDNN 8 (driver 580 compatible)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata software-properties-common ca-certificates curl git build-essential \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils \
 && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv \
 && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel


ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    PYTHONFAULTHANDLER=1

RUN pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Crear usuario no-root y definir directorio de trabajo
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /workspace


CMD ["python3", "-c", "import torch, numpy as np; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version(), 'avail:', torch.cuda.is_available())"]
