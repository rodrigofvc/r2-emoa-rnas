# CUDA 12.1 + cuDNN 8 (driver 580 compatible)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN set -eux; \
    apt-get -o Acquire::AllowInsecureRepositories=true update; \
    apt-get -o Acquire::AllowInsecureRepositories=true \
            -o APT::Get::AllowUnauthenticated=true \
            install -y --no-install-recommends \
                python3 \
                python3-venv \
                python3-pip \
                python3-dev \
                build-essential \
                git \
                ca-certificates; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy==1.26.4"

RUN pip install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "scipy==1.13.1" \
    "pymoo==0.6.1.5" \
    "scikit-learn==1.4.2" \
    "pandas==2.2.2" \
    "matplotlib==3.8.4" \
    "pillow==10.3.0" \
    "tqdm==4.66.4" \
    "torchattacks==3.5.1" \
    "kiwisolver==1.4.5" \
    "torchprofile==0.0.4" \
    "torchmetrics==0.11.4"

ENV PYTHONFAULTHANDLER=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    TORCH_NUM_THREADS=1

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]

CMD ["python3", "-c", "import torch, numpy as np; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version(), 'avail:', torch.cuda.is_available())"]
