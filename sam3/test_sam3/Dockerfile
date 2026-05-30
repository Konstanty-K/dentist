ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.9.1-devel-ubuntu24.04 AS base

FROM base AS base-amd64

ENV NV_CUDNN_VERSION=9.10.2.21-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12=${NV_CUDNN_VERSION}

FROM base AS base-arm64

ENV NV_CUDNN_VERSION=9.10.2.21-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-12=${NV_CUDNN_VERSION}

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="SAM3 Environment"
LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, CUDNN, and libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install SAM3 Core Dependencies (from pyproject.toml)
RUN pip install --no-cache-dir \
    "timm>=1.0.17" \
    "numpy==1.26.*" \
    tqdm \
    "ftfy==6.1.1" \
    regex \
    "iopath>=0.1.10" \
    typing_extensions \
    huggingface_hub

# Install additional dependencies
RUN pip install --no-cache-dir \
    opencv-python \
    pycocotools \
    matplotlib \
    scikit-image \
    einops \
    decord \
    pandas \
    typer \
    wandb

# Replace it after SAM3 official release
RUN pip install git+https://github.com/huggingface/transformers.git

RUN apt-get update && apt-get install -y python3-tk

RUN pip install seaborn

RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    onnx \
    pytorch_lightning \
    albumentations \
    scikit-learn \
    segmentation_models_pytorch \
    monai \
    onnxscript

# Uninstall opencv-python-headless if installed to avoid conflicts
RUN pip uninstall -y opencv-python-headless

# Create a workspace directory
RUN mkdir -p /root/workspace
WORKDIR /root/workspace

# Copy the project files into the container
COPY . /root/workspace

CMD ["/bin/bash"]