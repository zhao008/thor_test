FROM nvcr.io/nvidia/cuda:13.1.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    wget \
    git \
    vim \
    build-essential \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone cutile-python and install dependencies
RUN git clone https://github.com/NVIDIA/cutile-python.git /workspace/cutile-python

WORKDIR /workspace/cutile-python

RUN pip install cuda-tile --break-system-packages && \
    pip install -r test/requirements.txt --break-system-packages && \
    pip install ./experimental --break-system-packages

# Set working directory back to workspace root
WORKDIR /workspace