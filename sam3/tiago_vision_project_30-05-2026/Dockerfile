# ROS 2 Environment with NVIDIA CUDA & PyTorch Optimization
# Konstanty Kaszubski

FROM ros:humble

# 1. System tools, USB camera drivers, and ROS2 CV Bridge
RUN apt-get update && apt-get -y upgrade && apt-get -y install \
    ros-humble-desktop \
    ros-humble-usb-cam \
    ros-humble-cv-bridge \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Triton Engine (required for PyTorch JIT compilation)
RUN pip3 install --no-cache-dir triton

# 3. Native PyTorch installation for Blackwell architecture (RTX 50-series)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Vision models and supporting libraries with strict versioning
RUN pip3 install --no-cache-dir transformers typer python-dotenv "numpy<2.0.0" "Pillow>=10.0.0" "opencv-python<4.11.0"
