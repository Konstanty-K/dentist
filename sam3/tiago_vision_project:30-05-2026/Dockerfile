FROM ros:humble

# 1. Narzędzia systemowe, kamera i pakiety ROS2
RUN apt-get update && apt-get -y upgrade && apt-get -y install \
    ros-humble-desktop \
    ros-humble-usb-cam \
    ros-humble-cv-bridge \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Silnik Triton (konieczny do optymalizacji JIT)
RUN pip3 install --no-cache-dir triton

# 3. Instalacja natywnego PyTorch dla architektury Blackwell (RTX 50-series)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Modele i biblioteki pomocnicze ze sztywnymi wersjami
RUN pip3 install --no-cache-dir transformers typer python-dotenv "numpy<2.0.0" "Pillow>=10.0.0" "opencv-python<4.11.0"
