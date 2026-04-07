# Bazujemy na oficjalnym obrazie ROS 2 Jazzy
FROM osrf/ros:jazzy-desktop

# Zapobieganie interaktywnym pytaniom podczas instalacji
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalacja zależności systemowych (Zaktualizowane nazwy dla Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-yaml \
    ros-jazzy-cv-bridge \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalacja bibliotek Python
# Wymuszamy numpy < 2.0.0, aby cv_bridge działał poprawnie
RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy<2.0.0" \
    ultralytics \
    opencv-python

# 3. Konfiguracja środowiska ROS
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

WORKDIR /Shared

# Automatyczne ładowanie środowiska ROS 2 przy każdym wejściu do kontenera
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
