#!/bin/bash
# ==========================================================
# Skrypt symulujący żywą kamerę z nagrania Orbbec Femto Bolt
# Moduł R&D - Tiago Pro Surgical Grasping
# ==========================================================

# 1. USTAWIENIA (Podmień ścieżkę na tę, gdzie leży Twój folder z nagraniem)
ROSBAG_PATH="/home/user/robotics/orbbec/shared/bag_001"

# Domyślny temat obrazu RGB w Orbbec Femto Bolt (sprawdź przez 'ros2 bag info' jeśli jest inny)
ORIGINAL_TOPIC="/camera/color/image_raw" 

echo "Uruchamiam symulator kamery..."
echo "Plik źródłowy: $ROSBAG_PATH"
echo "Przekierowuję temat: $ORIGINAL_TOPIC -> /image_raw"

# 2. Zabezpieczenie sieciowe (wyłączenie SHM), aby kontenery się widziały
export FASTRTPS_DEFAULT_PROFILES_FILE="$(pwd)/fastdds_no_shm.xml"

# 3. Załadowanie środowiska ROS 2 (zakładając, że odpalasz z hosta/bazowego kontenera)
source /opt/ros/humble/setup.bash

# 4. Odtwarzanie w pętli (--loop) z podmianą tematu (--remap)
ros2 bag play "$ROSBAG_PATH" --loop --remap "$ORIGINAL_TOPIC":="/image_raw"
