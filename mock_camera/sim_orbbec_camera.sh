#!/bin/bash
# ==========================================================
# Czysty Symulator Kamery Orbbec (JAZZY - Natywny MCAP)
# ==========================================================

ROSBAG_DIR="$HOME/robotics/orbbec/shared/bag_001"
DDS_XML="$HOME/robotics/dentist/fastdds_no_shm.xml"
ORIGINAL_TOPIC="/camera/color/image_raw"

echo "Uruchamiam kontener symulacyjny (ROS 2 Jazzy)..."
echo "Plik źródłowy: $ROSBAG_DIR"

# Używamy natywnego obrazu Jazzy (brak konieczności pobierania wtyczek!)
docker run --interactive --tty --rm \
    --network host \
    --volume "$ROSBAG_DIR":/bag_data:ro \
    --volume "$DDS_XML":/fastdds_no_shm.xml:ro \
    --env FASTRTPS_DEFAULT_PROFILES_FILE=/fastdds_no_shm.xml \
    --name mock_orbbec_camera \
    ros:jazzy-ros-base \
    bash -c "source /opt/ros/jazzy/setup.bash && echo 'Rozpoczynam nadawanie strumienia...' && ros2 bag play /bag_data --loop --remap $ORIGINAL_TOPIC:=/image_raw"
