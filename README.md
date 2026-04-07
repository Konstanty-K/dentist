# TiagoProDentist - Tool Perception System

Projekt wykrywania i pozycjonowania narzędzi stomatologicznych w przestrzeni 3D przy użyciu modelu **YOLOv8** oraz systemu **ROS 2 Jazzy**. System przetwarza dane z kamery RGB-D, publikując obraz z detekcjami oraz chmurę punktów wykrytych obiektów.

## 🛠️ Wymagania systemowe
* **System:** Ubuntu 24.04 (Noble Numbat).
* **Docker & Docker Compose** zainstalowane na hoście.
* **Sterowniki NVIDIA** (wymagane dla akceleracji GPU wewnątrz kontenera).
* **ROS 2 Jazzy** zainstalowany lokalnie (do uruchomienia RViz2 na hoście).

## 📁 Struktura projektu
* \`scripts/yolo_to_rviz.py\`: Główny węzeł ROS 2 integrujący YOLO z danymi Depth.
* \`weights/best.pt\`: Wytrenowane wagi modelu YOLO.
* \`bag/\`: Folder z nagraniami (RGB i Depth).
* \`fastdds_no_shm.xml\`: Konfiguracja DDS eliminująca błędy przesuwu dużych danych (No Shared Memory).

## 🚀 Szybki start

### 1. Budowanie i uruchomienie kontenera
Będąc w głównym folderze projektu, wykonaj:
`docker compose build`
`docker compose up -d`


### 2. Uruchomienie węzła percepcji
1. Wejdź do kontenera i odpal skrypt:
	* `docker exec -it TiagoProDentist bash`
	* `source /opt/ros/jazzy/setup.bash`
	* `python3 scripts/yolo_to_rviz.py`

### 3. Wizualizacja w RViz2 (Na Hoście)
Aby zobaczyć wyniki, otwórz nowy terminal na Ubuntu i:
1. Uruchom publikator statycznego układu współrzędnych:
   * `ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map camera_color_optical_frame`
2. Uruchom RViz2:
   * `rviz2`
3. W RViz skonfiguruj:
   * **Fixed Frame:** \`map\`
   * **Image Topic:** \`/detected_tool_image\` (Reliability: Reliable)
   * **PointCloud2 Topic:** \`/detected_tool_pc\` (Reliability: Reliable)

## ⚠️ Rozwiązywanie problemów
1. **Brak obrazu w RViz:** Upewnij się, że na hoście i w Dockerze ustawiono zmienną środowiskową dla FastDDS: 
	* `export FASTRTPS_DEFAULT_PROFILES_FILE=/Shared/fastdds_no_shm.xml`.
2. **Błąd rclpy:** Pamiętaj o wykonaniu wewnątrz kontenera przed uruchomieniem skryptu
	* `source /opt/ros/jazzy/setup.bash`
