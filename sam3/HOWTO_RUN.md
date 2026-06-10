# 🚀 Instrukcja uruchomienia: Węzeł SAM3 (Zero-Shot Tracking)

Poniższy poradnik przeprowadza przez proces uruchomienia środowiska sztucznej inteligencji, podpięcia źródła obrazu i wizualizacji wygenerowanych masek. Całość działa w architekturze ROS 2 Humble.

**Ważne przed startem:** Upewnij się, że w folderze `sam3` znajduje się Twój prywatny plik `.env` z kluczem Hugging Face (`HF_TOKEN=...`) oraz odchudzony plik `queries.json` (zawierający maksymalnie 2-3 narzędzia dla zachowania płynności FPS).

---

## Krok 1: Inicjalizacja środowiska AI (Terminal 1)

W pierwszej kolejności budujemy i podnosimy ciężki kontener obliczeniowy, upewniając się, że ma dostęp do klucza HF oraz profilu sieciowego.

1. Przejdź do folderu modułu SAM3:
   ```bash
   cd ~/robotics/dentist/sam3

```

2. Zbuduj obraz (wymagane tylko przy pierwszej instalacji lub po zmianie `Dockerfile`):
```bash
docker build -t tiago_vision_container .

```


3. Uruchom kontener z odpowiednim mapowaniem sprzętu i plików (podmontuje to obecny folder do `/workspace`):
```bash
xhost +local:root
docker run -it --rm --gpus all --net host --privileged --env-file .env -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=$DISPLAY -v $(pwd):/workspace --name tiago_vision tiago_vision_container bash

```


4. Wewnątrz kontenera skompiluj i załaduj środowisko ROS 2:
```bash
cd /workspace/ros2_ws
colcon build --packages-select vision_pipeline
source /opt/ros/humble/setup.bash
source install/setup.bash

```


5. **[KRYTYCZNE]** Załaduj profil FastDDS i wystartuj węzeł ze wskazanym słownikiem:
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/moje_fastdds.xml
ros2 run vision_pipeline sam_tracker --ros-args -p query_file:="/workspace/queries.json"

```



*Czekaj na dzwonek i komunikat: `🔔 MODEL GOTOWY!*`

---

## Krok 2: Uruchomienie strumienia wideo (Terminal 2)

Otwórz drugi, natywny terminal w systemie Ubuntu. Jako źródło obrazu musisz wybrać jedną z dwóch ścieżek: używasz symulatora **albo** sprzętu fizycznego.

**Opcja A: Symulator (mock_camera / rosbag)**
Jeśli nie masz podłączonego robota, uruchom skrypt odtwarzający nagranie z sąsiedniego katalogu:

```bash
cd ~/robotics/dentist/mock_camera
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=../sam3/moje_fastdds.xml

./sim_orbbec_camera.sh

```

**Opcja B: Kamera fizyczna (np. Orbbec / USB)**
Podłącz urządzenie i uruchom standardowy węzeł kamery:

```bash
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=~/robotics/dentist/sam3/moje_fastdds.xml

ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:="/dev/video0"

```

---

## Krok 3: Wizualizacja efektów w czasie rzeczywistym (Terminal 3)

Ostatni krok to podgląd wyników na natywnej instalacji ROS 2 na Ubuntu.

1. Uruchom interfejs RViz2:
```bash
source /opt/ros/humble/setup.bash
rviz2

```


2. W lewym dolnym rogu kliknij **Add** -> **By topic** i wybierz `/sam3/smoothed_mask`.
3. W panelu po lewej stronie, dla dodanego obrazu zmień parametr **Reliability Policy** na **Best Effort**, aby uniknąć problemu z brakiem klatek ("No Image").

**Gotowe!** Model analizuje obraz i nakłada surowe maski na obiekty. Z uwagi na zoptymalizowaną heurystykę dla powolnych modeli (Grace Period = 0), maski odzwierciedlają twardy stan faktyczny z danej mikrosekundy – znikną natychmiast, gdy narzędzie medyczne zostanie zasłonięte przez dłoń operatora lub usunięte z kadru.

