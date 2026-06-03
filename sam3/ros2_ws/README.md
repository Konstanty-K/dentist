#tiago #sam3 #ros2 #docker #project 

# Węzeł Wizyjny SAM3 - Tiago Pro Surgical Grasping

Ten moduł implementuje zero-shotową segmentację instancji z wykorzystaniem modelu Meta SAM3. Węzeł wykorzystuje algorytm **Smart FPS (Async Drop)**, który asynchronicznie odrzuca nadmiarowe klatki, zapobiegając opóźnieniom (lagom) pamięci RAM, oraz heurystykę Grace Period do stabilizacji masek podczas chwilowych okluzji. Ustabilizowane maski narzędzi medycznych są publikowane do środowiska ROS 2 w celu późniejszego wyznaczenia punktów chwytu przez robota Tiago Pro.

**Autor:** Konstanty Kaszubski

---

## 1. Zabezpieczenia i Konfiguracja (Krok Wstępny)

Moduł SAM3 wymaga dostępu do serwerów Hugging Face w celu autoryzacji modelu wagi ciężkiej. **Nigdy nie wpisuj tokena bezpośrednio w terminalu ani w kodzie!**

1. W głównym katalogu projektu utwórz ukryty plik `.env`:
   

```Bash
   touch .env
```

2. Wklej do niego swój token dostępu (z uprawnieniami Read dla modeli Meta):
    
    ```Bash
    HF_TOKEN=hf_TwojPrawdziwyTokenTutaj12345
    ```
    

_(Plik `.env` jest dodany do `.gitignore`, więc nie ma ryzyka, że wycieknie na GitHuba)._

## 2. Instalacja i Środowisko Docker (Pierwsze Uruchomienie)

Zakładając, że to świeża instalacja systemu, musisz zbudować obraz i uruchomić wyizolowany kontener z dostępem do GPU.

**Budowa obrazu:**

```Bash
docker build --tag tiago_vision_container --file Dockerfile .
```

**Uruchomienie kontenera (z bezpiecznym ładowaniem zmiennych z `.env`):**

```Bash
xhost +local:root
docker run --interactive --tty --rm \
    --gpus all \
    --network host \
    --privileged \
    --env-file .env \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --env DISPLAY=$DISPLAY \
    --volume $(pwd):/workspace \
    --name tiago_vision \
    tiago_vision_container bash
```

## 3. Codzienna Praca w Laboratorium

Jeśli masz już zbudowany obraz, a kontener "śpi", po prostu go obudź i wejdź do środka:

```Bash
docker start tiago_vision
docker exec --interactive --tty tiago_vision bash
```

### Budowanie paczki ROS 2 (Collective Construction)

Przy każdej zmianie w kodzie Pythona, przebuduj przestrzeń roboczą używając flagi `--symlink-install` (dzięki niej zmiany w kodzie są widoczne natychmiast, bez konieczności ponownego budowania):

```****
source /opt/ros/humble/setup.bash
cd /workspace/sam3/ros2_ws
colcon build --packages-select vision_pipeline --symlink-install
```

## 4. Uruchomienie Węzła SAM3

Aby uniknąć problemów z przesyłaniem dużych klatek wideo pomiędzy odizolowanymi kontenerami (błąd pamięci współdzielonej SHM), węzeł wymaga wczytania specjalnego profilu sieciowego.

Wewnątrz działającego kontenera wykonaj następujące kroki:

```Bash
# 1. Załaduj środowisko nowo zbudowanej paczki
source /workspace/sam3/ros2_ws/install/setup.bash

# 2. Wyłącz pamięć współdzieloną (SHM) dla DDS - KRYTYCZNE DLA OBRAZU!
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_no_shm.xml

# 3. Uruchom tracking
ros2 run vision_pipeline sam_tracker
```

Węzeł zasubskrybuje domyślnie temat `/image_raw` i opublikuje nałożone maski na `/sam3/smoothed_mask`.

## 5. Podpinanie Strumienia Wideo (Kamery i Rosbagi)

Węzeł wizyjny jest zaprojektowany jako czarna skrzynka (Interface-Driven) – można do niego podpiąć dowolne źródło nadające wiadomość `sensor_msgs/Image`.

**Opcja A: Odtwarzanie nagrania (Orbbec Femto Bolt):** W osobnym oknie hosta:

```Bash
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_no_shm.xml
ros2 bag play /ścieżka/do/nagrania/rosbag --loop
```

**Opcja B: Kamera wbudowana/USB:** W kontenerze z narzędziami `usb_cam`:

```Bash
source /opt/ros/humble/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_no_shm.xml
ros2 run usb_cam usb_cam_node_exe --ros-args --param video_device:="/dev/video0"
```