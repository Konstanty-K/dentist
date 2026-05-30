# ROS 2 SAM3 Live Perception Node

Ten projekt integruje model **Segment Anything Model 3 (SAM3)** z ekosystemem **ROS 2 Humble**. Węzeł subskrybuje surowy strumień z kamery, wykonuje inferencję (Open-Vocabulary Instance Segmentation) w czasie rzeczywistym i publikuje gotowe maski z powrotem do sieci ROS 2, umożliwiając podgląd w RViz2.

Projekt został stworzony z myślą o systemach wizyjnych dla robotów manipulacyjnych (np. Tiago Pro) i jest w pełni zoptymalizowany pod kątem najnowszych architektur NVIDIA (Blackwell / RTX 50-series) z wykorzystaniem środowiska CUDA 12.8 i kompilacji JIT.

## ⚙️ Wymagania wstępne
* Zainstalowany **Docker** oraz **NVIDIA Container Toolkit**.
* Sterowniki NVIDIA obsługujące CUDA 12.8+.
* Podłączona kamera USB.
* Wygenerowany token dostępu do Hugging Face (z uprawnieniami *Read*). Należy upewnić się, że zaakceptowano licencję modelu SAM3 na platformie HF.

---

## 🚀 Instalacja i Uruchomienie

### 1. Budowanie obrazu Docker
Pobierz repozytorium, przejdź do folderu głównego i zbuduj środowisko:
```bash
docker build -t ros2_sam3_vision .
2. Uruchomienie kontenera (Główny Serwer)
Poniższa komenda automatycznie podłącza obecny katalog (dzięki zmiennej $PWD), mapuje wszystkie potencjalne porty kamer wideo na hoście i uruchamia instancję.

Ważne: Ze względów cyberbezpieczeństwa nie używamy plików .env. Podmień TWÓJ_TOKEN na swój prawdziwy klucz Hugging Face bezpośrednio w komendzie!

Bash
xhost +local:root && docker run -it \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="HF_TOKEN=TWÓJ_TOKEN" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --device=/dev/video0 \
    --device=/dev/video1 \
    --device=/dev/video2 \
    --device=/dev/video3 \
    -v "$PWD:/workspace" \
    -w /workspace \
    --net=host \
    --privileged \
    --name=tiago_vision_container \
    ros2_sam3_vision:latest bash
🖥️ Procedura testowa (System 3 Terminali)
Aby uruchomić pełny potok przetwarzania wizji, musisz otworzyć 3 osobne okna terminala na hoście.

W KAŻDYM nowym oknie najpierw wejdź do kontenera i aktywuj środowisko ROS 2:

Bash
docker exec -it tiago_vision_container bash
source /opt/ros/humble/setup.bash
cd /workspace
Następnie uruchom poszczególne węzły:

Terminal 1: Strumień z kamery
Uruchom węzeł kamery. W środowiskach Linux fizyczna kamera sprzętowa jest często mapowana do /dev/video2 zamiast domyślnego video0. Uruchom węzeł z właściwym portem:

Bash
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video2
(Uwaga: Jeśli dioda kamery się nie zapali, przerwij komendę (Ctrl+C) i przetestuj /dev/video0).

Terminal 2: Węzeł Sztucznej Inteligencji (SAM3)
Uruchom główny skrypt percepcji. Domyślnie odpala się w trybie standardowym:

Bash
python3 sam3_live_node.py
🔥 Tryb wysokiej wydajności (JIT): Aby wycisnąć maksymalne FPS dla kart sprzętowych Nvidii, uruchom optymalizator flagą:

Bash
ros2 run robot_vision sam3_live_node --ros-args -p use_compile:=True
Skrypt poinformuje Cię potrójnym sygnałem dźwiękowym z terminala, gdy 3.5 GB wag modelu załaduje się do pamięci VRAM.

Terminal 3: Wizualizacja w RViz2
Uruchom środowisko graficzne:

Bash
rviz2
W RViz2:

Kliknij Add -> By topic.

Wybierz /sam3/annotated_image -> Image.

⚙️ Konfiguracja Słownika (Wykrywanie Obiektów)
Prompty (czyli to, co algorytm ma wykrywać na obrazie) znajdują się w pliku queries.json. Edytuj go na żywo z poziomu hosta bez konieczności restartowania systemu, aby natychmiast zaktualizować filtry węzła:

JSON
{
    "a surgical tool": {
        "confidence": 0.5,
        "mask_confidence": 0.5
    },
    "a robotic hand": {
        "confidence": 0.4,
        "mask_confidence": 0.5
    }
}

Konstanty Kaszubski
