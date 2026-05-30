# ROS 2 SAM3 Live Perception Node

Ten projekt integruje potężny model **Segment Anything Model 3 (SAM3)** z ekosystemem **ROS 2 Humble**. Węzeł subskrybuje surowy strumień z kamery, wykonuje inferencję (Open-Vocabulary Instance Segmentation) w czasie rzeczywistym i publikuje gotowe maski z powrotem do sieci ROS 2, umożliwiając ich podgląd w RViz2. 

Projekt został zaprojektowany z myślą o systemach wizyjnych dla robotów manipulacyjnych (np. Tiago Pro) i jest w pełni zoptymalizowany pod kątem najnowszych architektur NVIDIA (w tym serii RTX 50 / Blackwell `sm_120`), wykorzystując kompilację `torch.compile` (JIT) oraz środowisko CUDA 12.8.

## ⚙️ Wymagania wstępne

Zanim zaczniesz, upewnij się, że Twój system (Host) posiada:
* Zainstalowanego **Dockera** oraz **NVIDIA Container Toolkit** (aby kontener miał dostęp do GPU).
* Aktualne sterowniki NVIDIA obsługujące CUDA 12.8+.
* Podłączoną kamerę USB (domyślnie mapowaną jako `/dev/video0`).
* Wygenerowany token dostępu do Hugging Face (wymagany do pobrania wag modelu SAM3).

---

## 🚀 Instalacja i Uruchomienie

### 1. Klonowanie repozytorium
Pobierz projekt na swój dysk i przejdź do jego folderu:
```bash
git clone <adres_twojego_repozytorium>
cd tiago_vision_project
2. Budowanie obrazu Docker
Zbuduj dedykowane środowisko. Proces ten pobierze ROS 2, PyTorch oraz niezbędne biblioteki (może to zająć kilka minut):

Bash
docker build -t ros2_sam3_vision .
3. Uruchomienie kontenera
Aby kontener mógł wyświetlać interfejs graficzny (RViz2) oraz miał dostęp do sprzętu (kamera i GPU), uruchom go za pomocą poniższej komendy.

Ważne: Podmień TWÓJ_TOKEN na swój prawdziwy klucz z Hugging Face!

Bash
xhost +local:root && docker run -it \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="HF_TOKEN=TWÓJ_TOKEN" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --device="/dev/video0:/dev/video0" \
    -v "$PWD:/workspace" \
    -w /workspace \
    --net=host \
    --privileged \
    --name=tiago_vision_container \
    ros2_sam3_vision:latest bash
🖥️ Procedura testowa (System 3 Terminali)
Aby uruchomić pełny potok przetwarzania wizji, musisz otworzyć 3 osobne okna terminala. Wszystkie muszą być podłączone do działającego kontenera.

W każdym nowym oknie na hoście wpisz najpierw:

Bash
docker exec -it tiago_vision_container bash
source /opt/ros/humble/setup.bash
cd /workspace
Następnie uruchom poszczególne węzły:

Terminal 1: Strumień z kamery
Uruchom węzeł kamery, który zacznie publikować obraz na temacie /image_raw:

Bash
ros2 run usb_cam usb_cam_node_exe
Terminal 2: Węzeł Sztucznej Inteligencji (SAM3)
Uruchom główny skrypt percepcji. Domyślnie uruchomi się on w trybie natychmiastowym:

Bash
python3 sam3_live_node.py
🔥 Tryb wysokiej wydajności (JIT): Jeśli chcesz uzyskać więcej klatek na sekundę, użyj flagi włączającej kompilator. Pierwsza klatka zablokuje system na około minutę (optymalizacja kerneli), ale kolejne będą działać znacznie szybciej:

Bash
ros2 run robot_vision sam3_live_node --ros-args -p use_compile:=True
Terminal 3: Wizualizacja w RViz2
Uruchom środowisko graficzne:

Bash
rviz2
W RViz2:

Kliknij Add (w lewym dolnym rogu).

Wybierz zakładkę By topic.

Znajdź /sam3/annotated_image, wybierz Image i kliknij OK.

(Opcjonalnie) Jeśli obraz jest czarny, rozwiń ustawienia dodanego obrazu i zmień Reliability Policy na Best Effort.

⚙️ Konfiguracja (Co wykrywamy?)
Prompty (czyli to, czego model szuka na obrazie) są zdefiniowane w pliku queries.json. Możesz je edytować w locie bez konieczności restartowania lub przebudowywania Dockera.

Przykład:

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
Zapisanie tego pliku na hoście natychmiast udostępni go wewnątrz kontenera.
