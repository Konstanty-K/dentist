# Uruchomienie Węzła Śledzenia (SAM3 Tracker) - Faza 1

Ten dokument opisuje, jak zintegrować i przetestować węzeł obniżający częstotliwość klatek (decimation) przed docelowym podpięciem modelu SAM3.

## Wymagania wstępne
* Działający kontener Docker (`ros2_lab`) z dostępem do kamery.
* Utworzone środowisko robocze ROS 2 (np. `~/ros2/ros2_ws`).

---

## Krok 1: Utworzenie nowej paczki ROS 2
Węzeł ten wymaga paczki napisanej w Pythonie. Zaloguj się do swojego kontenera i przejdź do katalogu `src` Twojego środowiska (workspace).

```bash
# Wejście do kontenera (jeśli jeszcze w nim nie jesteś)
docker exec -it ros2_lab bash

# Załadowanie środowiska
source /opt/ros/humble/setup.bash

# Przejście do katalogu src
cd /root/ros2/ros2_ws/src

# Utworzenie nowej paczki o nazwie 'vision_pipeline' z węzłem 'sam_tracker'
ros2 pkg create --build-type ament_python vision_pipeline --node-name sam_tracker