# Instrukcja uruchomienia: Wielki Finał (Śledzenie SAM3 2D)

Poniższy poradnik przeprowadzi Cię krok po kroku przez proces uruchomienia węzłów ROS 2 odpowiedzialnych za pobieranie obrazu z kamery i nakładanie na niego inteligentnych masek przez sieć neuronową SAM3.

## Krok 1: Przygotowanie fizyczne
1. Upewnij się, że kamera jest podłączona, a jej obiektyw nie jest niczym zasłonięty.
2. Połóż na stole przed kamerą narzędzie chirurgiczne (np. nożyczki, strzykawkę), które znajduje się w słowniku `queries.json`.

## Krok 2: Uruchomienie strumienia wideo (Terminal 1)
Otwórz pierwszy terminal na swoim laptopie i uruchom podstawowy kontener z systemem ROS 2, aby aktywować węzeł kamery.

1. Wejdź do środowiska roboczego:
   `docker exec -it ros2_lab bash`
2. Załaduj zmienne systemowe ROS 2:
   `source /opt/ros/humble/setup.bash`
3. Uruchom kamerę na odpowiednim porcie (zwróć uwagę, czy zapaliła się dioda obok obiektywu):
   `ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:="/dev/video2"`
   *(Zostaw ten terminal otwarty, aby kamera nadawała obraz w tle).*

## Krok 3: Uruchomienie modułu sztucznej inteligencji (Terminal 2)
Otwórz drugi, zupełnie nowy terminal. Tutaj uruchomimy cięższy kontener (AI), który ma dostęp do karty graficznej (GPU) i potrafi przetworzyć obraz.

1. Przejdź do folderu modułu SAM3:
   `cd ~/robotics/dentist/sam3`
2. Uruchom kontener z modelem AI (upewnij się, że uruchamiasz go z flagą sieci hosta, np. `--net=host`, aby widział kamerę z pierwszego kontenera).
3. Wejdź do folderu z paczką i skompiluj ją:
   `cd /workspace/ros2_ws` *(Zmień /workspace/ na właściwą ścieżkę mapowania, jeśli jest inna)*
   `colcon build --packages-select vision_pipeline`
4. Załaduj nowo zbudowane środowisko:
   `source install/setup.bash`
5. Uruchom węzeł śledzący:
   `ros2 run vision_pipeline sam_tracker`

*Poczekaj chwilę. W terminalu usłyszysz sygnał dźwiękowy (dzwonek), a w logach pojawi się informacja: `🔔 MODEL GOTOWY! Wagi załadowane do VRAM.` To oznacza, że sieć działa i przetwarza klatki.*

## Krok 4: Wizualizacja efektów (Terminal 3)
Otwórz trzeci terminal (może być natywnie na systemie Ubuntu lub w lekkim kontenerze `ros2_lab`) i uruchom interfejs graficzny:

1. Wpisz komendę:
   `rviz2`
2. W oknie programu RViz, w lewym dolnym rogu kliknij przycisk **Add**.
3. Wybierz zakładkę **By topic** i znajdź temat `/sam3/smoothed_mask`.
4. Kliknij na niego dwukrotnie, aby dodać okno podglądu wideo.

**🎉 Gotowe!** Widzisz teraz obraz z kamery z nałożonymi kolorowymi maskami. Spróbuj na ułamek sekundy zasłonić narzędzie dłonią – dzięki zaimplementowanej heurystyce (Temporal Smoothing), maska nie zniknie natychmiast, zachowując stabilność śledzenia dla robota!