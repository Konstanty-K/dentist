Symulator Kamery Orbbec (Mock Camera)
Ten moduł to wyizolowane środowisko testowe. Służy do odtwarzania historycznych nagrań z kamery (rosbag) i symulowania ich jako fizycznego strumienia wideo z urządzenia Orbbec Femto Bolt w czasie rzeczywistym.

Dzięki wykorzystaniu natywnego obrazu ROS 2 Jazzy, moduł nie wymaga żadnych instalacji na komputerze hosta ani ingerencji w główne kontenery projektu. Po zakończeniu pracy kontener ulega całkowitemu zniszczeniu, pozostawiając system czystym.

Wymagania
Zainstalowany Docker

Plik konfiguracyjny omijający Shared Memory (SHM) w głównym folderze: ../fastdds_no_shm.xml

Nagranie kamery w formacie MCAP (np. bag_001_0.mcap) z zapisanym plikiem metadata.yaml

Jak uruchomić symulację?
Przejdź do folderu modułu:
cd ~/robotics/dentist/mock_camera

Uruchom skrypt startowy:
./sim_orbbec_camera.sh

Skrypt automatycznie:

Pobierze czysty obraz ros:jazzy-ros-base (tylko przy pierwszym uruchomieniu).

Podmontuje nagranie w trybie tylko do odczytu (Read-Only), chroniąc oryginalne pliki.

Zaaplikuje konfigurację FastDDS (wyłączenie SHM).

Zacznie odtwarzać nagranie w nieskończonej pętli, przemapowując oryginalny temat z /camera/color/image_raw na uniwersalny /image_raw.

Zakończenie pracy następuje poprzez wciśnięcie Ctrl+C w terminalu. Kontener zostanie usunięty automatycznie.
