# TiagoProDentist - Tool Perception System

Projekt wykrywania i pozycjonowania narzędzi stomatologicznych w przestrzeni 3D przy użyciu zaawansowanych systemów wizyjnych oraz systemu **ROS 2**. System posiada dwa niezależne potoki (pipelines) detekcji dla robota TIAGo Pro:
1. **Szybka detekcja (YOLOv8)** - przetwarzająca dane z kamery RGB-D, publikująca obraz z detekcjami oraz chmurę punktów.
2. **Precyzyjna segmentacja Zero-Shot (SAM3)** - działająca na dedykowanym węźle z własną heurystyką śledzenia (Temporal Smoothing).

### Architektura Systemu Wieloagentowego
```mermaid
graph TD
    Cam[Kamera RGB laptopa] --> N_Cam[Węzeł: usb_cam_node]
    DepthCam[Kamera 3D np. RealSense] --> N_Grasp[Węzeł: graspnet_node]

    subgraph Kontener AI SAM3 GPU
        N_Sam[Węzeł: sam_tracker_node]
        P_SAM3{{Model SAM3 PyTorch}}
        N_Sam -->|Zapytanie wizyjne| P_SAM3
        P_SAM3 -->|Maska z inferencji| N_Sam
    end

    subgraph Kontener Contact-GraspNet
        N_Grasp
        P_Grasp{{Model Contact-GraspNet}}
        N_Grasp -->|Ocena punktów 3D| P_Grasp
        P_Grasp -->|Współrzędne docelowe| N_Grasp
    end

    N_Cam -->|Topik: /image_raw ~30 FPS| N_Sam
    N_Sam -->|Topik: /sam3/smoothed_mask ~3 FPS| N_Grasp
    N_Grasp -->|Topik: /tiago/grasp_pose| Robot[Robot TIAGo Pro]

    style Kontener AI SAM3 GPU fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style Kontener Contact-GraspNet fill:#e8f5e9,stroke:#388e3c,stroke-width:2px