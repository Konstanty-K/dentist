import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import glob
import struct
from sensor_msgs.msg import PointCloud2, PointField, Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Zapobiega błędom uprawnień YOLO wewnątrz Dockera
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

class ToolPerceptionNode(Node):
    def __init__(self):
        super().__init__('tool_perception_node')
        
        self.get_logger().info("--- INICJALIZACJA WĘZŁA ---")
        
        # 1. Ścieżka do modelu (zgodna z nową strukturą weights/)
        # /Shared to główny folder projektu wewnątrz kontenera
        model_path = '/Shared/weights/best.pt'
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"BRAK PLIKU MODELU: {model_path}")
            # Próba znalezienia modelu, jeśli ścieżka by się różniła
            self.get_logger().info(f"Zawartość /Shared/weights: {os.listdir('/Shared/weights') if os.path.exists('/Shared/weights') else 'folder nie istnieje'}")
        
        self.model = YOLO(model_path) 
        self.bridge = CvBridge()
        
        self.current_frame_idx = 0
        
        # 2. QoS na RELIABLE - sprawdzone, że działa z Twoim RViz
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pc_pub = self.create_publisher(PointCloud2, '/detected_tool_pc', qos_profile)
        self.marker_pub = self.create_publisher(Marker, '/tool_centroid', qos_profile)
        self.image_pub = self.create_publisher(Image, '/detected_tool_image', qos_profile)

        # Parametry kamery (Intrinsics)
        self.fx, self.fy = 750.65, 750.58
        self.cx, self.cy = 643.81, 363.89
        
        # 3. Ścieżka do danych 
        # Upewnij się, że folder 'data' znajduje się w głównym katalogu projektu na Hoście
        self.base_path = "/Shared/bag"
        
        # Timer: 0.2s = 5 FPS (bezpieczna wartość dla stabilności obrazu)
        self.create_timer(0.2, self.run_inference)
        self.get_logger().info("DIAGNOSTYKA: Węzeł gotowy. Czekam na klatki...")

    def create_pc2(self, points):
        """Tworzenie chmury punktów ROS 2"""
        msg = PointCloud2()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_data = [struct.pack('fff', float(p[0]), float(p[1]), float(p[2])) for p in points]
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.point_step = 12
        msg.row_step = 12 * len(points)
        msg.data = b''.join(cloud_data)
        return msg

    def run_inference(self):
        rgb_dir = os.path.join(self.base_path, "camera-color-image_raw")
        depth_dir = os.path.join(self.base_path, "camera-depth-image_raw")
        
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

        if not rgb_files or not depth_files:
            return

        self.current_frame_idx = (self.current_frame_idx + 1) % len(rgb_files)
        
        # Wczytywanie RGB i Depth (ważne: cv2.IMREAD_UNCHANGED dla głębi!)
        rgb = cv2.imread(rgb_files[self.current_frame_idx])
        depth = cv2.imread(depth_files[self.current_frame_idx], cv2.IMREAD_UNCHANGED)
        
        if rgb is None or depth is None:
            return

        results = self.model(rgb, verbose=False)[0]
        annotated_frame = results.plot() 
        
        all_points = []

        # Przetwarzanie każdej wykrytej ramki (Box)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Wycięcie fragmentu głębi dla narzędzia
            roi_depth = depth[y1:y2, x1:x2]
            
            # Szybka konwersja na punkty 3D (Wersja zoptymalizowana)
            v_indices, u_indices = np.indices(roi_depth.shape)
            u_global = u_indices + x1
            v_global = v_indices + y1
            
            z = roi_depth.astype(float) / 1000.0  # Konwersja na metry
            mask = (z > 0.1) & (z < 2.0)  # Filtrowanie błędów (szumów)
            
            if np.any(mask):
                x = (u_global[mask] - self.cx) * z[mask] / self.fx
                y = (v_global[mask] - self.cy) * z[mask] / self.fy
                z_vals = z[mask]
                
                # Łączenie w listę punktów [x, y, z]
                points = np.stack((x, y, z_vals), axis=-1).tolist()
                all_points.extend(points)

        # 1. Publikacja obrazu
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            img_msg.header.frame_id = "camera_color_optical_frame"
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Błąd obrazu: {e}")

        # 2. Publikacja chmury punktów
        if all_points:
            pc_msg = self.create_pc2(all_points)
            self.pc_pub.publish(pc_msg)
            if self.current_frame_idx % 20 == 0:
                self.get_logger().info(f"Wysłano chmurę: {len(all_points)} punktów")
        rgb_dir = os.path.join(self.base_path, "camera-color-image_raw")
        depth_dir = os.path.join(self.base_path, "camera-depth-image_raw")
        
        # Pobieranie list plików
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

        if self.current_frame_idx == 0:
            self.get_logger().info(f"Szukam obrazów w: {rgb_dir}")
            self.get_logger().info(f"Znaleziono zdjęć RGB: {len(rgb_files)}")

        if not rgb_files:
            return

        # Iteracja po klatkach
        self.current_frame_idx = (self.current_frame_idx + 1) % len(rgb_files)
        rgb = cv2.imread(rgb_files[self.current_frame_idx])
        
        if rgb is None:
            return

        # Detekcja YOLO
        results = self.model(rgb, verbose=False)[0]
        annotated_frame = results.plot() 
        
        # Publikacja obrazu z naniesionymi detekcjami
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            img_msg.header.frame_id = "camera_color_optical_frame"
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)
            
            if self.current_frame_idx % 20 == 0:
                self.get_logger().info("Obraz wysyłany do RViz...")
        except Exception as e:
            self.get_logger().error(f"Błąd cv_bridge: {e}")

def main():
    rclpy.init()
    node = ToolPerceptionNode()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"KRACH WĘZŁA: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()