import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class SamTrackerNode(Node):
    def __init__(self):
        super().__init__('sam_tracker_node')
        
        # Konfiguracja
        self.target_fps = 5.0  # Chcemy przetwarzać np. 5 klatek na sekundę
        self.camera_fps = 30.0 # Zakładana prędkość kamery (dostosuj do sprzętu)
        
        # Obliczamy co którą klatkę przepuszczać (Decimation factor)
        # Przy 30 FPS kamery i celu 5 FPS, przepuszczamy co 6 klatkę.
        self.frame_skip_ratio = int(self.camera_fps / self.target_fps)
        self.frame_counter = 0

        # Subskrypcja i publikacja
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',  # Zmień, jeśli topik kamery w Tiago nazywa się inaczej
            self.image_callback,
            10
        )
        
        # Topik, na którym docelowo pojawi się przetworzona maska (na razie wypuścimy czysty obraz dla testu)
        self.mask_pub = self.create_publisher(Image, '/sam3/smoothed_mask', 10)
        
        self.get_logger().info(f"Węzeł śledzenia SAM zainicjowany. Decimation: przetwarzam co {self.frame_skip_ratio} klatkę.")

    def image_callback(self, msg):
        self.frame_counter += 1
        
        # Logika pomijania klatek (Decimation)
        if self.frame_counter % self.frame_skip_ratio != 0:
            return  # Ignoruj klatkę, oszczędzaj zasoby

        # Odbiór klatki z ROS-a do OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji obrazu: {e}")
            return

        # ---------------------------------------------------------
        # TUTAJ TRAFI KROK 1.2 i 1.3:
        # - Wnioskowanie SAM3 na `cv_image`
        # - Logika uśredniania/podtrzymywania masek (Heurystyka)
        # ---------------------------------------------------------
        
        # Na potrzeby tego kroku (zanim wepniemy SAM3), symulujemy pracę wypisując log
        # i publikując z powrotem otrzymany obraz, by sprawdzić spadek FPS w RViz
        
        self.get_logger().info(f"Przetwarzam klatkę nr {self.frame_counter}...")
        
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.mask_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SamTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()