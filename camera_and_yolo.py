import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import struct

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
import math
from collections import deque

os.environ['YOLO_CONFIG_DIR'] = '/tmp'

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH       = '/Shared/weights/best.pt'
CONF_THRESHOLD   = 0.60
DEPTH_MIN_M      = 0.10
DEPTH_MAX_M      = 3.00
TOOL_HEIGHT_M    = 0.015   # zakładana wysokość chwytu narzędzia nad stołem [m]
DEPTH_PERCENTILE = 80      # percentyl głębokości → estymacja powierzchni stołu
MARKER_SCALE     = 0.03    # rozmiar markera w RViz [m]
SYNC_SLOP        = 0.15    # tolerancja synchronizacji klatek [s]

# Filtr temporalny orientacji
ANGLE_BUFFER_SIZE = 10     # liczba ostatnich klatek do uśrednienia kąta
ANGLE_JUMP_DEG    = 30.0   # próg skoku kąta – odrzucamy pomiary dalsze niż X stopni od mediany

# Tematy ROS 2
RGB_TOPIC         = '/camera/color/image_raw'
DEPTH_TOPIC       = '/camera/depth/image_raw'
RGB_INFO_TOPIC    = '/camera/color/camera_info'
DEPTH_INFO_TOPIC  = '/camera/depth/camera_info'
OUT_IMAGE_TOPIC   = '/detected_tool_image'
OUT_MARKERS_TOPIC = '/tool_markers'
OUT_PC_TOPIC      = '/detected_tool_pc'
OUT_POSE_TOPIC    = '/detected_tool_pose'

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACJA depth_optical_frame → color_optical_frame
#
# UWAGA: tf2_echo zwraca transformację A→B jako "jak wyrazić układ B w układzie A".
# Nam potrzebna jest transformacja punktów Z UKŁADU DEPTH DO UKŁADU COLOR,
# czyli: P_color = R * P_depth + T
#
# Poprawne źródło: tf2_echo camera_color_optical_frame camera_depth_optical_frame
# (odwrotny kierunek daje właściwą macierz dla transformacji punktów)
#
# Translation: [-0.032, -0.001,  0.003] m
# Rotation matrix (row-major):
#   [ 1.000  0.003  0.006 ]
#   [-0.003  0.994  0.109 ]
#   [-0.005 -0.109  0.994 ]
# ─────────────────────────────────────────────────────────────────────────────
R_DEPTH_TO_COLOR = np.array([
    [ 1.000,  0.003,  0.006],
    [-0.003,  0.994,  0.109],
    [-0.005, -0.109,  0.994]
], dtype=np.float64)

T_DEPTH_TO_COLOR = np.array([-0.032, -0.001, 0.003], dtype=np.float64)


class ToolPerceptionLiveNode(Node):
    """
    Węzeł ROS 2 do detekcji narzędzi w czasie rzeczywistym z kamerą Orbbec.

    Rozwiązuje problem niezgodności rozdzielczości:
      - Depth:  640x576  (parametry kamery depth)
      - RGB:   1280x720  (parametry kamery kolorowej)

    Algorytm dla każdego piksela depth:
      1. Piksel depth (u_d, v_d) + wartość Z → punkt 3D w układzie depth
      2. Obrót + translacja → punkt 3D w układzie color
      3. Rzut na płaszczyznę RGB → piksel (u_c, v_c)
      4. Sprawdzenie czy (u_c, v_c) leży w bounding boxie YOLO
    """

    def __init__(self):
        super().__init__('tool_perception_live_node')
        self.get_logger().info("=== INICJALIZACJA WĘZŁA TOOL PERCEPTION (ORBBEC) ===")

        # ── Model YOLO ────────────────────────────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            self.get_logger().fatal(f"Brak pliku modelu: {MODEL_PATH}")
            raise FileNotFoundError(MODEL_PATH)

        self.model  = YOLO(MODEL_PATH)
        self.bridge = CvBridge()

        # ── Parametry kamer (wypełniane z CameraInfo) ─────────────────────────
        self.rgb_fx = self.rgb_fy = self.rgb_cx = self.rgb_cy = None
        self.rgb_w  = self.rgb_h  = None
        self.d_fx   = self.d_fy   = self.d_cx   = self.d_cy   = None

        self.rgb_info_ready   = False
        self.depth_info_ready = False

        # ── Cache reprojekcji (przeliczany gdy rozmiar depth się zmieni) ──────
        self._ray_dirs        = None   # (H_d, W_d, 3) – kierunki promieni depth
        self._last_depth_shape = None

        # ── Filtry temporalne kątów (osobny bufor na każdy det_id/klasę) ─────
        # Klucz: cls_name (str) → deque ostatnich kątów [stopnie]
        self._angle_buffers: dict[str, deque] = {}

        # ── QoS ───────────────────────────────────────────────────────────────
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ── Publishery ────────────────────────────────────────────────────────
        self.image_pub  = self.create_publisher(Image,       OUT_IMAGE_TOPIC,   reliable_qos)
        self.marker_pub = self.create_publisher(MarkerArray, OUT_MARKERS_TOPIC, reliable_qos)
        self.pc_pub     = self.create_publisher(PointCloud2, OUT_PC_TOPIC,      reliable_qos)
        self.pose_pub   = self.create_publisher(PoseStamped,  OUT_POSE_TOPIC,    reliable_qos)

        # ── Subskrypcje CameraInfo (jednorazowe) ──────────────────────────────
        self._rgb_info_sub = self.create_subscription(
            CameraInfo, RGB_INFO_TOPIC, self._rgb_info_cb, reliable_qos)
        self._depth_info_sub = self.create_subscription(
            CameraInfo, DEPTH_INFO_TOPIC, self._depth_info_cb, reliable_qos)

        # ── Synchronizowane subskrypcje RGB + Depth ───────────────────────────
        self.rgb_sub   = message_filters.Subscriber(
            self, Image, RGB_TOPIC,   qos_profile=best_effort_qos)
        self.depth_sub = message_filters.Subscriber(
            self, Image, DEPTH_TOPIC, qos_profile=best_effort_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=SYNC_SLOP
        )
        self.ts.registerCallback(self._image_callback)

        self.get_logger().info(
            f"Nasłuchuję: RGB({RGB_TOPIC}), Depth({DEPTH_TOPIC})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CALLBACKI CameraInfo
    # ─────────────────────────────────────────────────────────────────────────

    def _rgb_info_cb(self, msg: CameraInfo):
        if self.rgb_info_ready:
            return
        K = msg.k
        self.rgb_fx, self.rgb_fy = K[0], K[4]
        self.rgb_cx, self.rgb_cy = K[2], K[5]
        self.rgb_w, self.rgb_h   = msg.width, msg.height
        self.rgb_info_ready = True
        self.get_logger().info(
            f"RGB CameraInfo: {msg.width}x{msg.height} "
            f"fx={self.rgb_fx:.2f} cx={self.rgb_cx:.2f} cy={self.rgb_cy:.2f}"
        )
        self.destroy_subscription(self._rgb_info_sub)

    def _depth_info_cb(self, msg: CameraInfo):
        if self.depth_info_ready:
            return
        K = msg.k
        self.d_fx, self.d_fy = K[0], K[4]
        self.d_cx, self.d_cy = K[2], K[5]
        self.depth_info_ready = True
        self.get_logger().info(
            f"Depth CameraInfo: {msg.width}x{msg.height} "
            f"fx={self.d_fx:.2f} cx={self.d_cx:.2f} cy={self.d_cy:.2f}"
        )
        self.destroy_subscription(self._depth_info_sub)

    # ─────────────────────────────────────────────────────────────────────────
    # GŁÓWNY CALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _image_callback(self, rgb_msg: Image, depth_msg: Image):

        if not (self.rgb_info_ready and self.depth_info_ready):
            self.get_logger().warn(
                "Brak CameraInfo – pomijam klatkę.",
                throttle_duration_sec=2.0
            )
            return

        # ── Konwersja ROS → OpenCV ────────────────────────────────────────────
        try:
            rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Błąd konwersji obrazu: {e}")
            return

        # ── Buduj cache kierunków promieni (raz na zmianę rozmiaru) ──────────
        if depth.shape != self._last_depth_shape:
            self._build_ray_dirs(depth.shape)
            self._last_depth_shape = depth.shape

        # Głębokość [m]
        depth_m = depth.astype(np.float32) / 1000.0

        # ── Detekcja YOLO ─────────────────────────────────────────────────────
        results    = self.model(rgb, verbose=False)[0]
        annotated  = rgb.copy()
        marker_arr = MarkerArray()
        all_points = []

        for det_id, box in enumerate(results.boxes):
            conf = float(box.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue

            cls_id   = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # ── Reprojekcja depth → bbox RGB ─────────────────────────────────
            z, x_3d, y_3d, roi_pts = self._depth_in_bbox(
                depth_m, rgb, x1, y1, x2, y2
            )

            if z is None:
                cv2.putText(annotated, "BRAK DEPTH", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                self.get_logger().warn(
                    f"Brak głębokości dla '{cls_name}' [{det_id}]",
                    throttle_duration_sec=1.0
                )
                continue

            # ── Orientacja narzędzia ──────────────────────────────────────
            angle_raw, _, annotated = self._estimate_orientation(
                rgb, annotated, x1, y1, x2, y2
            )
            angle_deg = self._filter_angle(cls_name, angle_raw)
            # Przelicz kwaternion z przefiltrowanego kąta
            half_rad = math.radians(angle_deg) / 2.0
            quat = (0.0, 0.0, math.sin(half_rad), math.cos(half_rad))

            # ── Adnotacje 2D ─────────────────────────────────────────────────
            cx_2d = (x1 + x2) // 2
            cy_2d = (y1 + y2) // 2
            label = f"{cls_name} {conf:.2f} | Z={z:.3f}m | a={angle_deg:.1f}deg"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.circle(annotated, (cx_2d, cy_2d), 6, (0, 0, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)

            marker_arr.markers.append(
                self._build_marker(rgb_msg, det_id, cls_name, x_3d, y_3d, z, quat)
            )
            all_points.extend(roi_pts)

            # ── Publikacja PoseStamped ────────────────────────────────────────
            pose_msg = PoseStamped()
            pose_msg.header          = rgb_msg.header
            pose_msg.header.frame_id = "camera_color_optical_frame"
            pose_msg.pose.position.x = float(x_3d)
            pose_msg.pose.position.y = float(y_3d)
            pose_msg.pose.position.z = float(z)
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            self.pose_pub.publish(pose_msg)

            self.get_logger().info(
                f"[{cls_name}] conf={conf:.2f} | "
                f"3D x={x_3d:.3f} y={y_3d:.3f} z={z:.3f} | kat={angle_deg:.1f}deg",
                throttle_duration_sec=0.5
            )

        # ── Publikacja ────────────────────────────────────────────────────────
        self._publish_image(annotated, rgb_msg)

        if marker_arr.markers:
            self.marker_pub.publish(marker_arr)

        if all_points:
            self.pc_pub.publish(self._build_pointcloud(all_points, rgb_msg))

    # ─────────────────────────────────────────────────────────────────────────
    # REPROJEKCJA
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ray_dirs(self, depth_shape):
        """
        Prekalkuluje kierunki promieni dla każdego piksela obrazu depth.
        Wynik: self._ray_dirs shape (H_d, W_d, 3), float32.

        Promień piksela (u_d, v_d) w układzie depth:
            dir = [ (u_d - cx_d)/fx_d,  (v_d - cy_d)/fy_d,  1.0 ]
        Punkt 3D = dir * Z_m
        """
        H_d, W_d = depth_shape
        u_d, v_d = np.meshgrid(np.arange(W_d, dtype=np.float32),
                                np.arange(H_d, dtype=np.float32))
        x_n = (u_d - self.d_cx) / self.d_fx
        y_n = (v_d - self.d_cy) / self.d_fy
        self._ray_dirs = np.stack([x_n, y_n, np.ones((H_d, W_d), dtype=np.float32)], axis=2)
        self.get_logger().info(f"Zbudowano cache promieni dla depth {W_d}x{H_d}")

    def _depth_in_bbox(self, depth_m: np.ndarray, rgb: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int):
        """
        Dla danego bounding boxa (w układzie RGB) wyszukuje odpowiadające
        piksele depth i zwraca estymowaną pozycję 3D narzędzia.

        Kroki:
          1. P_depth = ray_dirs * Z_m           (3D w układzie depth)
          2. P_color = R @ P_depth + T          (3D w układzie color)
          3. u_c = fx_rgb * Xc/Zc + cx_rgb      (rzut na RGB)
          4. Maska: u_c w [x1,x2], v_c w [y1,y2]
          5. Percentyl głębokości → estymacja stołu → pozycja narzędzia

        Zwraca: (z_tool, x_3d, y_3d, roi_points) lub (None, None, None, [])
        """
        H_d, W_d = depth_m.shape

        # Punkt 3D w układzie depth: shape (H_d, W_d, 3)
        P_depth = self._ray_dirs * depth_m[:, :, np.newaxis]

        # Transformacja do układu color – wektoryzowana
        # P_flat: (N,3), wynik: (N,3)
        P_flat  = P_depth.reshape(-1, 3).astype(np.float64)
        P_color = (R_DEPTH_TO_COLOR @ P_flat.T).T + T_DEPTH_TO_COLOR
        P_color = P_color.reshape(H_d, W_d, 3).astype(np.float32)

        # Rzut na płaszczyznę RGB
        Zc = P_color[:, :, 2]
        valid = Zc > 0.001

        u_c = np.zeros((H_d, W_d), dtype=np.float32)
        v_c = np.zeros((H_d, W_d), dtype=np.float32)
        u_c[valid] = P_color[:, :, 0][valid] / Zc[valid] * self.rgb_fx + self.rgb_cx
        v_c[valid] = P_color[:, :, 1][valid] / Zc[valid] * self.rgb_fy + self.rgb_cy

        # Maska pikseli depth trafiających w bounding box RGB
        in_box = (
            valid &
            (u_c >= x1) & (u_c < x2) &
            (v_c >= y1) & (v_c < y2) &
            (depth_m > 0)
        )

        valid_depths = depth_m[in_box]
        if valid_depths.size == 0:
            return None, None, None, []

        # Estymacja głębokości stołu i pozycji narzędzia
        table_m = float(np.percentile(valid_depths, DEPTH_PERCENTILE))
        z_tool  = table_m - TOOL_HEIGHT_M

        if not (DEPTH_MIN_M < z_tool < DEPTH_MAX_M):
            self.get_logger().warn(
                f"Głębokość poza zakresem: z={z_tool:.3f} m (stół={table_m:.3f} m)",
                throttle_duration_sec=1.0
            )
            return None, None, None, []

        # Środek 3D: mediana rzutowanych współrzędnych → deprojekcja z rgb_fx
        u_med = float(np.median(u_c[in_box]))
        v_med = float(np.median(v_c[in_box]))
        x_3d  = (u_med - self.rgb_cx) * z_tool / self.rgb_fx
        y_3d  = (v_med - self.rgb_cy) * z_tool / self.rgb_fy

        # Chmura punktów ROI (co 3. piksel)
        ys, xs = np.where(in_box)
        roi_pts = []
        for i in range(0, len(ys), 3):
            yy, xx = ys[i], xs[i]
            p = P_color[yy, xx]
            if not (DEPTH_MIN_M < float(p[2]) < DEPTH_MAX_M):
                continue
            # Kolor z obrazu RGB (po rzucie)
            u_rgb = int(np.clip(u_c[yy, xx], 0, rgb.shape[1] - 1))
            v_rgb = int(np.clip(v_c[yy, xx], 0, rgb.shape[0] - 1))
            b, g, r = rgb[v_rgb, u_rgb]
            packed = (int(r) << 16) | (int(g) << 8) | int(b)
            roi_pts.append((float(p[0]), float(p[1]), float(p[2]), packed))

        return z_tool, x_3d, y_3d, roi_pts

    # ─────────────────────────────────────────────────────────────────────────
    # ESTYMACJA ORIENTACJI NARZĘDZIA
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_angle(self, cls_name: str, angle_raw: float) -> float:
        """
        Filtr temporalny kąta orientacji narzędzia.

        Algorytm:
          1. Dodaj nowy pomiar do bufora (deque o rozmiarze ANGLE_BUFFER_SIZE)
          2. Odrzuć wartości odstające (dalej niż ANGLE_JUMP_DEG od mediany)
          3. Zwróć średnią z pozostałych próbek

        Uwaga: kąty są cykliczne (0° == 180°), więc normalizujemy różnice
        względem mediany tak aby mieściły się w przedziale (-90°, 90°).
        """
        if cls_name not in self._angle_buffers:
            self._angle_buffers[cls_name] = deque(maxlen=ANGLE_BUFFER_SIZE)

        buf = self._angle_buffers[cls_name]
        buf.append(angle_raw)

        if len(buf) < 2:
            return angle_raw

        angles = np.array(buf, dtype=np.float32)
        median  = float(np.median(angles))

        # Normalizacja różnic z uwzględnieniem cykliczności [0, 180)
        diffs = angles - median
        diffs = (diffs + 90.0) % 180.0 - 90.0   # zakres (-90, 90)

        # Odrzuć wartości odstające
        inliers = angles[np.abs(diffs) < ANGLE_JUMP_DEG]

        if inliers.size == 0:
            return median   # wszystko odrzucone → wróć do mediany

        return float(np.mean(inliers))

    def _estimate_orientation(self, rgb: np.ndarray, annotated: np.ndarray,
                               x1: int, y1: int, x2: int, y2: int):
        """
        Estymuje orientację narzędzia metodą PCA na pikselach obiektu.

        Algorytm:
          1. Wytnij ROI z obrazu RGB
          2. Wielopoziomowa segmentacja:
             a) Progowanie Otsu na kanale V (jasność w HSV) – łapie jasne narzędzia
             b) Progowanie adaptacyjne jako fallback gdy Otsu zawodzi
             c) Morfologia dla wyczyszczenia maski
          3. PCA na współrzędnych pikseli należących do obiektu
             → pierwsza składowa = oś główna narzędzia
             → stabilniejsze niż minAreaRect bo używa WSZYSTKICH pikseli, nie tylko konturu
          4. Kąt osi głównej PCA → kwaternion

        PCA vs minAreaRect:
          - minAreaRect operuje na konturze (brzeg obiektu) – wrażliwy na szum segmentacji
          - PCA operuje na wszystkich pikselach wewnątrz maski – uśrednia szum naturalnie
          - Wynik PCA jest znacznie stabilniejszy przy małych zmianach maski
        """
        roi = rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, (0.0, 0.0, 0.0, 1.0), annotated

        h_roi, w_roi = roi.shape[:2]

        # ── Segmentacja wielopoziomowa ────────────────────────────────────────
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)   # kanał V (jasność)

        # Poziom 1: Otsu na jasności
        otsu_thresh, mask_otsu = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Poziom 2: progowanie adaptacyjne (odporne na zmiany oświetlenia)
        mask_adapt = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=-2
        )

        # Połącz obie maski (AND) – trzymamy tylko piksele jasne globalnie I lokalnie
        mask = cv2.bitwise_and(mask_otsu, mask_adapt)

        # Morfologia: usuń szum i połącz pobliskie fragmenty
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_small)  # usuń szum
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)  # połącz fragmenty

        # Jeśli maska jest pusta lub prawie pusta – fallback do Otsu
        fill_ratio = np.sum(mask > 0) / (h_roi * w_roi)
        if fill_ratio < 0.05:
            mask = mask_otsu
            self.get_logger().warn(
                "Maska AND pusta – używam samego Otsu.",
                throttle_duration_sec=2.0
            )

        # ── PCA na pikselach maski ────────────────────────────────────────────
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            self.get_logger().warn(
                "Za mało pikseli do PCA – zwracam kąt 0.",
                throttle_duration_sec=1.0
            )
            return 0.0, (0.0, 0.0, 0.0, 1.0), annotated

        # Macierz danych: każdy wiersz to [x, y] jednego piksela
        pts = np.column_stack([xs, ys]).astype(np.float32)

        # Wyśrodkowanie
        mean_pt = pts.mean(axis=0)
        pts_centered = pts - mean_pt

        # Kowariancja i wektory własne
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Pierwsza składowa PCA = kierunek największej wariancji = oś główna
        primary_axis = eigvecs[:, np.argmax(eigvals)]   # wektor [dx, dy]

        # Kąt osi głównej względem osi X (w stopniach)
        angle_rad = math.atan2(float(primary_axis[1]), float(primary_axis[0]))
        angle_deg = math.degrees(angle_rad) % 180.0     # normalizacja do [0, 180)

        # ── Rysowanie osi na obrazie ──────────────────────────────────────────
        cx_roi = int(mean_pt[0]) + x1
        cy_roi = int(mean_pt[1]) + y1
        half_len = int(max(w_roi, h_roi) * 0.5)

        dx = int(half_len * math.cos(angle_rad))
        dy = int(half_len * math.sin(angle_rad))

        # Oś główna – żółta linia
        cv2.line(annotated,
                 (cx_roi - dx, cy_roi - dy),
                 (cx_roi + dx, cy_roi + dy),
                 (0, 255, 255), 2)
        # Grot kierunku – zielony
        cv2.arrowedLine(annotated,
                        (cx_roi, cy_roi),
                        (cx_roi + dx, cy_roi + dy),
                        (0, 200, 0), 2, tipLength=0.25)

        # ── Kąt → kwaternion ─────────────────────────────────────────────────
        half_q = angle_rad / 2.0
        quat = (0.0, 0.0, math.sin(half_q), math.cos(half_q))

        return angle_deg, quat, annotated

    # ─────────────────────────────────────────────────────────────────────────
    # BUDOWANIE WIADOMOŚCI ROS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_marker(self, rgb_msg, det_id, cls_name, x, y, z, quat=(0,0,0,1)) -> Marker:
        m = Marker()
        m.header          = rgb_msg.header
        m.header.frame_id = "camera_color_optical_frame"
        m.ns              = "tools"
        m.id              = det_id
        
        # ZMIANA: Zmieniamy kulę na strzałkę, żeby widzieć kąt obrotu
        m.type            = Marker.ARROW
        m.action          = Marker.ADD
        
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.x = float(quat[0])
        m.pose.orientation.y = float(quat[1])
        m.pose.orientation.z = float(quat[2])
        m.pose.orientation.w = float(quat[3])
        
        # Dla ARROW skala oznacza co innego:
        # x = długość strzałki, y = grubość trzonu, z = szerokość grotu
        m.scale.x = 0.08  # 8 cm długości
        m.scale.y = 0.01  # 1 cm grubości trzonu
        m.scale.z = 0.015 # 1.5 cm szerokości grotu
        
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.4
        m.color.b = 0.0
        m.text = cls_name
        return m

    def _build_pointcloud(self, points: list, rgb_msg) -> PointCloud2:
        msg              = PointCloud2()
        msg.header       = rgb_msg.header
        msg.header.frame_id = "camera_color_optical_frame"
        msg.height       = 1
        msg.width        = len(points)
        msg.is_bigendian = False
        msg.is_dense     = False
        msg.point_step   = 16
        msg.row_step     = 16 * len(points)
        msg.fields       = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
        ]
        msg.data = b''.join(
            struct.pack('fffI', p[0], p[1], p[2], p[3]) for p in points
        )
        return msg

    def _publish_image(self, frame, rgb_msg):
        try:
            out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            out.header          = rgb_msg.header
            out.header.frame_id = "camera_color_optical_frame"
            self.image_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f"Błąd publikacji obrazu: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    try:
        node = ToolPerceptionLiveNode()
        rclpy.spin(node)
    except FileNotFoundError as e:
        print(f"[FATAL] Brak pliku: {e}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[FATAL] Nieoczekiwany błąd: {e}")
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()