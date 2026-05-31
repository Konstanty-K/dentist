#!/usr/bin/env python3
"""
SAM3 Robust Tracker Node for ROS 2
Project: Tiago Pro Surgical Tool Grasping
Author: Konstanty Kaszubski

This node combines zero-shot instance segmentation (Meta SAM3) with 
temporal smoothing (Grace Period Heuristic) to prevent mask flickering 
during momentary occlusions or loss of confidence.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import os
import sys
import json
import torch
import torch._dynamo
import numpy as np
import cv2
from transformers import Sam3Processor, Sam3Model

# Wymuszenie zapisu skompilowanych binariów CUDA do zmapowanego folderu
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/workspace/.torch_cache"

class SamTrackerNode(Node):
    def __init__(self):
        super().__init__('sam_tracker_node')
        
        # --- 1. Konfiguracja Przepływu i Heurystyki ---
        self.camera_fps = 8.5 
        self.target_fps = 3.0 
        self.frame_skip_ratio = max(1, int(self.camera_fps / self.target_fps))
        self.frame_counter = 0

        self.MAX_GRACE_FRAMES = 3
        # Słownik śledzenia dla każdego obiektu z queries.json
        # Struktura: {"nazwa_z_json": {"mask": np.array, "grace_left": int}}
        self.tracked_objects = {}

        # --- 2. Konfiguracja Środowiska SAM3 ---
        self.declare_parameter('use_compile', False)
        use_compile = self.get_parameter('use_compile').get_parameter_value().bool_value
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            self.get_logger().error("Brak HF_TOKEN w środowisku (flaga --env)!")
            raise RuntimeError("Missing HF_TOKEN")

        query_path = '/workspace/queries.json'
        with open(query_path, 'r') as f:
            self.query_dict = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Inicjalizacja węzła na: {self.device}")

        # --- 3. Ładowanie wag modelu ---
        self.get_logger().info("Trwa ładowanie wag SAM3 do VRAM...")
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=hf_token
        ).to(self.device)
        
        if use_compile and self.device == "cuda":
            self.get_logger().info("Tryb JIT aktywny. Zapisywanie cache do /workspace/.torch_cache")
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model, mode="max-autotune")
        
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

        sys.stdout.write("\a\a\a")
        sys.stdout.flush()
        self.get_logger().info(f"🔔 MODEL GOTOWY! Decimation: 1/{self.frame_skip_ratio}, Grace: {self.MAX_GRACE_FRAMES} frames")

        # --- 4. ROS 2 Komunikacja ---
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 1)
        self.publisher = self.create_publisher(Image, '/sam3/smoothed_mask', 1)


    def image_callback(self, msg):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip_ratio != 0:
            return 
    
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Przygotowanie obrazu (skalowanie przyspiesza inferencję)
        small_frame = cv2.resize(cv_image, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        overlay = small_frame.copy()

        # --- 5. Główna Pętla Inferencji i Śledzenia ---
        for query_idx, (query, thresholds) in enumerate(self.query_dict.items()):
            color = self.colors[query_idx % len(self.colors)]
            
            # Inicjalizacja stanu w słowniku dla nowych obiektów
            if query not in self.tracked_objects:
                self.tracked_objects[query] = {"mask": None, "grace_left": 0}

            inputs = self.processor(images=rgb_frame, text=query, return_tensors="pt").to(self.device)
            
            for key in inputs.keys():
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(self.model.dtype)

            # Wnioskowanie SAM3
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs, 
                threshold=thresholds.get("confidence", 0.5), 
                mask_threshold=thresholds.get("mask_confidence", 0.5),
                target_sizes=[(240, 320)]
            )[0]

            masks = results["masks"]
            object_found_in_current_frame = False
            combined_mask = np.zeros((240, 320), dtype=bool)

            # Jeśli model coś wykrył, łączymy wszystkie instancje w jedną maskę
            if len(masks) > 0:
                for mask in masks:
                    mask_np = mask.squeeze().cpu().numpy() > 0.5
                    if np.any(mask_np):
                        combined_mask = np.logical_or(combined_mask, mask_np)
                        object_found_in_current_frame = True

            # --- 6. Logika Heurystyki (Temporal Smoothing) ---
            if object_found_in_current_frame:
                # Obiekt widoczny - aktualizujemy wiedzę i odnawiamy życia
                self.tracked_objects[query]["mask"] = combined_mask
                self.tracked_objects[query]["grace_left"] = self.MAX_GRACE_FRAMES
            else:
                # Obiekt zgubiony - sprawdzamy, czy mamy życia
                if self.tracked_objects[query]["grace_left"] > 0 and self.tracked_objects[query]["mask"] is not None:
                    self.tracked_objects[query]["grace_left"] -= 1
                    # Podmieniamy pustą maskę na ostatnią znaną
                    combined_mask = self.tracked_objects[query]["mask"]
                    self.get_logger().info(f"Ostrzeżenie: {query} zgubiony. Podtrzymuję (życia: {self.tracked_objects[query]['grace_left']})")
                else:
                    # Brak żyć - obiekt ostatecznie wyczyszczony z pamięci
                    self.tracked_objects[query]["mask"] = None

            # --- 7. Rysowanie ustabilizowanych masek ---
            if np.any(combined_mask):
                overlay[combined_mask] = color
                y_coords, x_coords = np.where(combined_mask)
                x_min, y_min = int(x_coords.min()), int(y_coords.min())
                cv2.putText(small_frame, query, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Nałożenie przezroczystości i publikacja do RViz
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, small_frame, 1 - alpha, 0, small_frame)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(small_frame, "bgr8")
            out_msg.header = msg.header
            self.publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish Error: {e}")

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