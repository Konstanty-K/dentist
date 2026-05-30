#!/usr/bin/env python3
"""
SAM3 Live Perception Node for ROS 2 Humble
Project: Tiago Pro Surgical Tool Grasping
Author: Konstanty Kaszubski
Team: Konstanty Kaszubski, Jakub Jagodziński, Adam Klimczak
Institution: Poznań University of Technology

This node subscribes to a raw camera feed, performs zero-shot instance segmentation 
using Meta's SAM3 model, and publishes the annotated masks back to ROS 2 (RViz2).
Optimized for NVIDIA Blackwell (RTX 50-series) architectures via torch.compile.
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

class Sam3LiveNode(Node):
    def __init__(self):
        super().__init__('sam3_live_node')
        
        # --- Parametry ROS2 ---
        self.declare_parameter('use_compile', False)
        use_compile = self.get_parameter('use_compile').get_parameter_value().bool_value
        
        self.frame_counter = 0
        self.bridge = CvBridge()
        
        self.camera_topic = '/image_raw'
        self.output_topic = '/sam3/annotated_image'
        query_path = '/workspace/queries.json'
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            self.get_logger().error("Brak HF_TOKEN w środowisku (flaga --env)!")
            raise RuntimeError("Missing HF_TOKEN")

        with open(query_path, 'r') as f:
            self.query_dict = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Inicjalizacja węzła na: {self.device}")

        # --- Ładowanie modelu ---
        self.get_logger().info("Trwa ładowanie wag SAM3 do VRAM...")
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=hf_token
        ).to(self.device)
        
        # Opcjonalna kompilacja sprzętowa (Uruchamiana flagą)
        if use_compile and self.device == "cuda":
            self.get_logger().info("Tryb JIT aktywny. Zapisywanie cache do /workspace/.torch_cache")
            self.get_logger().info("Ostrzeżenie: Pierwsza klatka zablokuje węzeł na kilka minut!")
            torch._dynamo.config.suppress_errors = True
            self.model = torch.compile(self.model, mode="max-autotune")
        
        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        # --- Sygnał dźwiękowy oznaczający gotowość AI ---
        sys.stdout.write("\a\a\a")
        sys.stdout.flush()
        self.get_logger().info("🔔 MODEL GOTOWY! Wagi załadowane do VRAM.")

        # --- Komunikacja ---
        self.subscription = self.create_subscription(Image, self.camera_topic, self.image_callback, 1)
        self.publisher = self.create_publisher(Image, self.output_topic, 1)

        self.get_logger().info(f"Węzeł gotowy! Subskrybuję temat: {self.camera_topic}")

    def image_callback(self, msg):
        # Pomijanie klatek w celu uniknięcia opóźnień (lagów)
        self.frame_counter += 1
        if self.frame_counter % 3 != 0:
            return 
    
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Skalowanie w dół dla poprawy FPS (OpenCV format: Szerokość, Wysokość)
        small_frame = cv2.resize(cv_image, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        overlay = small_frame.copy()

        # --- Inferencja AI ---
        for query_idx, (query, thresholds) in enumerate(self.query_dict.items()):
            color = self.colors[query_idx % len(self.colors)]
            inputs = self.processor(images=rgb_frame, text=query, return_tensors="pt").to(self.device)
            
            # Autocast do płynnego i bezpiecznego zarządzania typami (float32 -> float16)
            for key in inputs.keys():
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(self.model.dtype)

            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs, 
                threshold=thresholds.get("confidence", 0.5), 
                mask_threshold=thresholds.get("mask_confidence", 0.5),
                target_sizes=[(240, 320)] # HuggingFace format: Wysokość, Szerokość
            )[0]

            # --- Rysowanie masek ---
            masks = results["masks"]
            if len(masks) > 0:
                for idx, mask in enumerate(masks):
                    mask_np = mask.squeeze().cpu().numpy() > 0.5
                    if not np.any(mask_np): continue
                    
                    overlay[mask_np] = color
                    y_coords, x_coords = np.where(mask_np)
                    x_min, y_min = int(x_coords.min()), int(y_coords.min())
                    cv2.putText(small_frame, query, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, small_frame, 1 - alpha, 0, small_frame)

        # --- Publikacja ---
        try:
            out_msg = self.bridge.cv2_to_imgmsg(small_frame, "bgr8")
            out_msg.header = msg.header
            self.publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Sam3LiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
