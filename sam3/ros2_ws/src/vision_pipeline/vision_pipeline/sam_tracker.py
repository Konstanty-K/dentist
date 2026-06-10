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
import numpy as np
import cv2
from transformers import Sam3Processor, Sam3Model

class SamTrackerNode(Node):
    def __init__(self):
        super().__init__('sam_tracker_node')
        
        # --- 1. Konfiguracja Przepływu i Heurystyki ---
        self.is_processing = False
        self.frame_counter = 0

        self.MAX_GRACE_FRAMES = 0
        self.tracked_objects = {}

        # --- 2. Konfiguracja Środowiska SAM3 ---
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

        self.processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

        # --- 4. OPTYMALIZACJA: Cache dla wektorów tekstu ---
        self.get_logger().info("Optymalizacja promptów tekstowych...")
        self.cached_queries = {}
        dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Przeliczamy słowa tylko raz i zapisujemy w pamięci podręcznej!
        for query in self.query_dict.keys():
            inputs = self.processor(images=dummy_image, text=query, return_tensors="pt").to(self.device)
            for key in inputs.keys():
                if torch.is_tensor(inputs[key]) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(self.model.dtype)
            self.cached_queries[query] = inputs

        sys.stdout.write("\a\a\a")
        sys.stdout.flush()
        self.get_logger().info(f"🔔 MODEL GOTOWY! Smart FPS (Async Drop), Grace: {self.MAX_GRACE_FRAMES} frames")

        # --- 5. ROS 2 Komunikacja ---
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 1)
        self.publisher = self.create_publisher(Image, '/sam3/smoothed_mask', 1)

    def image_callback(self, msg):
        if self.is_processing:
            return  # Pomijamy klatkę, jeśli RTX 5060 wciąż liczy poprzednią
        self.is_processing = True
    
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            self.is_processing = False  # Zwalniamy blokadę przy błędzie
            return

        # Skalowanie w dół dla poprawy FPS
        small_frame = cv2.resize(cv_image, (640, 480))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        overlay = small_frame.copy()

        # Przetwarzamy obraz wideo (baza do połączenia z tekstem)
        vision_inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)

        # --- 6. Główna Pętla Inferencji i Śledzenia (Z heurystyką) ---
        for query_idx, (query, thresholds) in enumerate(self.query_dict.items()):
            color = self.colors[query_idx % len(self.colors)]
            
            if query not in self.tracked_objects:
                self.tracked_objects[query] = {"mask": None, "grace_left": 0}

            # Składamy obraz z gotowym wektorem tekstu z pamięci (brak strat FPS na tekst!)
            combined_inputs = {**vision_inputs}
            cached_text = self.cached_queries[query]
            
            for key in cached_text.keys():
                if key not in combined_inputs:
                    combined_inputs[key] = cached_text[key]

            for key in combined_inputs.keys():
                if torch.is_tensor(combined_inputs[key]) and combined_inputs[key].dtype == torch.float32:
                    combined_inputs[key] = combined_inputs[key].to(self.model.dtype)

            # Szybka inferencja
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    outputs = self.model(**combined_inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs, 
                threshold=thresholds.get("confidence", 0.5), 
                mask_threshold=thresholds.get("mask_confidence", 0.5),
                target_sizes=[(480, 640)]
            )[0]

            masks = results["masks"]
            object_found_in_current_frame = False
            combined_mask = np.zeros((480, 640), dtype=bool)

            if len(masks) > 0:
                for mask in masks:
                    mask_np = mask.squeeze().cpu().numpy() > 0.5
                    if np.any(mask_np):
                        combined_mask = np.logical_or(combined_mask, mask_np)
                        object_found_in_current_frame = True

            # --- Logika Heurystyki (Temporal Smoothing) ---
            if object_found_in_current_frame:
                self.tracked_objects[query]["mask"] = combined_mask
                self.tracked_objects[query]["grace_left"] = self.MAX_GRACE_FRAMES
            else:
                if self.tracked_objects[query]["grace_left"] > 0 and self.tracked_objects[query]["mask"] is not None:
                    self.tracked_objects[query]["grace_left"] -= 1
                    combined_mask = self.tracked_objects[query]["mask"]
                else:
                    self.tracked_objects[query]["mask"] = None

            # --- Rysowanie masek ---
            if np.any(combined_mask):
                overlay[combined_mask] = color
                y_coords, x_coords = np.where(combined_mask)
                x_min, y_min = int(x_coords.min()), int(y_coords.min())
                cv2.putText(small_frame, query, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, small_frame, 1 - alpha, 0, small_frame)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(small_frame, "bgr8")
            out_msg.header = msg.header
            self.publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish Error: {e}")
        
        self.is_processing = False  # <--- Krytyczne otwarcie drzwi na kolejną klatkę!

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