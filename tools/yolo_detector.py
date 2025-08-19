"""
yolo_detector.py — Wildlife Object Detection with YOLOv8
--------------------------------------------------------

Provides animal detection and smart cropping using a YOLOv8 model.

Features:
* Loads YOLOv8 model (default: `yolov8n.pt`)
* Runs detection on PIL images
* Filters results by confidence, area, and aspect ratio
* Returns high-quality crops or falls back to full image

Used during RAW image ingestion to improve species identification
by focusing on detected animals before CLIP embedding.

Dependencies:
- ultralytics YOLO (v8)
- PIL (Pillow)
- torch
"""

from ultralytics import YOLO
from PIL import Image
import torch


class YOLODetector:
    """
    Wrapper for YOLOv8 animal detection and image cropping.

    Args:
        model_path (str): Path to YOLOv8 model weights (.pt file)
        device (str): 'cuda', 'cpu', or None (auto-detects available device)
    """
    def __init__(self, model_path="yolov8n.pt", device=None):
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def detect_and_crop(self, pil_image: Image.Image, conf_threshold=0.3):
        """
        Detects animals in an image and returns valid crops along with their labels.

        Applies multiple filtering criteria:
        * Minimum crop area (to remove tiny/noisy detections)
        * Maximum aspect ratio (to remove long, thin boxes)
        * Confidence threshold

        Args:
            pil_image (PIL.Image): The image to analyze.
            conf_threshold (float): Minimum confidence for valid detection.

        Returns:
            list of (cropped_image, label, bbox): Valid image crops with YOLO label and bbox.
            Falls back to the full image if no valid detections.
        """
        results = self.model(pil_image)
        detections = results[0]

        # Filtering parameters
        min_crop_area = 5000
        max_aspect_ratio = 4.0

        crops_with_labels = []
        for box in detections.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            aspect_ratio = max(w / h, h / w)

            if area < min_crop_area or aspect_ratio > max_aspect_ratio:
                continue

            # Retrieve the predicted label from model class mapping
            class_idx = int(box.cls[0])
            if hasattr(self.model, "names"):
                label = self.model.names[class_idx]
            else:
                label = str(class_idx)

            crop = pil_image.crop((x1, y1, x2, y2))
            bbox_tuple = (x1, y1, x2, y2)
            crops_with_labels.append((crop, label, bbox_tuple))

        if crops_with_labels:
            return crops_with_labels

        # Fallback: return the full image if no valid detections
        return [(pil_image, "unknown", (0, 0, pil_image.width, pil_image.height))]
