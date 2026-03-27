"""
detect.py -- Object detection module using YOLOv8.

Loads a pretrained YOLOv8 model and runs inference on individual frames.
Only returns detections that match the retail-relevant COCO classes
specified in the project configuration (person, bottle, cup, etc.).
"""

import os
import sys

from ultralytics import YOLO
import cv2

# Resolve project root for model path resolution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ObjectDetector:
    """
    Thin wrapper around the Ultralytics YOLO model.
    Keeps the model loaded in memory and exposes a single `detect` method
    that operates on raw BGR frames coming from OpenCV.
    """

    def __init__(self, config):
        weights = config["model"]["weights"]
        self.conf_thresh = config["model"]["confidence_threshold"]
        self.iou_thresh = config["model"]["iou_threshold"]

        # Classes we actually care about in a retail context
        self.target_class_ids = set(config["product_classes"] + [config["person_class"]])
        self.person_class_id = config["person_class"]

        # Reverse lookup: class id -> readable name
        self.class_names = {v: k for k, v in config["target_classes"].items()}

        # Resolve model path relative to project root
        if not os.path.isabs(weights):
            weights = os.path.join(_PROJECT_ROOT, weights)

        if not os.path.exists(weights):
            print(f"[detect] ERROR: Model weights not found at: {weights}")
            print("[detect] Please place yolov8n.pt in the models/ directory.")
            print("[detect] You can download it with:")
            print("         pip install ultralytics && yolo export model=yolov8n.pt")
            sys.exit(1)

        print(f"[detect] Loading YOLO model: {weights}")
        self.model = YOLO(weights)
        print("[detect] Model loaded successfully.")

    def detect(self, frame):
        """
        Run YOLO inference on a single BGR frame.

        Returns a list of dicts, each containing:
            - bbox: [x1, y1, x2, y2] in pixel coordinates
            - class_id: COCO class index
            - class_name: human-readable name
            - confidence: float 0-1
            - is_person: bool
        """
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in self.target_class_ids:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i].item())
                name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": round(conf, 3),
                    "is_person": cls_id == self.person_class_id,
                })

        return detections


def draw_detections(frame, detections):
    """
    Overlay bounding boxes and labels on the frame.
    Persons get a blue box; products get a green box.
    Uses anti-aliased text and padded label backgrounds for readability.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'

        if det["is_person"]:
            color = (255, 120, 0)   # blue-ish for people
        else:
            color = (0, 220, 100)   # green for products

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Padded dark background behind text for readability
        pad = 4
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame,
                      (x1, y1 - th - pad * 2),
                      (x1 + tw + pad * 2, y1),
                      color, -1)
        cv2.putText(frame, label, (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return frame
