import os
import sys
import cv2
from ultralytics import YOLO

def get_resource_path(relative_path):
    """
    Resolves resource paths dynamically for PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class LicensePlateDetector:
    def __init__(self, model_path="weights/best.pt"):
        """
        Initializes the YOLOv8 License Plate Detector.
        If weights/best.pt is not found, falls back to yolov8n.pt.
        """
        # Resolve the weights path dynamically for PyInstaller support
        resolved_path = get_resource_path(model_path)
        self.model_path = resolved_path
        
        if not os.path.exists(resolved_path):
            print(f"[WARNING] Model weights not found at '{resolved_path}'.")
            fallback_path = get_resource_path("yolov8n.pt")
            print(f"Falling back to default '{fallback_path}'...")
            self.model = YOLO(fallback_path)
        else:
            self.model = YOLO(resolved_path)

    def detect(self, image_input, conf_threshold=0.25):
        """
        Runs object detection on the input image.
        image_input: Can be a file path (str) or a numpy array (loaded via cv2).
        conf_threshold: Minimum confidence score to filter detections.
        
        Returns:
            annotated_image (numpy array): Image with drawn bounding boxes.
            crops (list of numpy arrays): List of cropped license plates.
            detections (list of dict): Details of each detection (box, confidence, class_id).
        """
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        else:
            image = image_input.copy()

        # Run inference
        results = self.model(image, conf=conf_threshold)[0]
        
        # Draw bounding boxes and prepare crops
        annotated_image = image.copy()
        crops = []
        detections = []

        # Get box info
        boxes = results.boxes
        for box in boxes:
            # Get coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            # Store detection details
            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })

            # Crop license plate
            # Add a small margin to crop for better OCR context
            h, w = image.shape[:2]
            margin_x = int((x2 - x1) * 0.05)
            margin_y = int((y2 - y1) * 0.05)
            
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(w, x2 + margin_x)
            crop_y2 = min(h, y2 + margin_y)
            
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            crops.append(crop)

            # Draw rectangle and text on annotated image
            color = (0, 255, 0) # Green for license plates
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Label text
            label = f"Plate: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Adjust label position if near the top edge
            if y1 - label_h - 10 > 0:
                rect_y1 = y1 - label_h - 10
                rect_y2 = y1
                text_y = y1 - 5
            else:
                rect_y1 = y1
                rect_y2 = y1 + label_h + 10
                text_y = y1 + label_h + 5
                
            cv2.rectangle(annotated_image, (x1, rect_y1), (x1 + label_w, rect_y2), color, -1)
            cv2.putText(annotated_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated_image, crops, detections
