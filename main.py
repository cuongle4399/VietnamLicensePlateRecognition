import os
import sys
import cv2
import time
import base64
import socket
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify

# Import core modules
from src.detector import LicensePlateDetector
from src.ocr import LicensePlateOCR

app = Flask(__name__)

# Lock for thread safety in deep learning inference
inference_lock = threading.Lock()

# Initialize AI models globally
print("[INFO] Loading YOLOv8 Plate Detector...")
detector = LicensePlateDetector()
print("[INFO] Loading PaddleOCR Reader...")
ocr = LicensePlateOCR()
print("[INFO] All models loaded successfully!")

def get_local_ip():
    """
    Retrieves the PC's local IP address on the network
    so mobile devices can connect to it.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a public server (doesn't send packet)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

@app.route("/")
def index():
    """Serves the main dashboard user interface."""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Handles image upload, runs YOLOv8 detection, crops regions,
    runs PaddleOCR character recognition, and returns JSON results.
    """
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided."}), 400
            
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename."}), 400

        # Read confidence threshold from client (default to 0.25)
        conf_threshold = float(request.form.get("conf", 0.25))

        # Convert uploaded image bytes to OpenCV format
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Invalid image format."}), 400

        # Run pipeline in a thread-safe lock block
        with inference_lock:
            start_time = time.time()
            annotated_img, crops, detections = detector.detect(img, conf_threshold)
            
            # Run OCR on all crops
            plates = []
            for crop in crops:
                plate_text, ocr_conf = ocr.read_plate(crop)
                
                # Encode cropped plate to base64
                _, crop_buf = cv2.imencode(".jpg", crop)
                crop_base64 = base64.b64encode(crop_buf).decode("utf-8")
                
                plates.append({
                    "crop_image": f"data:image/jpeg;base64,{crop_base64}",
                    "text": plate_text,
                    "confidence": round(ocr_conf * 100, 1)
                })
                
            elapsed_time = round(time.time() - start_time, 3)

        # Encode annotated output image to base64
        _, ann_buf = cv2.imencode(".jpg", annotated_img)
        ann_base64 = base64.b64encode(ann_buf).decode("utf-8")

        return jsonify({
            "success": True,
            "annotated_image": f"data:image/jpeg;base64,{ann_base64}",
            "plates": plates,
            "elapsed_time": elapsed_time
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000
    print("\n" + "="*70)
    print(f" VIETNAM LICENSE PLATE RECOGNITION WEB APP ACTIVE")
    print(f" - Local URL:   http://localhost:{port}")
    print(f" - Network URL: http://{local_ip}:{port}")
    print(" Connect your phone or other device in the same VLAN / Wi-Fi to the Network URL.")
    print("="*70 + "\n")
    
    # Run server on all interfaces (0.0.0.0)
    app.run(host="0.0.0.0", port=port, debug=False)
