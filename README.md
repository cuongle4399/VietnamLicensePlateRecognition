# Vietnam License Plate Recognition (Nhận Diện Biển Số Xe Việt Nam)

Dự án phát hiện và nhận diện biển số xe Việt Nam sử dụng YOLOv8 và PaddleOCR, giao diện Web Dashboard (Flask) tiện lợi hỗ trợ chạy trên máy tính và truy cập bằng điện thoại qua cùng mạng nội bộ (VLAN/Wi-Fi).

## 🔗 Nguồn Dataset
- [Roboflow Vietnam License Plate Dataset v2](https://universe.roboflow.com/nguyn-khang-0apuu/vietnam-license-plate-hjswj-xcnw3/dataset/2)

---

## 1. Cấu Trúc Thư Mục
```
detecteLicensePlate/
├── dataset/                  # Dataset (train, valid, test và data.yaml)
├── weights/                  # Trọng số mô hình (weights/best.pt)
├── src/                      # Xử lý Logic (detector.py, ocr.py)
├── templates/                # Giao diện HTML
├── static/                   # CSS & JavaScript
├── main.py                   # Điểm khởi chạy Web Server (Flask)
└── requirements.txt          # Thư viện cần cài đặt
```

---

## 2. Cài Đặt (Windows CMD)

Mở terminal tại thư mục dự án và chạy:
```cmd
# Tạo và kích hoạt môi trường ảo
python -m venv venv
call venv\Scripts\activate

# Cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Huấn Luyện YOLOv8
```cmd
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=8 device=0
```
*Lưu ý: Sau khi train xong, copy file `best.pt` từ `runs/detect/train/weights/best.pt` vào thư mục `weights/best.pt`.*

---

## 4. Chạy Web Server & Kết Nối VLAN

1. Kích hoạt môi trường ảo và chạy:
   ```cmd
   python main.py
   ```
2. Giao diện server sẽ tự động tìm kiếm địa chỉ IP trong mạng cục bộ của bạn và in ra terminal:
   ```text
   ======================================================================
    VIETNAM LICENSE PLATE RECOGNITION WEB APP ACTIVE
    - Local URL:   http://localhost:5000
    - Network URL: http://192.168.1.15:5000
    Connect your phone or other device in the same VLAN / Wi-Fi to the Network URL.
   ======================================================================
   ```
3. **Cách sử dụng trên điện thoại:**
   - Đảm bảo điện thoại kết nối chung Wi-Fi/VLAN với máy tính chạy server.
   - Nhập địa chỉ **Network URL** (ví dụ: `http://192.168.1.15:5000`) vào trình duyệt điện thoại.
   - Nhấn **📸 Take Photo** để trực tiếp mở camera của điện thoại và chụp ảnh xe. Ứng dụng sẽ tự động phân tích và hiển thị kết quả ngay lập tức!
   - Hoặc nhấn **📁 Browse Files** (hoặc vùng Drag & Drop) để tải ảnh có sẵn từ thư viện ảnh.
