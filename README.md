# Phát hiện và Nhận dạng Ký tự trên Biển số Xe bằng Kỹ thuật Xử lý Ảnh số

Dự án nghiên cứu và phát triển hệ thống **Phát hiện và Nhận dạng Ký tự trên Biển số Xe Việt Nam** kết hợp kỹ thuật xử lý ảnh số và học sâu (Deep Learning). Hệ thống sử dụng mô hình **YOLOv8** để định vị (Object Detection) và **PaddleOCR** để trích xuất ký tự (OCR).

## 🔗 Nguồn Dataset
- [Roboflow Vietnam License Plate Dataset v2](https://universe.roboflow.com/nguyn-khang-0apuu/vietnam-license-plate-hjswj-xcnw3/dataset/2)

---

## 1. Quy Trình Hệ Thống (Pipeline)
1. **Phát hiện & Cắt biển số (YOLOv8)**: Định vị vùng biển số xe trên ảnh đầu vào, crop vùng biển số và tối ưu hóa vẽ bounding box.
2. **Nhận dạng ký tự (PaddleOCR)**: Quét và đọc các vùng chứa ký tự chữ và số.
3. **Kỹ thuật xử lý ảnh số & Hậu xử lý**:
   - **Sắp xếp không gian (Spatial Sorting)**: Nhóm hộp chữ theo tọa độ tâm trục Y để phân hàng (biển số vuông 2 dòng), sau đó sắp xếp theo trục X từ trái qua phải để nối chuỗi ký tự đúng thứ tự.
   - **Sửa lỗi ký tự (Confusion Corrector)**: Ánh xạ và chuyển đổi tự động các lỗi nhận diện ký tự nhiễu dựa theo quy tắc phân bổ biển số xe Việt Nam (ví dụ: chuyển chữ `O` thành số `0` ở phần tỉnh thành, sê-ri hoặc số đuôi).

---

## 2. Cấu Trúc Thư Mục
```
detecteLicensePlate/
├── dataset/                  # Dataset (train, valid, test và data.yaml)
├── weights/                  # Trọng số mô hình (weights/best.pt)
├── src/                      # Xử lý logic (detector.py, ocr.py)
├── templates/ & static/      # Giao diện Web Dashboard (HTML, CSS, JS)
├── main.py                   # Điểm khởi chạy Web Server (Flask)
└── requirements.txt          # Danh sách thư viện Python
```

---

## 3. Cài Đặt (Windows CMD)
```cmd
# Tạo và kích hoạt môi trường ảo
python -m venv venv
call venv\Scripts\activate

# Cài đặt thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Huấn Luyện YOLOv8
```cmd
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=8 device=0
```
*Lưu ý: Sau khi train xong, sao chép file `best.pt` từ `runs/detect/train/weights/best.pt` vào thư mục `weights/best.pt`.*

---

## 5. Khởi Chạy & Kết Nối VLAN (Điện thoại di động)
1. Khởi động Web Server:
   ```cmd
   python main.py
   ```
2. Giao diện máy chủ Flask sẽ hiển thị thông tin kết nối cục bộ và địa chỉ IP trong mạng nội bộ:
   ```text
   ======================================================================
    VIETNAM LICENSE PLATE RECOGNITION WEB APP ACTIVE
    - Local URL:   http://localhost:5000
    - Network URL: http://192.168.1.15:5000
   ======================================================================
   ```
3. Kết nối điện thoại chung Wi-Fi/VLAN với máy tính, truy cập **Network URL** qua trình duyệt trên điện thoại. Nhấn **📸 Take Photo** để mở trực tiếp camera chụp ảnh xe nhận diện.
