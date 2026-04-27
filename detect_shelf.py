# detect_shelf.py
from ultralytics import YOLO
import cv2

# โหลดโมเดลที่เทรนแล้ว
model = YOLO("best.pt")  # เปลี่ยน path ตามจริง

# ตรวจจับจากภาพ
image_path = "path_to_your_image.jpg"
results = model(image_path)

# แสดงผลลัพธ์
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = model.names[cls]
        print(f"พบ {label} ที่ตำแหน่ง ({int(x1)},{int(y1)}) ความมั่นใจ {conf:.2f}")

# แสดงภาพพร้อม bounding box (optional)
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO("runs/train/shelf_detection/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Shelf Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()