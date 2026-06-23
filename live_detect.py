from ultralytics import YOLO
import cv2
from picamera2 import Picamera2

model = YOLO("best (1).onnx")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

while True:
    frame = picam2.capture_array()
    results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Live Detection", annotated)
    print("Vehicles:", len(results[0].boxes))
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()