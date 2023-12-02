import easyocr
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
license_plate_detector = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/best.pt')
reader = easyocr.Reader(['en'], gpu=True)

ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    cv2.imshow('Feed Capture', frame)
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow('crop',license_plate_crop)
            bbox, license_plate_text, license_plate_text_score = reader.readtext(license_plate_crop_thresh)
            print(license_plate_text)
    if cv2.waitKey(1) == ord('q'):
        break 
