import cv2
from ultralytics import YOLO
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from collections import defaultdict, Counter
import sqlite3
import pandas as pd

def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

def setup_database(db_path="car_records.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS cars (
            track_id INTEGER PRIMARY KEY,
            licence_plate TEXT
        )
    """)
    conn.commit()
    return conn, c

def save_to_db(c, conn, track_id, plate):
    c.execute("INSERT OR REPLACE INTO cars (track_id, licence_plate) VALUES (?, ?)",
              (track_id, plate))
    conn.commit()

def export_db_to_csv(conn, csv_path="car_records.csv"):
    df = pd.read_sql_query("SELECT * FROM cars", conn)
    df.to_csv(csv_path, index=False)

def detect_cars(model, frame):
    results = model(frame, verbose=False)
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label in ["car", "truck", "bus", "motorbike"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, label))
    return detections

def detect_plate_text(alpr_model, reader, cropped_car, offset=(0,0)):
    plate_texts = []
    boxes = []
    if cropped_car.size == 0:
        return plate_texts, boxes
    try:
        alpr_results = alpr_model(cropped_car, verbose=False)
        for plate in alpr_results[0].boxes:
            px1, py1, px2, py2 = map(int, plate.xyxy[0])
            plate_crop = cropped_car[py1:py2, px1:px2]
            plate_crop = preprocess_plate(plate_crop)
            if plate_crop.size == 0:
                continue
            results_ocr = reader.readtext(plate_crop, detail=0)
            if results_ocr:
                plate_texts.extend(results_ocr)
                abs_x1, abs_y1 = offset[0] + px1, offset[1] + py1
                abs_x2, abs_y2 = offset[0] + px2, offset[1] + py2
                boxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
    except Exception as e:
        print("ALPR Error:", e)
    return plate_texts, boxes

def main(video_path="4K Road traffic video for object detection and tracking - free download now!.mp4"):
    car_model = YOLO("yolov8l.pt")
    alpr_model = YOLO("OverWatch-002D.pt")
    tracker = DeepSort(max_age=30)
    reader = easyocr.Reader(['en'])
    conn, c = setup_database()
    plate_history = defaultdict(list)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        detections = detect_cars(car_model, frame)
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cropped_car = frame[t:b, l:r]
            plate_texts, plate_boxes = detect_plate_text(alpr_model, reader, cropped_car, offset=(l,t))
            plate_history[track_id].extend(plate_texts)

            for bx in plate_boxes:
                abs_x1, abs_y1, abs_x2, abs_y2 = bx
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)

            if plate_history[track_id]:
                most_common_plate = Counter(plate_history[track_id]).most_common(1)[0][0]
                cv2.putText(frame, most_common_plate, (l, b + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                save_to_db(c, conn, track_id, most_common_plate)

        cv2.imshow("IRIS OVER-WATCH", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    export_db_to_csv(conn)
    conn.close()
    print("Database exported to car_records.csv")

if __name__ == "__main__":
    main()
