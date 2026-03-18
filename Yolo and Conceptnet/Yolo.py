import cv2
from ultralytics import YOLO
from conceptnet import get_info

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam did not open!")
    exit()

print("Webcam is running! Press Q to stop.")

cache = {}

def package(label, x1, y1, x2, y2):
    return {
        "label"   : label,
        "center_x": int((x1 + x2) / 2),
        "center_y": int((y1 + y2) / 2),
        "bbox"    : (int(x1), int(y1), int(x2), int(y2))
    }

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    frame_detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label      = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            det = package(label, x1, y1, x2, y2)
            frame_detections.append(det)

            if label not in cache:
                print(f"[ConceptNet] Looking up '{label}'...")
                cache[label] = get_info(label)
            cn = cache[label]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.0%}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (det["center_x"], det["center_y"]), 5, (0, 0, 255), -1)

            if cn:
                rel  = list(cn.keys())[0]
                fact = cn[rel][0]
                cv2.putText(frame, f"{rel}: {fact}", (int(x1), int(y2) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    if frame_detections:
        print("\n── Detections ──")
        for d in frame_detections:
            print(f"  {d}")

    cv2.imshow("YOLO + ConceptNet  |  Q = Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")