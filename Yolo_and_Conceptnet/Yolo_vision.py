import cv2
import torch
import time
from ultralytics import YOLO
from Yolo_and_Conceptnet import get_info
from reachy_mini import ReachyMini
from Depth_and_3D import DepthToQSR

model = YOLO("yolov8n.pt")
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

depth_processor = DepthToQSR()
conceptnet_cache = {}
frame_id = 0
collected_frames = []


def build_detection(label, x1, y1, x2, y2):
    return {
        "label": label,
        "center_x": int((x1 + x2) / 2),
        "center_y": int((y1 + y2) / 2),
        "bbox": (int(x1), int(y1), int(x2), int(y2))
    }


def get_frame(mini):
    # Wait until first frame arrives
    while True:  # repeat until frame
        frame = mini.media.get_frame()
        if frame is not None:
            timestamp = time.time()
            return frame, timestamp
        time.sleep(0.05)  # not ready yet, wait and try again


time.sleep(3)
with ReachyMini(media_backend="default", host="172.20.10.4", connection_mode="network") as mini:
    time.sleep(3)  # give stream time to start

    for _ in range(500):  # how many frames to collect

        frame_id += 1

        frame, frame_timestamp = get_frame(mini)
        frame = frame.copy()

        with torch.no_grad():
            detection_results = model(frame, conf=0.5, verbose=False)

        detections_in_frame = []
        for result in detection_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                object_label = model.names[int(box.cls[0])]
                confidence_score = float(box.conf[0])

                detection = build_detection(object_label, x1, y1, x2, y2)
                detections_in_frame.append(detection)

                if object_label not in conceptnet_cache:
                    print(f"[ConceptNet] Looking up '{object_label}'...")
                    conceptnet_cache[object_label] = get_info(object_label)
                concept_info = conceptnet_cache[object_label]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{object_label} {confidence_score:.0%}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (detection["center_x"], detection["center_y"]), 5, (0, 0, 255), -1)

                if concept_info:
                    relation = list(concept_info.keys())[0]
                    fact = concept_info[relation][0]
                    cv2.putText(frame, f"{relation}: {fact}", (int(x1), int(y2) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 220, 255), 1)

        if len(detections_in_frame) > 0:  # don't append frames with no detections, could be a vision error
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb, depth, scene_package = depth_processor.process_image(
                frame_rgb, detections_in_frame, frame_id, frame_timestamp
            )  # attach frame id and timestamp to get full scene package in depth pipeline
            cv2.imshow("depth image:", depth)
            collected_frames.append(scene_package)

        cv2.imshow("Reachy Camera", frame)  # got a frame, show it
        cv2.waitKey(1)

# print(collected_frames)
cv2.destroyAllWindows()