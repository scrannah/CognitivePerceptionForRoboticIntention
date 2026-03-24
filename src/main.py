import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO
from src.Yolo_and_Conceptnet import get_info
from reachy_mini import ReachyMini
from src.Depth_and_3D import DepthToQSR

model = YOLO("../yolo26l-seg.pt") # segmentation helps noisy bounding box results
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

depth_processor = DepthToQSR()
conceptnet_cache = {}
frame_id = 0
collected_frames = []

def build_detection(label, x1, y1, x2, y2):
    return {
        "label"   : label,
        "centre_x": int((x1 + x2) / 2),
        "centre_y": int((y1 + y2) / 2),
        "bbox"    : (int(x1), int(y1), int(x2), int(y2))
    } # keep as fallback for bad masks and visualisation

def build_segmentation(label, centre_x, centre_y, bbox):
    return {
        "label"   : label,
        "centre_x": int(centre_x),
        "centre_y": int(centre_y),
        "bbox"    : (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    } # main method


def get_frame(mini):
    # Wait until first frame arrives
    while True: # repeat until frame
        frame = mini.media.get_frame()
        if frame is not None:
            timestamp = time.time()
            return frame, timestamp
        time.sleep(0.05)  # not ready yet, wait and try again


with ReachyMini(media_backend="default", host="172.20.10.4", connection_mode="network") as mini: # declare ip here to prevent it defaulting to local
    # library may need edits if refusing to connect
    time.sleep(3)  # give stream time to start

    for _ in range (500): # how many frames to collect, maybe change to a timer

        frame_id += 1

        frame, frame_timestamp = get_frame(mini)
        frame_display = frame.copy() # copy to write for display

        with torch.no_grad(): # reduce comp load, have yolo in eval
            detection_results = model(frame, conf=0.5, verbose=False)

        detections_in_frame = []
        for result in detection_results:
            boxes = result.boxes
            masks = result.masks
            for i in range(len(boxes)):
                box = boxes[i] # iterate through boxes
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                object_label   = model.names[int(box.cls[0])]
                confidence_score = float(box.conf[0])

                if masks is not None:
                    mask = masks.data[i].cpu().numpy() # iterate through masks, put to cpu for numpy
                    mask = cv2.resize(mask, (frame_display.shape[1], frame_display.shape[0])) # resize to frame display, there's a mismatch between yolo and display
                    y, x = np.where(mask > 0) # where there is a pixel value

                    if len(x) > 0 and len(y) > 0: # and if the mask has worked
                        bbox = (x1, y1, x2, y2)
                        centre_x = x.mean() # calculate centroid based on mask
                        centre_y = y.mean()
                        detection = build_segmentation(object_label, centre_x, centre_y ,bbox)
                    else:
                        detection = build_detection(object_label, x1, y1, x2, y2)  # mask fallback
                else:
                    detection = build_detection(object_label, x1, y1, x2, y2)  # mask fallback

                detections_in_frame.append(detection)

                if object_label not in conceptnet_cache:
                    print(f"[ConceptNet] Looking up '{object_label}'...")
                    conceptnet_cache[object_label] = get_info(object_label)
                concept_info = conceptnet_cache[object_label]


                cv2.rectangle(frame_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                mask_binary = (mask > 0).astype(np.uint8) # draw mask contours instead of rectangle
                colour = np.array([0, 0, 255], dtype=np.uint8)
                coloured_mask = np.zeros_like(frame_display)
                coloured_mask[mask_binary == 1] = colour
                frame_display = cv2.addWeighted(frame_display, 1.0, coloured_mask, 0.5, 0)

                cv2.putText(frame_display, f"{object_label} {confidence_score:.0%}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame_display, (detection["centre_x"], detection["centre_y"]), 5, (0, 0, 255), -1)

                if concept_info:
                    relation = list(concept_info.keys())[0]
                    fact = concept_info[relation][0]
                    cv2.putText(frame_display, f"{relation}: {fact}", (int(x1), int(y2) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        if len(detections_in_frame) > 0: # don't append frames with no detections, could be a vision error
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb, depth, scene_package = depth_processor.process_image(
                frame_rgb, detections_in_frame, frame_id, frame_timestamp
            ) # attach frame id and timestamp to get full scene package in depth pipeline
            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth = depth.astype(np.uint8)
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
            cv2.imshow("depth image:", depth) # test output
            collected_frames.append(scene_package)


        cv2.imshow("Reachy Camera", frame_display)  # got a frame, show it
        cv2.waitKey(1)

# print(collected_frames)
cv2.destroyAllWindows()