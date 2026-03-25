import cv2
import time
import numpy as np
from reachy_mini import ReachyMini
from Yolo_and_Conceptnet import YOLOPipeline
from Depth_and_3D import DepthToQSR

class FullPipeline:

    def __init__(self):

        self.collected_frames = []
        self.frame_length = 500
        self.frame_id = 0

        self.YOLOPipeline = YOLOPipeline()
        self.DepthPipeline = DepthToQSR()
        # self.QSRPipeline = QSR()

    def get_frame(self, mini):
        # Wait until first frame arrives
        while True:  # repeat until frame
            frame = mini.media.get_frame()
            if frame is not None:
                frame_display = frame.copy()  # copy to write for display
                timestamp = time.time()
                return frame, timestamp, frame_display
            time.sleep(0.05)  # not ready yet, wait and try again

    def run(self):
        with ReachyMini(media_backend="default", host="172.20.10.4", connection_mode="network") as mini:  # declare ip here to prevent it defaulting to local
            # library may need edits if refusing to connect
            time.sleep(3)  # give stream time to start

            for _ in range(self.frame_length):  # how many frames to collect, maybe change to a timer
                self.frame_id += 1

                frame, frame_timestamp, frame_display = self.get_frame(mini)

                detection_results = self.YOLOPipeline.runYolo(frame)

                detections_in_frame = self.YOLOPipeline.processDetections(detection_results, frame)

                frame_display = self.YOLOPipeline.drawDetections(frame_display, detections_in_frame)

                if len(detections_in_frame) > 0:  # don't append frames with no detections, could be a vision error
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb, depth, scene_package = self.DepthPipeline.process_image(
                        frame_rgb, detections_in_frame, self.frame_id, frame_timestamp
                    )  # attach frame id and timestamp to get full scene package in depth pipeline
                    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                    depth = depth.astype(np.uint8)
                    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
                    cv2.imshow("depth image:", depth)  # test output
                    self.collected_frames.append(scene_package)

                    # TO QSR HERE

                cv2.imshow("Reachy Camera", frame_display)  # got a frame, show it
                cv2.waitKey(1)

        cv2.destroyAllWindows()

pipeline = FullPipeline()
pipeline.run()
