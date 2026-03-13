from reachy_mini import ReachyMini
import cv2
import time

frames = []
with ReachyMini(media_backend="default") as mini:
    time.sleep(3) # give stream time to start
    # Wait until first frame arrives
    while True:
        frame = mini.media.get_frame()
        print(frame.shape)
        frames.append(frame)
        if frame is None:
            time.sleep(0.05)  # not ready yet, wait and try again
            continue
        cv2.imshow("Reachy Camera", frame)  # got a frame, show it
        cv2.waitKey(1)


cv2.destroyAllWindows()