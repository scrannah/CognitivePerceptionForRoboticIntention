from reachy_mini import ReachyMini
import cv2

with ReachyMini(media_backend="default") as mini:
    for _ in range(100):  # try 100 frames
        frame = mini.media.get_frame()
        print(f"Frame: {type(frame)}, value: {frame is None}")

        if frame is None:
            continue

        print(f"Frame shape: {frame.shape}")
        cv2.imshow("Reachy Camera", frame)
        cv2.waitKey(1)

cv2.destroyAllWindows()