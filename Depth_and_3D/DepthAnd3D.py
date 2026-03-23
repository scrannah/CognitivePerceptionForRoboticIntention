import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from reachy_mini import ReachyMini


class DepthToQSR:

    def __init__(self, fov_degrees=120.0, fov_type="diagonal", model_name="DPT_Large"):
        # camera field of view
        self.fov_degrees = fov_degrees
        self.fov_type = fov_type

        # load MiDaS model
        # hybrid might be better suited later if using time series
        self.midas = torch.hub.load("intel-isl/MiDaS", model_name)
        self.midas.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.midas = self.midas.to(self.device)

        # load MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

        # camera intrinsics, filled later once image size known
        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None


    def estimate_depth(self, img_rgb):
        # apply MiDaS transform
        input_batch = self.transform(img_rgb)
        input_batch = input_batch.cuda()

        # run model
        with torch.no_grad():
            pred = self.midas(input_batch)

        # resize prediction to original image size
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

        # convert tensor -> numpy
        depth = pred.cpu().numpy()

        return depth


    def show_results(self, frame_rgb, depth):
        # show RGB and depth map
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.title("RGB")
        plt.imshow(frame_rgb)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("MIDAS relative depth")
        plt.imshow(depth)
        plt.axis("off")

        plt.show()

        print("depth shape:", depth.shape,
              "min/max:", float(depth.min()), float(depth.max()))


    def compute_intrinsics(self, depth):
        # get height and width
        h, w = depth.shape

        # diagonal length (pythagoras)
        d = np.sqrt(w**2 + h**2)

        # optical centre
        self.cx = w / 2.0
        self.cy = h / 2.0

        # convert FOV to radians
        fov = np.deg2rad(self.fov_degrees)

        # focal length calculation
        if self.fov_type == "horizontal" or self.fov_type == "vertical":

            self.fx = (w / 2.0) / np.tan(fov / 2.0)

        elif self.fov_type == "diagonal":

            self.fx = (d / 2.0) / np.tan(fov / 2.0)

        else:
            raise ValueError(f"Invalid fov_type '{self.fov_type}'. Must be 'horizontal', 'vertical', or 'diagonal'.") # for use in other camera models

        # assume square pixels
        self.fy = self.fx


    def bbox_center(self, bbox):
        # bbox format [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        # centre of bounding box
        # later this comes from upstream
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        return u, v


    def pixel_to_3d(self, u, v, depth):
        # numpy arrays indexed row, column [v, u]
        z = float(depth[v, u])

        # convert pixel coordinates to camera coordinates
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return x, y, z


    def package_object(self, label, x, y, z):
        # package one object for QSR
        return {
            "label": label,
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }


    def package_scene(self, frame_id, timestamp, objects):
        # package whole scene/frame for QSR
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects": objects
        }

    def process_detection(self, detection, depth):

        bbox = detection["bbox"]
        label = detection["label"]

        # get centre pixel of detection
        u, v = self.bbox_center(bbox)

        # convert centre pixel to 3d, consider using median depth for bounding box
        # or use a cropped centre to remove background
        x, y, z = self.pixel_to_3d(u, v, depth)

        # package just the object info
        result = self.package_object(label, x, y, z) # get object as a list

        return result


    def process_image(self, image_rgb, detections, frame_id, timestamp):
        # pipeline
        depth = self.estimate_depth(image_rgb)


        # compute camera parameters once depth size known
        self.compute_intrinsics(depth)

        objects = [] # empty list to start from, each frame needs new object list

        # process each detection
        for detection in detections: # going over the detections in frame list
            result = self.process_detection(detection, depth) # for each detection on depth map, process
            objects.append(result) # append each detection result to object list


        scene_package = self.package_scene(frame_id, timestamp, objects) # package full scene as frame, with objects within

        print(scene_package)
        return image_rgb, depth, scene_package # original frame, depth map, package