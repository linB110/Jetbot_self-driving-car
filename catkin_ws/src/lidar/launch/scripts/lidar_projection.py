#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2


class LidarLineProjection(object):
    def __init__(self):
        # === read parameters from YAML ===
        yaml_path = rospy.get_param("~config_file",
                                    "/home/lab605/catkin_ws/src/lidar/config/lidar_camera.yaml")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ===== camera intrinsics =====
        cam = cfg["camera"]
        self.image_topic = cam["image_topic"]
        self.camera_matrix = np.array(cam["camera_matrix"]["data"]).reshape(3,3)
        self.dist_coeffs   = np.array(cam["distortion_coeffs"]["data"])

        self.fx = self.camera_matrix[0,0]
        self.fy = self.camera_matrix[1,1]
        self.cx = self.camera_matrix[0,2]
        self.cy = self.camera_matrix[1,2]

        # ===== LiDAR topic =====
        self.scan_topic = cfg["lidar"]["scan_topic"]

        # ===== extrinsics =====
        ext = cfg["extrinsics"]
        self.R = np.array(ext["rotation"]["data"]).reshape(3,3)
        self.T = np.array(ext["translation"]["data"]).reshape(3,1)

        # ===== projection =====
        proj = cfg["projection"]
        self.point_size = proj["point_size"]
        self.color = tuple(proj["color"])

        # === establish ROS subscriber ===
        self.bridge = CvBridge()
        self.latest_image = None

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)

        self.image_pub = rospy.Publisher("/lidar_projected_image", Image, queue_size=1)

        rospy.loginfo("Loaded config from %s", yaml_path)
        rospy.loginfo("Using image_topic=%s", self.image_topic)
        rospy.loginfo("Using scan_topic=%s", self.scan_topic)


    def image_cb(self, msg):
        self.latest_image = msg

    def scan_cb(self, scan_msg):
        if self.latest_image is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(
                self.latest_image, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge error: %s", str(e))
            return

        img_h, img_w = cv_img.shape[:2]

        angle = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        step = 1
        points_uv = []

        # ---- debug ----
        raw_valid = 0        # r > 0 and not inf/NaN
        z_positive = 0       # Zc > 0 
        in_image = 0         # projected points

        for i in range(0, len(scan_msg.ranges), step):
            r = scan_msg.ranges[i]
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                angle += angle_increment * step
                continue

            raw_valid += 1

            # LiDAR frame: x_l front, y_l left, z_l = 0
            x_l = r * math.cos(angle)
            y_l = r * math.sin(angle)
            z_l = 0.0
            angle += angle_increment * step

            # ======= LiDAR -> Camera (using R, T) =======
            p_l = np.array([[x_l],
                            [y_l],
                            [z_l]])
            p_c = self.R.dot(p_l) + self.T  # 3x3 * 3x1 + 3x1
            Xc, Yc, Zc = p_c.flatten()

            if Zc <= 0.0:
                continue

            z_positive += 1

            # ======= project to image plane =======
            u = self.fx * Xc / Zc + self.cx
            v = self.fy * Yc / Zc + self.cy

            if 0 <= u < img_w and 0 <= v < img_h:
                points_uv.append((int(u), int(v)))
                in_image += 1

        # === Debug Info ===
        #rospy.loginfo_throttle(
        #    1.0,
        #    "Lidar raw_valid=%d, Z>0=%d, in_image=%d" %
        #    (raw_valid, z_positive, in_image)
        #)

        for (u, v) in points_uv:
            cv2.circle(cv_img, (u, v), self.point_size, self.color, -1)

        out_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        out_msg.header = self.latest_image.header
        self.image_pub.publish(out_msg)



if __name__ == "__main__":
    rospy.init_node("lidar_projection")
    node = LidarLineProjection()
    rospy.spin()
