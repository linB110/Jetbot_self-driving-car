#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import yaml
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from yolo_detection.msg import detection as YoloDetection
from std_msgs.msg import String

class LidarYoloFusion(object):
    def __init__(self):
        # === get YAML pre-defiend parameters ===
        yaml_path = rospy.get_param("~config_file",
                                    "/home/lab605/catkin_ws/src/lidar/config/lidar_camera.yaml")

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # ===== camera intrinsics =====
        cam = cfg["camera"]
        self.camera_matrix = np.array(cam["camera_matrix"]["data"]).reshape(3,3)
        self.fx = self.camera_matrix[0,0]
        self.fy = self.camera_matrix[1,1]
        self.cx = self.camera_matrix[0,2]
        self.cy = self.camera_matrix[1,2]

        # ===== extrinsics (LiDAR -> Camera) =====
        ext = cfg["extrinsics"]
        self.R = np.array(ext["rotation"]["data"]).reshape(3,3)
        self.T = np.array(ext["translation"]["data"]).reshape(3,1)

        # ===== topics =====
        self.scan_topic   = cfg["lidar"]["scan_topic"]         # e.g. "/scan"
        self.yolo_topic   = rospy.get_param("~yolo_topic", "/yolo/detection_node")

        fusion_cfg = cfg.get("fusion", {})
        self.threshold = fusion_cfg.get("threshold", 0.5)
        self.pixel_radius = fusion_cfg.get("pixel_radius", 20)

        # store YOLO latest detection
        self.latest_detection = None

        # Subscribers
        rospy.Subscriber(self.yolo_topic, YoloDetection, self.yolo_cb, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)

        self.valid_motion_pub = rospy.Publisher("/valid_motion", String, queue_size=1)
        rospy.loginfo("LidarYoloFusion init")

        rospy.loginfo("LidarYoloFusion init")
        rospy.loginfo("  yolo_topic   = %s", self.yolo_topic)
        rospy.loginfo("  scan_topic   = %s", self.scan_topic)
        rospy.loginfo("  threshold    = %.3f m", self.threshold)
        rospy.loginfo("  pixel_radius = %d px", self.pixel_radius)

    def yolo_cb(self, msg):

        self.latest_detection = msg

    def scan_cb(self, scan_msg):
        """
        fuse YOLO detectio with latest LiDAR /scan
        """
        det = self.latest_detection
        if det is None:
            return

        cx_det = det.cx
        cy_det = det.cy

        angle = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        min_depth = None  

        # traverse all range
        for r in scan_msg.ranges:
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                angle += angle_increment
                continue

            # LiDAR frame: x_l front, y_l left, z_l = 0
            x_l = r * math.cos(angle)
            y_l = r * math.sin(angle)
            z_l = 0.0
            angle += angle_increment

            # LiDAR -> Camera
            p_l = np.array([[x_l],
                            [y_l],
                            [z_l]])
            p_c = self.R.dot(p_l) + self.T
            Xc, Yc, Zc = p_c.flatten()

            if Zc <= 0.0:
                # ignore all points behind camera
                continue

            # project to image plane
            u = self.fx * Xc / Zc + self.cx
            v = self.fy * Yc / Zc + self.cy

            # determind LiDAR projected result is within YOLO (cx, cy) or not
            du = u - cx_det
            dv = v - cy_det
            dist2 = du*du + dv*dv

            if dist2 <= self.pixel_radius * self.pixel_radius:
                # within center -> consider as a hit
                if (min_depth is None) or (r < min_depth):
                    min_depth = r

        if min_depth is not None:
            rospy.loginfo_throttle(
                0.5,
                "Detected %s, depth = %.3f m (threshold=%.3f)" %
                (det.class_name, min_depth, self.threshold)
            )

            # lower than threshold
            is_close = (min_depth < self.threshold)

            # lower than threshold -> valid motion instruction
            if is_close:
                self.valid_motion_pub.publish(String(data=det.class_name))



if __name__ == "__main__":
    rospy.init_node("lidar_yolo_fusion")
    node = LidarYoloFusion()
    rospy.spin()
