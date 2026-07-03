#!/usr/bin/env python3
"""ROS2 node that captures webcam feed and runs VLM inference on demand."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMshowcase")))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

SHOWCASE_DIR = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMshowcase")
YOLO_DIR = Path("/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection/yolo11-26/models")
AVAILABLE_MODELS = sorted(YOLO_DIR.glob("yolo2[16]*.pt")) + sorted(YOLO_DIR.glob("yolo11*.pt"))


class VLMWebcamNode(Node):
    def __init__(self):
        super().__init__("vlm_webcam_node")
        self.bridge = CvBridge()
        self.model_key = self.declare_parameter("model", "yolo26n").value
        self.confidence = self.declare_parameter("confidence", 0.25).value
        self.publish_rate = self.declare_parameter("rate", 10.0).value
        self.show_display = self.declare_parameter("display", True).value

        model_path = YOLO_DIR / f"{self.model_key}.pt"
        if not model_path.exists():
            self.get_logger().error(f"Model not found: {model_path}")
            raise FileNotFoundError(f"Model {model_path} not found")

        self.get_logger().info(f"Loading YOLO model: {model_path.name}")
        self.model = YOLO(str(model_path))
        self.get_logger().info(f"Model loaded. Available models: {[p.stem for p in AVAILABLE_MODELS[:5]]}...")

        self.publisher_ = self.create_publisher(Image, "vlm_detections", 10)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open webcam")
            raise RuntimeError("Webcam not available")

        self.get_logger().info(
            f"VLM Webcam Node started — model={self.model_key} conf={self.confidence} rate={self.publish_rate}Hz"
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return

        results = self.model(frame, conf=self.confidence, verbose=False)
        annotated = frame.copy()

        for r in results:
            if r.boxes is not None:
                for box, cls, conf_val in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{r.names[int(cls)]} {conf_val:.2f}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if r.keypoints is not None:
                for kps in r.keypoints.data.cpu().numpy():
                    for kp in kps:
                        x, y, conf_kp = int(kp[0]), int(kp[1]), kp[2]
                        if conf_kp > self.confidence:
                            cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

        msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

        if self.show_display:
            n_objs = len(r.boxes) if results and r.boxes is not None else 0
            cv2.putText(annotated, f"YOLO {self.model_key} | {n_objs} objects | q=quit",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("VLMshowcase ROS2 — Webcam", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.get_logger().info("User requested quit")
                rclpy.shutdown()
                sys.exit(0)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = VLMWebcamNode()
        rclpy.spin(node)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
