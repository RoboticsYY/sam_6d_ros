import json
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from pose_interfaces.srv import GetPose
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class PoseEstimation(Node):
    def __init__(self):
        super().__init__("pose_estimation")
        self._declare_parameters()
        self.sam6d_root = self.get_parameter("sam6d_root").value
        self.output_dir = self.get_parameter("output_dir").value
        self.cad_path = self.get_parameter("cad_path").value
        self.templates_dir = self.get_parameter("templates_dir").value
        self.detector_path = self.get_parameter("detector_path").value
        self.sam2_variant = self.get_parameter("sam2_variant").value
        self.device = self.get_parameter("device").value
        self.precision = self.get_parameter("precision").value
        self.detector_threshold = self.get_parameter("detector_threshold").value
        self._validate_paths()

        self._lock = threading.Lock()
        self._rgb = None
        self._depth = None
        self._camera_matrix = None
        self._depth_scale = None
        self.create_subscription(
            Image, self.get_parameter("rgb_topic").value,
            self._color_callback, qos_profile_sensor_data)
        self.create_subscription(
            Image, self.get_parameter("depth_topic").value,
            self._depth_callback, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._camera_info_callback, qos_profile_sensor_data)
        self.create_service(GetPose, "get_pose/sam6d", self._handle_get_pose)
        self.get_logger().info(
            f"Ready: YOLO + new OpenVINO SAM-6D on {self.device}/{self.precision}; "
            "replayed ROS CameraInfo is authoritative")

    def _declare_parameters(self):
        defaults = {
            "sam6d_root": "",
            "output_dir": "",
            "cad_path": "",
            "templates_dir": "",
            "detector_path": "",
            "sam2_variant": "sam2_hiera_small",
            "inference_python": "/home/intel/miniforge3/envs/ov_sam6d/bin/python",
            "device": "GPU",
            "precision": "fp16",
            "detector_threshold": 0.2,
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/aligned_depth_to_color/camera_info",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _validate_paths(self):
        required = [
            self.sam6d_root,
            self.cad_path,
            self.templates_dir,
            os.path.join(self.detector_path, "yolo.xml"),
            os.path.join(
                self.sam6d_root, "Instance_Segmentation_Model", "checkpoints",
                "ov_models", self.sam2_variant, "ov_image_encoder.xml"),
            os.path.join(
                self.sam6d_root, "Instance_Segmentation_Model", "checkpoints",
                "ov_models", self.sam2_variant, "ov_mask_predictor.xml"),
            os.path.join(
                self.sam6d_root, "Pose_Estimation_Model", "model_save",
                "ov_pem_model_cpu.xml"),
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError("Missing required SAM-6D assets: " + ", ".join(missing))
        os.makedirs(os.path.join(self.output_dir, "templates"), exist_ok=True)
        if os.path.realpath(self.templates_dir) != os.path.realpath(
                os.path.join(self.output_dir, "templates")):
            output_templates = os.path.join(self.output_dir, "templates")
            if not os.listdir(output_templates):
                os.rmdir(output_templates)
                os.symlink(self.templates_dir, output_templates)

    def _color_callback(self, message):
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.width, -1)[:, :, :3]
        if message.encoding.lower() == "bgr8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._rgb = image.copy()

    def _depth_callback(self, message):
        dtype = np.uint16 if message.encoding.lower() in ("16uc1", "mono16") else np.float32
        depth = np.frombuffer(message.data, dtype=dtype).reshape(message.height, message.width)
        depth_scale = 1.0 if dtype == np.uint16 else 1000.0
        with self._lock:
            self._depth = depth.copy()
            self._depth_scale = depth_scale

    def _camera_info_callback(self, message):
        bag_matrix = np.asarray(message.k).reshape(3, 3)
        with self._lock:
            first_message = self._camera_matrix is None
            self._camera_matrix = bag_matrix
        if first_message:
            self.get_logger().info(
                f"Using bag calibration: width={message.width}, height={message.height}, "
                f"fx={bag_matrix[0, 0]:.6f}, fy={bag_matrix[1, 1]:.6f}, "
                f"cx={bag_matrix[0, 2]:.6f}, cy={bag_matrix[1, 2]:.6f}")

    def _handle_get_pose(self, request, response):
        response.pose = Pose()
        response.pose.orientation.w = 1.0
        response.ret = 1
        if request.command_id not in (1, 2):
            self.get_logger().error(
                "command_id must be 1 (YOLO + SAM + PEM) or 2 (ISM + PEM)")
            return response
        with self._lock:
            rgb = None if self._rgb is None else self._rgb.copy()
            depth = None if self._depth is None else self._depth.copy()
            camera_matrix = None if self._camera_matrix is None else self._camera_matrix.copy()
            depth_scale = self._depth_scale
        if rgb is None or depth is None or camera_matrix is None or depth_scale is None:
            self.get_logger().error(
                "No color/depth/camera-info frame received; replay the ROS2 bag first")
            return response
        try:
            start = time.perf_counter()
            frame_dir = os.path.join(self.output_dir, "ros_frame")
            os.makedirs(frame_dir, exist_ok=True)
            rgb_path = os.path.join(frame_dir, "rgb.png")
            depth_path = os.path.join(frame_dir, "depth.png")
            camera_path = os.path.join(frame_dir, "camera_from_bag.json")
            pipeline = "yolo-sam" if request.command_id == 1 else "ism"
            result_path = os.path.join(frame_dir, f"worker_result_{pipeline}.json")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_path, depth.astype(np.uint16))
            with open(camera_path, "w", encoding="utf-8") as camera_file:
                json.dump({
                    "cam_K": camera_matrix.reshape(-1).tolist(),
                    "depth_scale": depth_scale,
                }, camera_file)
            command = [
                self.get_parameter("inference_python").value,
                os.path.join(os.path.dirname(__file__), "sam6d_worker.py"),
                "--sam6d-root", self.sam6d_root,
                "--output-dir", self.output_dir,
                "--cad-path", self.cad_path,
                "--templates-dir", self.templates_dir,
                "--camera-path", camera_path,
                "--detector-path", self.detector_path,
                "--rgb-path", rgb_path,
                "--depth-path", depth_path,
                "--result-path", result_path,
                "--device", self.device,
                "--precision", self.precision,
                "--detector-threshold", str(self.detector_threshold),
                "--pipeline", pipeline,
                "--sam2-variant", self.sam2_variant,
            ]
            subprocess.run(command, check=True)
            with open(result_path, encoding="utf-8") as result_file:
                result = json.load(result_file)
            translation = result["translation_m"]
            quaternion = result["quaternion_xyzw"]
            response.pose.position.x, response.pose.position.y, response.pose.position.z = translation
            response.pose.orientation.x, response.pose.orientation.y = quaternion[:2]
            response.pose.orientation.z, response.pose.orientation.w = quaternion[2:]
            response.ret = 0
            score_detail = (
                f"YOLO={result['yolo_score']:.4f}, SAM={result['sam_score']:.4f}"
                if result["pipeline"] == "yolo-sam"
                else f"ISM={result['ism_score']:.4f}")
            self.get_logger().info(
                f"Pose complete in {time.perf_counter() - start:.3f}s; {score_detail}")
        except Exception as error:
            self.get_logger().error(f"Pose inference failed: {error}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimation()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()