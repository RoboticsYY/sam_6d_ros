import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage, CameraInfo as RosCameraInfo
from sam_6d_interfaces.srv import GetPose
from geometry_msgs.msg import Pose
# from ament_index_python.packages import get_package_share_directory

import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from sam_6d_ros.instance_segmentation_model.segment_anything.utils.amg import rle_to_mask

from sam_6d_ros.instance_segmentation_model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from sam_6d_ros.instance_segmentation_model.utils.bbox_utils import CropResizePad
from sam_6d_ros.instance_segmentation_model.model.utils import Detections, convert_npz_to_json
from sam_6d_ros.instance_segmentation_model.model.loss import Similarity
from sam_6d_ros.instance_segmentation_model.utils.inout import load_json, save_json_bop23

class PoseEstimation:
    def __init__(self, node: Node):
        self.node = node
        self.img_np = None
        self.img_depth_np = None
        self.camera_info = None
        self.color_sub = node.create_subscription(
            RosImage,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )
        self.depth_info_sub = node.create_subscription(
            RosCameraInfo,
            '/camera/aligned_depth_to_color/camera_info',
            self.depth_info_callback,
            10
        )
        self.aligned_depth_sub = node.create_subscription(
            RosImage,
            '/camera/aligned_depth_to_color/image_raw',
            self.aligned_depth_callback,
            10
        )
        self.srv = node.create_service(GetPose, 'get_pose', self.handle_get_pose)
        
        self.segmentor_model = node.get_parameter('segmentor_model').value
        self.node.get_logger().info(f'Segmenration model is: {self.segmentor_model}')

        self.output_dir = node.get_parameter('output_dir').value
        self.node.get_logger().info(f'Output directory is: {self.output_dir}')

        self.cad_path = node.get_parameter('cad_path').value
        self.node.get_logger().info(f'CAD file path is: {self.cad_path}')

        self.stability_score_thresh = node.get_parameter('stability_score_thresh').value
        self.node.get_logger().info(f'Stability score threshold is: {self.stability_score_thresh}')

        self.node.get_logger().info(f"Current working directory: {os.getcwd()}")
        with initialize(version_base=None, config_path='instance_segmentation_model/configs'):
            self.cfg = compose(config_name='run_inference.yaml')
        
        if self.segmentor_model == "sam":
            with initialize(version_base=None, config_path="instance_segmentation_model/configs/model"):
                self.cfg.model = compose(config_name='ISM_sam.yaml')
            self.cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh
        elif self.segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="instance_segmentation_model/configs/model"):
                self.cfg.model = compose(config_name='ISM_fastsam.yaml')
        else:
            raise ValueError("The segmentor_model {} is not supported now!".format(self.segmentor_model))        

        self.node.get_logger().info("Initializing model")
        self.model = instantiate(self.cfg.model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(device)
        self.model.descriptor_model.model.device = device
        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(device)
            )
        else:
            self.model.segmentor_model.model.setup_model(device=device, verbose=True)
        self.node.get_logger().info(f"Moving models to {device} done!")

        self.node.get_logger().info("Initializing template")
        template_dir = os.path.join(self.output_dir, 'templates')
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

        self.model.ref_data = {}
        self.model.ref_data["descriptors"] = self.model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        self.model.ref_data["appe_descriptors"] = self.model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
        self.node.get_logger().info("Initializing template done!")


    def color_callback(self, msg):
        # Convert ROS Image message to numpy array (OpenCV format)
        self.img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # If encoding is not 'rgb8', convert using cv2
        if msg.encoding.lower() == 'bgr8':
            self.img_np = cv2.cvtColor(self.img_np, cv2.COLOR_BGR2RGB)
        self.node.get_logger().info(f'Received color image: {msg.header.stamp}, shape: {self.img_np.shape}, dtype: {self.img_np.dtype}')
        # Now img_np is a (height, width, 3) uint8 RGB array        

    def depth_info_callback(self, msg):
        self.node.get_logger().info(f'Received camera info: {msg.header.stamp}')

    def aligned_depth_callback(self, msg):
        self.node.get_logger().info(f'Received aligned depth image: {msg.header.stamp}')

    def handle_get_pose(self, request, response):
        # Dummy implementation: always return identity pose
        response.pose = Pose()
        return response
    
    def run_segmentation_inference(self, image):
        # Placeholder for segmentation inference logic
        pass

    def run_detection_inference(self, image, depth_image, camera_info):
        # Placeholder for detection inference logic
        pass

    def run_pose_estimation_inference(self, image, depth_image, camera_info):
        # Placeholder for pose estimation logic
        pass

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('pose_estimation', automatically_declare_parameters_from_overrides=True)
    pose_estimation = PoseEstimation(node)

    while rclpy.ok():
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
