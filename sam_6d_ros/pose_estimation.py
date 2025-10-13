import rclpy
from rclpy.node import Node
from sam_6d_ros.instance_segmentation_model import model
from sam_6d_ros.instance_segmentation_model.utils.inout import get_root_project
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

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(Image.fromarray(rgb), (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def visualize_all_masks(rgb, detections, save_dir="tmp_masks"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    img_base = rgb.copy()
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33
    images = []

    for mask_idx, det in enumerate(detections):
        img = img_base.copy()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]
        img[edge, :] = 255

        img_pil = Image.fromarray(np.uint8(img))
        save_path = os.path.join(save_dir, f"mask_{mask_idx}.png")
        img_pil.save(save_path)
        images.append(img_pil)

    return images

class PoseEstimation:
    def __init__(self, node: Node):
        self.node = node
        self.img_np = None
        self.img_depth_np = None
        self.cam_k = None
        self.depth_scale = None
        self.visualize = True

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

        self.depth_scale = node.get_parameter('depth_scale').value
        self.depth_scale = np.array(self.depth_scale).astype(np.float32)
        self.node.get_logger().info(f'Depth scale is: {self.depth_scale}')

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
        self.ism_model = instantiate(self.cfg.model)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ism_model.descriptor_model.model = self.ism_model.descriptor_model.model.to(self.device)
        self.ism_model.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.ism_model.segmentor_model, "predictor"):
            self.ism_model.segmentor_model.predictor.model = (
                self.ism_model.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.ism_model.segmentor_model.model.setup_model(device=self.device, verbose=True)
        self.node.get_logger().info(f"Moving models to {self.device} done!")

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
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.ism_model.ref_data = {}
        self.ism_model.ref_data["descriptors"] = self.ism_model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        self.ism_model.ref_data["appe_descriptors"] = self.ism_model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
        self.node.get_logger().info("Initializing template done!")

        mesh = trimesh.load_mesh(self.cad_path)
        self.model_points = mesh.sample(2048).astype(np.float32) / 1000.0

    def color_callback(self, msg):
        # Convert ROS Image message to numpy array (OpenCV format)
        self.img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # If encoding is not 'rgb8', convert using cv2
        if msg.encoding.lower() == 'bgr8':
            self.img_np = cv2.cvtColor(self.img_np, cv2.COLOR_BGR2RGB)
        # self.node.get_logger().info(f'Received color image: {msg.header.stamp}, shape: {self.img_np.shape}, dtype: {self.img_np.dtype}')
        # Now img_np is a (height, width, 3) uint8 RGB array        

    def depth_info_callback(self, msg):
        # Convert camera intrinsic matrix to (3, 3) numpy array
        K = np.array(msg.k).reshape((3, 3))
        self.cam_k = K
        # self.node.get_logger().info(f'Camera intrinsic matrix: {K}')

    def aligned_depth_callback(self, msg):
        # Convert ROS Image message to numpy array for depth
        depth_np = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        # Convert to int32 for consistency
        depth_np = depth_np.astype(np.int32)
        self.img_depth_np = depth_np
        # self.node.get_logger().info(f'Received aligned depth image: {msg.header.stamp}, shape: {self.img_depth_np.shape}, dtype: {self.img_depth_np.dtype}')
        # Each value in img_depth_np is the depth at that pixel

    def handle_get_pose(self, request, response):
        # Dummy implementation: always return identity pose
        self.run_segmentation_inference()
        response.pose = Pose()
        return response
    
    def batch_input_data(self, device):
        batch = {}
        batch["depth"] = torch.from_numpy(self.img_depth_np).unsqueeze(0).to(device)
        batch["cam_intrinsic"] = torch.from_numpy(self.cam_k).unsqueeze(0).to(device)
        batch['depth_scale'] = torch.from_numpy(self.depth_scale).unsqueeze(0).to(device)
        return batch

    def get_obj_poses_from_template_level(
        self, level, pose_distribution, return_cam=False, return_index=False
    ):
        root_project = get_root_project()
        if return_cam:
            obj_poses_path = os.path.join(
                root_project, f"utils/poses/predefined_poses/cam_poses_level{level}.npy"
            )
            obj_poses = np.load(obj_poses_path)
        else:
            obj_poses_path = os.path.join(
                root_project, f"utils/poses/predefined_poses/obj_poses_level{level}.npy"
            )
            obj_poses = np.load(obj_poses_path)
        print(obj_poses_path)

        if pose_distribution == "all":
            if return_index:
                index = np.arange(len(obj_poses))
                return index, obj_poses
            else:
                return obj_poses
        elif pose_distribution == "upper":
            cam_poses_path = os.path.join(
                root_project, f"utils/poses/predefined_poses/cam_poses_level{level}.npy"
            )
            cam_poses = np.load(cam_poses_path)
            if return_index:
                index = np.arange(len(obj_poses))[cam_poses[:, 2, 3] >= 0]
                return index, obj_poses[cam_poses[:, 2, 3] >= 0]
            else:
                return obj_poses[cam_poses[:, 2, 3] >= 0]

    def run_segmentation_inference(self):
        # Placeholder for segmentation inference logic
        detections = self.ism_model.segmentor_model.generate_masks(np.array(self.img_np))
        detections = Detections(detections)
        query_decriptors, query_appe_descriptors = self.ism_model.descriptor_model.forward(np.array(self.img_np), detections)
        
        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.ism_model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.ism_model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        # compute the geometric score
        batch = self.batch_input_data(self.device)
        template_poses = self.get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.ism_model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

        self.ism_model.ref_data["pointcloud"] = torch.tensor(self.model_points).unsqueeze(0).data.to(self.device)

        image_uv = self.ism_model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = self.ism_model.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.ism_model.visible_thred
            )
        
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))

        if self.visualize:
            detections.to_numpy()
            save_path = f"{self.output_dir}/sam6d_results/detection_ism"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
            detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
            save_json_bop23(save_path+".json", detections)
            vis_img = visualize(self.img_np, detections, f"{self.output_dir}/sam6d_results/vis_ism.png")
            visualize_all_masks(self.img_np, detections, save_dir=f"{self.output_dir}/sam6d_results/tmp_masks")
            vis_img.save(f"{self.output_dir}/sam6d_results/vis_ism.png")

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
