import rclpy
from rclpy.node import Node
from sam_6d_ros.instance_segmentation_model import model
from sam_6d_ros.instance_segmentation_model.utils.inout import get_root_project
from sensor_msgs.msg import Image as RosImage, CameraInfo as RosCameraInfo
from sam_6d_interfaces.srv import GetPose
from geometry_msgs.msg import Pose
from ament_index_python.packages import get_package_share_directory

import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from PIL import Image
import logging
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
from hydra.utils import instantiate
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
from torch.serialization import safe_globals
from ultralytics.nn.tasks import SegmentationModel

import gorilla
import random
import importlib
import json

import torchvision.transforms as transforms

from sam_6d_ros.pose_estimation_model.utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from sam_6d_ros.pose_estimation_model.utils.draw_utils import draw_detections, calculate_2d_projections, draw_3d_pts
import pycocotools.mask as cocomask

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

def visualize_ism(rgb, detections, save_path="tmp.png"):
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

def visualize_pem(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz


def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose

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

        visualize_param = node.get_parameter('visualize').value
        self.visualize = bool(visualize_param)
        self.node.get_logger().info(f'Visualize is: {self.visualize}')

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
        # Allowlist classes needed for unpickling FastSAM (PyTorch >=2.6 weights_only security)
        try:
            torch.serialization.add_safe_globals([SegmentationModel, nn.Sequential])
        except Exception as e:
            self.node.get_logger().warn(f"Failed registering safe globals: {e}")
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

        self.mesh = trimesh.load_mesh(self.cad_path)
        self.model_points = self.mesh.sample(2048).astype(np.float32) / 1000.0

        # Load pose estimation model configuration
        config_path = os.path.join(get_package_share_directory('sam_6d_ros'), 'pose_estimation_model', 'config', 'base.yaml')
        self.pem_cfg = gorilla.Config.fromfile(config_path)
        self.pem_cfg.model_name = 'sam_6d_ros.pose_estimation_model.model.pose_estimation_model'
        self.pem_cfg.det_score_thresh = 0.2
        
        print("=> creating model ...")
        MODEL = importlib.import_module(self.pem_cfg.model_name)
        self.pem_model = MODEL.Net(self.pem_cfg.model)
        self.pem_model = self.pem_model.cuda()
        self.pem_model.eval()
        checkpoint = os.path.join(os.getcwd(), 'pose_estimation_model', 'checkpoints', 'sam-6d-pem-base.pth')
        gorilla.solver.load_checkpoint(model=self.pem_model, filename=checkpoint)
        print(f"=> loaded checkpoint '{checkpoint}'")

        print("=> extracting templates ...")
        tem_path = os.path.join(self.output_dir, 'templates')
        all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, self.pem_cfg.test_dataset)
        with torch.no_grad():
            self.all_tem_pts, self.all_tem_feat = self.pem_model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)
        print("=> initialization done!")

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
        self.run_pose_estimation_inference()
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

    def get_test_data(self):
        # Collect detections above threshold and keep only top 3 by score
        dets = [det for det in self.detections if det['score'] > self.pem_cfg.det_score_thresh]
        dets.sort(key=lambda d: d['score'], reverse=True)
        dets = dets[:3]
        print("ninstance (top3 after threshold): ", len(dets))

        whole_depth = self.img_depth_np * self.depth_scale / 1000.0
        whole_pts = get_point_cloud_from_depth(whole_depth, self.cam_k)
        print("Point cloud shape:", whole_pts.shape)

        radius = np.max(np.linalg.norm(self.model_points, axis=1))
        print("radius: ", radius)

        all_rgb = []
        all_cloud = []
        all_rgb_choose = []
        all_score = []
        all_dets = []
        for inst in dets:
            seg = inst['segmentation']
            score = inst['score']

            # Normalize segmentation to binary mask
            mask = None
            if isinstance(seg, dict) and 'size' in seg:  # our RLE-like dict
                try:
                    h, w = seg['size']
                    rle = cocomask.frPyObjects(seg, h, w)
                    mask = cocomask.decode(rle)
                except Exception:
                    pass
            if mask is None:
                # seg might already be decoded or list of counts (our mask_to_rle format)
                if isinstance(seg, dict) and 'counts' in seg and 'size' in seg:
                    try:
                        mask = cocomask.decode({'counts': seg['counts'], 'size': seg['size']})
                    except Exception:
                        pass
            if mask is None:
                # fallback: treat seg as ndarray
                if isinstance(seg, np.ndarray):
                    mask = seg
            if mask is None:
                print(f"Warning: could not parse segmentation for instance; skipping")
                continue
            if mask.ndim == 3:  # sometimes decode returns (H,W,1)
                mask = mask.squeeze(-1)

            mask = np.logical_and(mask > 0, whole_depth > 0)
            if np.sum(mask) > 32:
                bbox = get_bbox(mask)
                y1, y2, x1, x2 = bbox
            else:
                continue
            mask = mask[y1:y2, x1:x2]
            # mask_offset = np.zeros_like(mask)
            # mask_offset[:, 5:] = mask[:, :-5]
            # mask = mask_offset
            choose = mask.astype(np.float32).flatten().nonzero()[0]
            print("npts before cropping: ", len(choose))

            # pts
            cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :] 
            center = np.mean(cloud, axis=0)
            tmp_cloud = cloud - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            if np.sum(flag) < 4:
                continue
            choose = choose[flag]
            cloud = cloud[flag]
            print("npts after cropping: ", len(cloud))

            if len(choose) <= self.pem_cfg.test_dataset.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), self.pem_cfg.test_dataset.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.pem_cfg.test_dataset.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # rgb
            rgb = self.img_np.copy()[y1:y2, x1:x2, :][:,:,::-1]
            if self.pem_cfg.test_dataset.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
            rgb = cv2.resize(rgb, (self.pem_cfg.test_dataset.img_size, self.pem_cfg.test_dataset.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb_transform(np.array(rgb))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.pem_cfg.test_dataset.img_size)

            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(inst)

        ret_dict = {}
        print("all_cloud shape: ", len(all_cloud))
        ret_dict['pts'] = torch.stack(all_cloud).cuda()
        ret_dict['rgb'] = torch.stack(all_rgb).cuda()
        ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
        ret_dict['score'] = torch.FloatTensor(all_score).cuda()

        ninstance = ret_dict['pts'].size(0)
        ret_dict['model'] = torch.FloatTensor(self.model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        ret_dict['K'] = torch.FloatTensor(self.cam_k).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        return ret_dict, self.img_np, whole_pts.reshape(-1, 3), self.model_points, all_dets

    def run_segmentation_inference(self):
        # Placeholder for segmentation inference logic
        self.detections = self.ism_model.segmentor_model.generate_masks(np.array(self.img_np))
        self.detections = Detections(self.detections)
        query_decriptors, query_appe_descriptors = self.ism_model.descriptor_model.forward(np.array(self.img_np), self.detections)

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.ism_model.compute_semantic_score(query_decriptors)

        # update detections
        self.detections.filter(idx_selected_proposals)
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

        image_uv = self.ism_model.project_template_to_image(best_template, pred_idx_objects, batch, self.detections.masks)

        geometric_score, visible_ratio = self.ism_model.compute_geometric_score(
            image_uv, self.detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.ism_model.visible_thred
            )
        
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
        self.detections.add_attribute("scores", final_score)
        self.detections.add_attribute("object_ids", torch.zeros_like(final_score))

        # Convert to list-of-dicts (in-memory) before visualization / downstream use
        detection_list = self.detections.to_dict_list(scene_id=0, image_id=0, runtime=0.0, dataset_name="Custom")
        self.detections = detection_list
        if self.visualize:
            save_dir = f"{self.output_dir}/sam6d_results"
            os.makedirs(save_dir, exist_ok=True)
            save_json_bop23(os.path.join(save_dir, "detection_ism.json"), self.detections)
            vis_img = visualize_ism(self.img_np, self.detections, os.path.join(save_dir, "vis_ism.png"))
            visualize_all_masks(self.img_np, self.detections, save_dir=os.path.join(save_dir, "tmp_masks"))
            vis_img.save(os.path.join(save_dir, "vis_ism.png"))

    def run_detection_inference(self):
        # Placeholder for detection inference logic
        pass

    def run_pose_estimation_inference(self):
        # Placeholder for pose estimation logic
        print("=> addressing input data ...")
        input_data, img, whole_pts, model_points, detections = self.get_test_data()
        ninstance = input_data['pts'].size(0)

        print("=> running PEM model ...")
        with torch.no_grad():
            input_data['dense_po'] = self.all_tem_pts.repeat(ninstance,1,1)
            input_data['dense_fo'] = self.all_tem_feat.repeat(ninstance,1,1)
            out = self.pem_model(input_data)

        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000
        print("=> Got object pose")

        if self.visualize:
            for idx in range(ninstance):
                save_path = os.path.join(f"{self.output_dir}/sam6d_results", f'vis_pem_{idx}.png')
                K = input_data['K'][idx:idx+1].detach().cpu().numpy()  # shape (1, 3, 3)
                vis_img = visualize_pem(
                    img,
                    pred_rot[idx:idx+1],
                    pred_trans[idx:idx+1],
                    model_points*1000,
                    K,
                    save_path
                )
                print(f"pred_rot[{idx}]: ", pred_rot[idx])
                print(f"pred_trans[{idx}]: ", pred_trans[idx])
                vis_img.save(save_path)

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
