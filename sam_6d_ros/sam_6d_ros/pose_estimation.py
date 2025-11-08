# ROS2 related imports
import rclpy
from rclpy.node import Node
from sam_6d_ros.instance_segmentation_model.utils.inout import get_root_project
from sensor_msgs.msg import Image as RosImage, CameraInfo as RosCameraInfo
from pose_interfaces.srv import GetPose
from geometry_msgs.msg import Pose
from ament_index_python.packages import get_package_share_directory

# YOLO related imports
import torch
from torchvision.ops import nms

# ISM related imports
import os
import numpy as np
import time
import torch
import torch.nn as nn
from PIL import Image
import logging
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
from hydra.utils import instantiate
import glob
from omegaconf import OmegaConf
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from sam_6d_ros.instance_segmentation_model.segment_anything.utils.amg import rle_to_mask

from sam_6d_ros.instance_segmentation_model.utils.poses.pose_utils import load_index_level_in_level2
from sam_6d_ros.instance_segmentation_model.utils.bbox_utils import CropResizePad
from sam_6d_ros.instance_segmentation_model.model.utils import Detections
from sam_6d_ros.instance_segmentation_model.utils.inout import save_json_bop23
from ultralytics.nn.tasks import SegmentationModel

from sam_6d_ros.image_helper import preprocess_image, postprocess_masks, ResizeLongestSide

# PEM related imports
# Suppress NVML errors when CUDA is not available
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Environment setup for CUDA compatibility
def setup_cuda_environment():
    """Setup CUDA environment and handle NVML errors gracefully"""
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    # Try to import gpustat but handle the case when NVML is not available
    gpustat_module = None
    try:
        import gpustat
        gpustat_module = gpustat
        if cuda_available:
            print("GPU monitoring enabled")
        else:
            print("CUDA not available, GPU monitoring disabled")
    except ImportError:
        print("Warning: gpustat not available, GPU monitoring disabled")
    except Exception as e:
        if "NVML" in str(e) or "nvidia" in str(e).lower():
            print("Warning: NVIDIA Management Library not available, GPU monitoring disabled")
        else:
            print(f"Warning: Error importing gpustat: {e}")
    
    return cuda_available, gpustat_module

# Try to import gorilla with error handling
gorilla_module = None
try:
    import gorilla
    gorilla_module = gorilla
    print("Gorilla library imported successfully")
except Exception as e:
    if "NVML" in str(e) or "nvidia" in str(e).lower():
        print("Warning: Gorilla library not available due to NVML error, using fallback configuration")
        gorilla_module = None
    else:
        print(f"Warning: Error importing gorilla: {e}")
        gorilla_module = None

import torchvision.transforms as transforms

from openvino import Core
from sam_6d_ros.pose_estimation_model.utils.draw_utils import draw_detections

# Setup environment
CUDA_AVAILABLE, gpustat = setup_cuda_environment()

from sam_6d_ros.pose_estimation_model.utils.data_utils import load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose

import pycocotools.mask as cocomask

from sam_6d_ros.pose_estimation_model.utils.model_utils import compute_coarse_Rt, compute_fine_Rt

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

def visualize_yolo_detections(img, yolo_results, class_names=None, score_thresh=0.2):
    """
    Visualize YOLO detections on an image.
    Args:
        img (np.ndarray): Input RGB image (H, W, 3), uint8.
        yolo_results (dict or list): YOLO output, expects 'boxes', 'scores', 'labels'.
        class_names (list, optional): List of class names for labels.
        score_thresh (float): Minimum score to visualize.
    Returns:
        np.ndarray: Image with detections drawn.
    """
    img_vis = img.copy()
    # Support both dict and list output
    if isinstance(yolo_results, dict):
        boxes = yolo_results.get('boxes', [])
        scores = yolo_results.get('scores', [])
        labels = yolo_results.get('labels', [])
    elif isinstance(yolo_results, list):
        # List of dicts
        boxes = [d['box'] for d in yolo_results]
        scores = [d['score'] for d in yolo_results]
        labels = [d['label'] for d in yolo_results]
    else:
        # Return side-by-side with original if no detections
        orig_img_pil = Image.fromarray(img)
        img_vis_pil = Image.fromarray(img)
        concat_img = Image.new('RGB', (orig_img_pil.width + img_vis_pil.width, orig_img_pil.height))
        concat_img.paste(orig_img_pil, (0, 0))
        concat_img.paste(img_vis_pil, (orig_img_pil.width, 0))
        return np.array(concat_img)

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        # box: [x_center, y_center, w, h]
        x_center, y_center, w, h = map(int, box)
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x_center + w/2)
        y2 = int(y_center + h/2)
        color = (0, 255, 0)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        label_text = f"box: {score:.2f}"
        cv2.putText(img_vis, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Concatenate side by side with original image
    orig_img_pil = Image.fromarray(img)
    vis_img_pil = Image.fromarray(img_vis)
    concat_img = Image.new('RGB', (orig_img_pil.width + vis_img_pil.width, orig_img_pil.height))
    concat_img.paste(orig_img_pil, (0, 0))
    concat_img.paste(vis_img_pil, (orig_img_pil.width, 0))
    return np.array(concat_img)

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
    prediction = img.copy()
    
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

def visualize_pem(rgb, pred_rot, pred_trans, model_points, K):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    prediction = img.copy()

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


def get_templates(path, cfg, device):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).to(device))
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).to(device))
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).to(device))
    return all_tem, all_tem_pts, all_tem_choose

class OVPEM_Sub2(nn.Module):
    def __init__(self, cfg, npoint=2048):
        super(OVPEM_Sub2, self).__init__()
        self.cfg = cfg

    def forward(self, coarse_Rt_atten, sparse_pm, sparse_po, coarse_Rt_model_pts):
        init_R, init_t = compute_coarse_Rt(coarse_Rt_atten, sparse_pm, 
                                           sparse_po, coarse_Rt_model_pts)

        return init_R, init_t

class OVPEM_Sub4(nn.Module):
    def __init__(self, cfg, npoint=2048):
        super(OVPEM_Sub4, self).__init__()
        self.cfg = cfg

    def forward(self, fine_Rt_atten, dense_pm, dense_po_out, fine_Rt_model_pts):
        pred_R, pred_t, pred_pose_score = compute_fine_Rt(fine_Rt_atten, dense_pm, 
                                           dense_po_out, fine_Rt_model_pts)

        return pred_R, pred_t, pred_pose_score

class PoseEstimation:
    def __init__(self, node: Node):
        self.node = node
        self.img_np = None
        self.img_depth_np = None
        self.cam_k = None
        self.depth_scale = None
        self.visualize = True

        #=================Load parameters from ROS2 parameter server====================
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

        visualize_param = node.get_parameter('use_detection').value
        self.use_detection = bool(visualize_param)
        self.node.get_logger().info(f'Use detection is: {self.use_detection}')

        self.model_dir = node.get_parameter('model_dir').value
        self.node.get_logger().info(f'Model directory is: {self.model_dir}')

        self.ov_extension_lib_path = node.get_parameter('ov_extension_lib_path').value
        self.node.get_logger().info(f'OpenVINO extension library path is: {self.ov_extension_lib_path}')

        self.rgb_topic = node.get_parameter('rgb_topic').value
        self.node.get_logger().info(f'RGB topic is: {self.rgb_topic}')

        self.depth_topic = node.get_parameter('depth_topic').value
        self.node.get_logger().info(f'Depth topic is: {self.depth_topic}')

        self.camera_info_topic = node.get_parameter('camera_info_topic').value
        self.node.get_logger().info(f'Camera info topic is: {self.camera_info_topic}')

        #============Check topic existence before creating subscribers==========
        topic_names_and_types = node.get_topic_names_and_types()
        def topic_exists(topic_name, msg_type_str):
            return any(topic_name == t[0] and msg_type_str in t[1] for t in topic_names_and_types)

        if topic_exists(self.rgb_topic, 'sensor_msgs/msg/Image'):
            self.color_sub = node.create_subscription(
                RosImage,
                self.rgb_topic,
                self.color_callback,
                10
            )
        else:
            self.node.get_logger().warn(f"RGB topic '{self.rgb_topic}' does not exist or has wrong type.")

        if topic_exists(self.camera_info_topic, 'sensor_msgs/msg/CameraInfo'):
            self.depth_info_sub = node.create_subscription(
                RosCameraInfo,
                self.camera_info_topic,
                self.depth_info_callback,
                10
            )
        else:
            self.node.get_logger().warn(f"Camera info topic '{self.camera_info_topic}' does not exist or has wrong type.")

        if topic_exists(self.depth_topic, 'sensor_msgs/msg/Image'):
            self.aligned_depth_sub = node.create_subscription(
                RosImage,
                self.depth_topic,
                self.aligned_depth_callback,
                10
            )
        else:
            self.node.get_logger().warn(f"Depth topic '{self.depth_topic}' does not exist or has wrong type.")

        self.srv = node.create_service(GetPose, 'get_pose/sam6d', self.handle_get_pose)

        #==========Create a ROS2 publisher for visualization images==========
        self.image_pub = node.create_publisher(RosImage, 'pose_estimation/image', 10)

        #=====================Load instance segmentation model configuration====================
        self.node.get_logger().info(f"Current working directory: {os.getcwd()}")
        with initialize(version_base=None, config_path='instance_segmentation_model/configs'):
            self.ism_cfg = compose(config_name='run_inference.yaml')
        
        if self.segmentor_model == "sam":
            with initialize(version_base=None, config_path="instance_segmentation_model/configs/model"):
                self.ism_cfg.model = compose(config_name='ISM_sam.yaml')
            self.ism_cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh
        elif self.segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="instance_segmentation_model/configs/model"):
                self.ism_cfg.model = compose(config_name='ISM_fastsam.yaml')
        else:
            raise ValueError("The segmentor_model {} is not supported now!".format(self.segmentor_model))        

        self.node.get_logger().info("Initializing model")
        # Allowlist classes needed for unpickling FastSAM (PyTorch >=2.6 weights_only security)
        try:
            torch.serialization.add_safe_globals([SegmentationModel, nn.Sequential])
        except Exception as e:
            self.node.get_logger().warn(f"Failed registering safe globals: {e}")
        self.ism_model = instantiate(self.ism_cfg.model)
        
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
            
        if len(templates) == 0:
            raise RuntimeError(f"No template images found in {template_dir}. Please check your template generation or path.")
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
        self.model_points = self.mesh.sample(1024).astype(np.float32) / 1000.0

        #=====================Load pose estimation model configuration====================
        config_path = os.path.join(get_package_share_directory('sam_6d_ros'), 'pose_estimation_model', 'config', 'ov_gpu_base.yaml')
        # Load config directly from YAML file
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create config object with all attributes
        class Config:
            def __init__(self, data):
                for key, value in data.items():
                    if isinstance(value, dict):
                        setattr(self, key, Config(value))
                    else:
                        setattr(self, key, value)
        
        self.pem_cfg = Config(config_data)
        self.pem_cfg.model_name = 'sam_6d_ros.pose_estimation_model.model.pose_estimation_model'
        self.pem_cfg.det_score_thresh = 0.2
        self.pem_cfg.device = 'GPU'
        self.pem_cfg.model = 'ov_pose_estimation_model'
        print(f"[OpenVINO] PEM Using device: {self.pem_cfg.device}")

        print("=> creating model ...")
        self.core = Core()

        print("=> extracting templates ...")
        tem_path = os.path.join(self.output_dir, 'templates')
        all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, self.pem_cfg.test_dataset, self.device)

        rgb_input = torch.cat(all_tem, dim=1)            # (B, T*3, H, W)
        pts_input = torch.cat(all_tem_pts, dim=1)            # (B, T*N, 3)
        choose_input = torch.cat(all_tem_choose, dim=1)      # (B, T*N)
        ov_fe_input = {
            "rgb_input": rgb_input,
            "pts_input": pts_input,
            "choose_input": choose_input,
        }

        self.ov_model_init(self.pem_cfg.device)

        print("=> compute templates features ...")
        ov_fe_compiled_model = self.ov_model_list[0]
        ov_fe_results = ov_fe_compiled_model(ov_fe_input)
        ov_fe_results_list = list(ov_fe_results.values())
        self.all_tem_pts = ov_fe_results_list[0]  # tem_pts_out
        self.all_tem_feat = ov_fe_results_list[1]  # tem_feat
        self.first_run = True

        #==========Initialize and load OpenVINO YOLO model from model_dir/det/==========
        if self.use_detection:
            self.yolo_model = None
            self.yolo_compiled_model = None
            yolo_model_path = os.path.join(self.model_dir, "det", "yolo.xml")
            yolo_weights_path = os.path.join(self.model_dir, "det", "yolo.bin")
            if os.path.exists(yolo_model_path) and os.path.exists(yolo_weights_path):
                try:
                    self.yolo_model = self.core.read_model(yolo_model_path, yolo_weights_path)
                    self.yolo_compiled_model = self.core.compile_model(self.yolo_model, self.pem_cfg.device)
                    print(f"=> YOLO model loaded from {yolo_model_path} and compiled for device {self.pem_cfg.device}")
                except Exception as e:
                    self.node.get_logger().warn(f"Failed to load YOLO model: {e}")
            else:
                self.node.get_logger().warn(f"YOLO model files not found in {os.path.join(self.model_dir, 'det')}")

        #==========Initialize and load OpenVINO SAM from model_dir/========
        if self.use_detection:
            self.sam_ov_encoder_model = None
            self.sam_ov_encoder_compiled_model = None
            sam_model_encoder_path = os.path.join(self.model_dir, "sam_image_encoder.xml")
            sam_weights_encoder_path = os.path.join(self.model_dir, "sam_image_encoder.bin")
            if os.path.exists(sam_model_encoder_path) and os.path.exists(sam_weights_encoder_path):
                try:
                    self.sam_ov_encoder_model = self.core.read_model(sam_model_encoder_path, sam_weights_encoder_path)
                    self.sam_ov_encoder_compiled_model = self.core.compile_model(self.sam_ov_encoder_model, 'CPU')
                    print(f"=> SAM encoder model loaded from {sam_model_encoder_path} and compiled for device CPU")
                except Exception as e:
                    self.node.get_logger().warn(f"Failed to load SAM model: {e}")
            else:
                self.node.get_logger().warn(f"SAM model files not found in {os.path.join(self.model_dir, 'sam')}")


            self.sam_ov_predictor_model = None
            self.sam_ov_predictor_compiled_model = None
            sam_model_predictor_path = os.path.join(self.model_dir, "sam_mask_predictor.xml")
            sam_weights_predictor_path = os.path.join(self.model_dir, "sam_mask_predictor.bin")
            if os.path.exists(sam_model_predictor_path) and os.path.exists(sam_weights_predictor_path):
                try:
                    self.sam_ov_predictor_model = self.core.read_model(sam_model_predictor_path, sam_weights_predictor_path)
                    self.sam_ov_predictor_compiled_model = self.core.compile_model(self.sam_ov_predictor_model, 'CPU')
                    print(f"=> SAM predictor model loaded from {sam_model_predictor_path} and compiled for device CPU")
                except Exception as e:
                    self.node.get_logger().warn(f"Failed to load SAM model: {e}")
            else:
                self.node.get_logger().warn(f"SAM model files not found in {os.path.join(self.model_dir, 'sam')}")

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
        self.node.get_logger().info(f"Handling get_pose request")
        pose_msg = Pose()
        ret = 1
        try:
            if self.use_detection:
                self.run_detection_inference()
                self.run_prompt_segmentation_inference()
            else:
                self.run_instance_segmentation_inference()
            self.run_pose_estimation_inference()

            # Set response.pose from prediction
            # Use the first instance's prediction
            if hasattr(self, 'pred_rot') and hasattr(self, 'pred_trans'):
                # Convert rotation matrix to quaternion
                from scipy.spatial.transform import Rotation as R
                rot = R.from_matrix(self.pred_rot[0])
                quat = rot.as_quat()  # [x, y, z, w]
                pose_msg.orientation.x = quat[0]
                pose_msg.orientation.y = quat[1]
                pose_msg.orientation.z = quat[2]
                pose_msg.orientation.w = quat[3]
                pose_msg.position.x = float(self.pred_trans[0][0])
                pose_msg.position.y = float(self.pred_trans[0][1])
                pose_msg.position.z = float(self.pred_trans[0][2])
                ret = 0
            else:
                self.node.get_logger().warn("No pose prediction available, returning identity pose.")
                pose_msg.orientation.w = 1.0
                ret = 1
            
        except Exception as e:
            self.node.get_logger().error(f"Error during pose estimation: {e}")
            pose_msg.orientation.w = 1.0
            ret = 1
        response.pose = pose_msg
        response.ret = ret
        self.first_run = False
        self.node.get_logger().info(f"Pose estimation completed, sending response")
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
        # print(obj_poses_path)

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
        dets = dets[0:1]
        # print("ninstance (top1 after threshold): ", len(dets))

        whole_depth = self.img_depth_np * self.depth_scale / 1000.0
        whole_pts = get_point_cloud_from_depth(whole_depth, self.cam_k)
        # print("Point cloud shape:", whole_pts.shape)

        radius = np.max(np.linalg.norm(self.model_points, axis=1))
        # print("radius: ", radius)

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
            # print("npts before cropping: ", len(choose))

            # pts
            cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :] 
            center = np.mean(cloud, axis=0)
            tmp_cloud = cloud - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            if np.sum(flag) < 4:
                continue
            choose = choose[flag]
            cloud = cloud[flag]
            # print("npts after cropping: ", len(cloud))

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
        print("    all_cloud shape: ", len(all_cloud))
        ret_dict['pts'] = torch.stack(all_cloud).to(self.device)
        ret_dict['rgb'] = torch.stack(all_rgb).to(self.device)
        ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).to(self.device)
        ret_dict['score'] = torch.FloatTensor(all_score).to(self.device)

        ninstance = ret_dict['pts'].size(0)
        ret_dict['model'] = torch.FloatTensor(self.model_points).unsqueeze(0).repeat(ninstance, 1, 1).to(self.device)
        ret_dict['K'] = torch.FloatTensor(self.cam_k).unsqueeze(0).repeat(ninstance, 1, 1).to(self.device)
        return ret_dict, self.img_np, whole_pts.reshape(-1, 3), self.model_points, all_dets

    def ov_model_init(self, device="CPU"):
        ov_fe_model_path = os.path.join(self.model_dir, "ov_fe_model.xml")
        ov_pem_sub1_model_path = os.path.join(self.model_dir, "ov_pem_sub1_model.xml")
        ov_pem_sub2_model_path = os.path.join(self.model_dir, "ov_pem_sub2_model.xml")
        ov_pem_sub3_model_path = os.path.join(self.model_dir, "ov_pem_sub3_model.xml")
        ov_pem_sub4_model_path = os.path.join(self.model_dir, "ov_pem_sub4_model.xml")

        ov_gpu_kernel_path = os.path.join(get_package_share_directory('sam_6d_ros'), 'pose_estimation_model', 'config', 'ov_gpu_custom_op.xml')
        self.core.add_extension(self.ov_extension_lib_path)

        if device == "GPU":
            self.core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
            self.core.set_property("GPU", {"CONFIG_FILE": ov_gpu_kernel_path})

        # ov load models
        ov_fe_model = self.core.read_model(ov_fe_model_path)
        ov_pem_sub1_model = self.core.read_model(ov_pem_sub1_model_path)
        ov_pem_sub2_model = self.core.read_model(ov_pem_sub2_model_path)
        ov_pem_sub3_model = self.core.read_model(ov_pem_sub3_model_path)
        ov_pem_sub4_model = self.core.read_model(ov_pem_sub4_model_path)

        self.ov_fe_compiled_model = self.core.compile_model(ov_fe_model, device)
        self.ov_pem_sub1_model_compiled = self.core.compile_model(ov_pem_sub1_model, device)
        self.ov_pem_sub2_model_compiled = self.core.compile_model(ov_pem_sub2_model, "CPU")
        self.ov_pem_sub3_model_compiled = self.core.compile_model(ov_pem_sub3_model, device)
        self.ov_pem_sub4_model_compiled = self.core.compile_model(ov_pem_sub4_model, "CPU")

        self.ov_model_list = [self.ov_fe_compiled_model, 
                        self.ov_pem_sub1_model_compiled, self.ov_pem_sub2_model_compiled,
                        self.ov_pem_sub3_model_compiled, self.ov_pem_sub4_model_compiled,
                        ]

    def run_instance_segmentation_inference(self):
        # Placeholder for segmentation inference logic
        print("=> running ISM model ...")
        ism_time_start = time.time()
        ism_generates_masks_time_start = time.time()
        # Use top-1 bbox from detection for mask generation
        # if hasattr(self, 'detections_nms') and self.detections_nms and 'box' in self.detections_nms[0]:
        #     bbox = self.detections_nms[0]['box']
        #     self.detections = self.ism_model.segmentor_model.generate_masks_from_bbox(np.array(self.img_np), bbox)
        # else:
        self.detections = self.ism_model.segmentor_model.generate_masks(np.array(self.img_np))
        print(f"    Size of self.detections: {len(self.detections)}")
        ism_generates_masks_time = time.time() - ism_generates_masks_time_start
        self.detections = Detections(self.detections)
        ism_descriptor_time_start = time.time()
        query_decriptors, query_appe_descriptors = self.ism_model.descriptor_model.forward(np.array(self.img_np), self.detections)
        ism_descriptor_time = time.time() - ism_descriptor_time_start

        ism_compute_scores_time_start = time.time()
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
        ism_compute_scores_time = time.time() - ism_compute_scores_time_start
        
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
        self.detections.add_attribute("scores", final_score)
        self.detections.add_attribute("object_ids", torch.zeros_like(final_score))
        ism_time = time.time() - ism_time_start

        if not self.first_run:
            print(f"[OpenVINO {self.pem_cfg.device}] ISM inference time: {ism_time*1000:.2f} ms")
            print(f"    generates_masks time: {ism_generates_masks_time*1000:.2f} ms")
            print(f"    descriptor time: {ism_descriptor_time*1000:.2f} ms")
            print(f"    compute_scores time: {ism_compute_scores_time*1000:.2f} ms")

        # Convert to list-of-dicts (in-memory) before visualization / downstream use
        detection_list = self.detections.to_dict_list(scene_id=0, image_id=0, runtime=0.0, dataset_name="Custom")
        self.detections = detection_list
        if self.visualize:
            save_dir = f"{self.output_dir}/sam6d_results"
            os.makedirs(save_dir, exist_ok=True)
            save_json_bop23(os.path.join(save_dir, "detection_ism.json"), self.detections)
            vis_img = visualize_ism(self.img_np, self.detections, os.path.join(save_dir, "vis_ism.png"))
            # visualize_all_masks(self.img_np, self.detections, save_dir=os.path.join(save_dir, "tmp_masks"))
            # Publish vis_img as a ROS2 Image message
            vis_img_np = np.array(vis_img)
            ros_img_msg = RosImage()
            ros_img_msg.header.stamp = self.node.get_clock().now().to_msg()
            ros_img_msg.header.frame_id = "pose_estimation"
            ros_img_msg.height = vis_img_np.shape[0]
            ros_img_msg.width = vis_img_np.shape[1]
            ros_img_msg.encoding = "rgb8"
            ros_img_msg.is_bigendian = False
            ros_img_msg.step = vis_img_np.shape[1] * 3
            ros_img_msg.data = vis_img_np.tobytes()
            self.image_pub.publish(ros_img_msg)

    def run_prompt_segmentation_inference(self):

        print("=> running prompt-based segmentation inference ...")
        sam_start_time = time.time()
        if self.sam_ov_encoder_compiled_model is None or self.sam_ov_predictor_compiled_model is None:
            print("SAM models are not loaded.")
            return
        if self.img_np is None:
            print("No RGB image data available for SAM inference.")
            return
        if not hasattr(self, 'detections_nms') or not self.detections_nms:
            print("No YOLO detections available for SAM prompt segmentation.")
            return

        # Use top-1 bbox from YOLO detection
        bbox = self.detections_nms[0]['box']  # [x1, y1, x2, y2]
        # print(f"Using bbox for SAM prompt: {bbox}")
        # Prepare prompt for SAM (normalized bbox coordinates)
        x_center, y_center, bbox_width, bbox_height = bbox
        x1, y1 = x_center - bbox_width / 2, y_center - bbox_height / 2
        x2, y2 = x_center + bbox_width / 2, y_center + bbox_height / 2
        bbox_prompt = np.array([
            x1,
            y1,
            x2,
            y2
        ], dtype=np.float32)
        # print(f"bbox_prompt: {bbox_prompt}")

        # Prepare image for SAM encoder (float32, RGB, NCHW)
        # sam_input_shape = self.sam_ov_encoder_compiled_model.input(0).shape
        # print(f"SAM encoder input shape: {sam_input_shape}")
        resizer = ResizeLongestSide(1024)
        preprocessed_image = preprocess_image(self.img_np, resizer=resizer)

        # Run SAM encoder
        encoding_results = self.sam_ov_encoder_compiled_model(preprocessed_image)
        image_embeddings = encoding_results[self.sam_ov_encoder_compiled_model.output(0)]

        # Prepare prompt for predictor (SAM expects [1,5,2] for input 1)
        input_box = np.array([[x1, y1, x2, y2]])

        # Concatenate prompts and pad to shape [1,5,2] for SAM
        box_coords = input_box.reshape(2, 2)
        box_labels = np.array([2, 3])

        coord = np.concatenate([box_coords, np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])], axis=0)[None, :, :]
        label = np.concatenate([box_labels, np.array([-1, -1, -1])], axis=0)[None, :].astype(np.float32)

        coord = resizer.apply_coords(coord, self.img_np.shape[:2]).astype(np.float32)

        # Run SAM mask predictor
        inputs = {
            "image_embeddings": image_embeddings,
            "point_coords": coord,
            "point_labels": label,
        }
        predictor_output = self.sam_ov_predictor_compiled_model(inputs)

        masks = predictor_output[self.sam_ov_predictor_compiled_model.output(0)]
        masks = postprocess_masks(masks, self.img_np.shape[:-1], resizer=resizer)
        masks = masks > 0.0
        sam_end_time = time.time()
        # print(f"SAM generated mask shape: {masks.shape}")
        print(f"    SAM inference time: {(sam_end_time - sam_start_time)*1000:.2f} ms")
        # Ensure masks is 2D (height, width)
        if masks.ndim == 4:
            masks = masks[0, 0]
        elif masks.ndim == 3:
            masks = masks[0]
        # Now masks should be (height, width)

        # Save mask as self.detections for PEM model
        # Wrap in a list of dicts to match expected format
        self.detections = [{
            'segmentation': masks.astype(np.uint8),
            'score': 1.0,  # or use a real score if available
            'category_id': 1  # or set appropriately
        }]

        overlay = self.img_np.copy()
        alpha = 0.5  # Set transparency level (0.0 = fully transparent, 1.0 = fully opaque)
        overlay[masks] = (alpha * np.array([0, 255, 0]) + (1 - alpha) * overlay[masks]).astype(np.uint8)
        vis_img = np.concatenate([self.img_np, overlay], axis=1)

        # Publish vis_img as a ROS2 Image message
        ros_img_msg = RosImage()
        ros_img_msg.header.stamp = self.node.get_clock().now().to_msg()
        ros_img_msg.header.frame_id = "pose_estimation"
        ros_img_msg.height = vis_img.shape[0]
        ros_img_msg.width = vis_img.shape[1]
        ros_img_msg.encoding = "rgb8"
        ros_img_msg.is_bigendian = False
        ros_img_msg.step = vis_img.shape[1] * 3
        ros_img_msg.data = vis_img.tobytes()
        self.image_pub.publish(ros_img_msg)

    def run_detection_inference(self):
        # Placeholder for detection inference logic
        print("=> running detection inference ...")
        yolo_start_time = time.time()
        if self.yolo_compiled_model is None:
            print("YOLO model is not loaded.")
            return
        if self.img_np is None:
            print("No RGB image data available for YOLO inference.")
            return

        # Prepare input for YOLO model (assuming NCHW, float32, normalized)
        img = self.img_np
        # Use original image size for YOLO input
        input_shape = self.yolo_compiled_model.input(0).shape
        h, w = input_shape[2], input_shape[3]
        x_scale = img.shape[1] / w
        y_scale = img.shape[0] / h
        # print(f"Resizing image to YOLO input size: ({img.shape}, {w}, {h})")
        img_resized = cv2.resize(img, (w, h))
        img_rgb = img_resized.astype(np.float32) / 255.0
        img_rgb = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

        # Run inference
        results = self.yolo_compiled_model({self.yolo_compiled_model.input(0): img_rgb})
        yolo_end_time = time.time()
        print(f"    YOLO inference time: {(yolo_end_time - yolo_start_time)*1000:.2f} ms")
        # print("YOLO inference results:", results)
        # Adapt to YOLO output format: {output: array([[[x1, y1, x2, y2, conf, ...], ...]], dtype=float32)}
        yolo_output = results
        output_arr = yolo_output[next(iter(yolo_output))]  # shape: (1, 5, 8400)
        # Transpose to (8400, 5) for easier iteration
        arr = output_arr.squeeze().T  # shape: (8400, 5)
        detections = []
        for det in arr:
            x1, y1, x2, y2, conf = det[:5]
            x1 *= x_scale
            x2 *= x_scale
            y1 *= y_scale
            y2 *= y_scale
            if conf < 0.2:
                continue
            label = 0
            detections.append([x1, y1, x2, y2, conf, label])
            # print(f"Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}")

        # Apply NMS using torchvision.ops.nms
        boxes = torch.tensor([d[:4] for d in detections], dtype=torch.float32)
        scores = torch.tensor([d[4] for d in detections], dtype=torch.float32)
        labels = [d[5] for d in detections]
        # torchvision expects boxes as [x1, y1, x2, y2]
        keep_indices = nms(boxes, scores, iou_threshold=0.2).tolist()
        self.detections_nms = [
            {'box': [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])],
                'score': float(scores[i]), 'label': labels[i]}
            for i in keep_indices
        ]
        # Filter to keep only the bbox with highest confidence
        if self.detections_nms:
            self.detections_nms = [max(self.detections_nms, key=lambda d: d['score'])]
        # print(f"Detections after NMS (top-1): {self.detections_nms}")

        if self.visualize:
            vis_img_np = visualize_yolo_detections(self.img_np, self.detections_nms)
            # Publish vis_img as a ROS2 Image message
            ros_img_msg = RosImage()
            ros_img_msg.header.stamp = self.node.get_clock().now().to_msg()
            ros_img_msg.header.frame_id = "pose_estimation"
            ros_img_msg.height = vis_img_np.shape[0]
            ros_img_msg.width = vis_img_np.shape[1]
            ros_img_msg.encoding = "rgb8"
            ros_img_msg.is_bigendian = False
            ros_img_msg.step = vis_img_np.shape[1] * 3
            ros_img_msg.data = vis_img_np.tobytes()
            self.image_pub.publish(ros_img_msg)

    def run_pose_estimation_inference(self):
        # Placeholder for pose estimation logic
        print("=> addressing input data ...")
        input_data, img, whole_pts, model_points, detections = self.get_test_data()
        ninstance = input_data['pts'].size(0)

        pem_time_start = time.time()
        ninstance = input_data['pts'].shape[0]
        dense_po = np.repeat(self.all_tem_pts, ninstance, axis=0)
        dense_fo = np.repeat(self.all_tem_feat, ninstance, axis=0)

        pts = input_data['pts']
        rgb = input_data['rgb']
        rgb_choose = input_data['rgb_choose']
        model = input_data['model']

        print("=> running PEM model ...")
        ov_pem_sub1_model_compiled = self.ov_model_list[1]
        ov_pem_sub1_inputs = {
                "pts": pts,
                "rgb": rgb,
                "rgb_choose": rgb_choose,
                "model": model,
                "dense_po": dense_po,
                "dense_fo": dense_fo,
            }
        time_start = time.time()
        ov_pem_sub1_results = ov_pem_sub1_model_compiled(ov_pem_sub1_inputs)
        ov_pem_sub1_results = list(ov_pem_sub1_results.values())
        pem_sub1_time = time.time() - time_start
        # print(f"    [OpenVINO {device}] ov_pem_sub1 inference time: {pem_sub1_time*1000:.2f} ms")

        coarse_Rt_atten = ov_pem_sub1_results[0]
        sparse_pm = ov_pem_sub1_results[1]
        sparse_po = ov_pem_sub1_results[2]
        coarse_Rt_model_pts = ov_pem_sub1_results[3]
        dense_pm = ov_pem_sub1_results[4]
        dense_fm = ov_pem_sub1_results[5]
        geo_embedding_m = ov_pem_sub1_results[6]
        fps_idx_m = ov_pem_sub1_results[7]
        dense_po_out = ov_pem_sub1_results[8] 
        dense_fo_out = ov_pem_sub1_results[9]
        geo_embedding_o = ov_pem_sub1_results[10]
        fps_idx_o = ov_pem_sub1_results[11]
        radius = ov_pem_sub1_results[12]

        # ===============[Pass]OpenVINO Sub1 model ===========================
        torch_pem_sub2_input =  (torch.from_numpy(coarse_Rt_atten), torch.from_numpy(sparse_pm), 
                                torch.from_numpy(sparse_po), torch.from_numpy(coarse_Rt_model_pts))
        torch_flag = True
        if torch_flag:
            pem_sub2_model = OVPEM_Sub2(self.pem_cfg.model)
            torch_device = torch.device("xpu")
            pem_sub2_model.to(torch_device)
            torch_pem_input_new = []
            for tmp_tensor in torch_pem_sub2_input:
                torch_pem_input_new.append(tmp_tensor.to(torch_device))
            time_start = time.time()
            with torch.no_grad():
                init_R, init_t = pem_sub2_model(*torch_pem_input_new)
                init_R = init_R.cpu().numpy()
                init_t = init_t.cpu().numpy()
            torch.xpu.empty_cache()
            pem_sub2_time = time.time() - time_start
            # print(f"    [Pytorch xpu] pem_sub2 inference time: {pem_sub2_time*1000:.2f} ms")

        else:
            ov_pem_sub2_input = {
                "coarse_Rt_atten": coarse_Rt_atten,
                "sparse_pm": sparse_pm,
                "sparse_po": sparse_po,
                "coarse_Rt_model_pts": coarse_Rt_model_pts
            }
            ov_pem_sub2_model_compiled = self.ov_model_list[2]
            time_start = time.time()
            ov_pem_sub2_results = ov_pem_sub2_model_compiled(ov_pem_sub2_input)
            ov_pem_sub2_results = list(ov_pem_sub2_results.values())
            pem_sub2_time = time.time() - time_start
            # print(f"    [OpenVINO {device}] ov_pem_sub2 inference time: {pem_sub2_time*1000:.2f} ms")
            
            init_R = ov_pem_sub2_results[0]
            init_t = ov_pem_sub2_results[1]
        # ================[Pass]OpenVINO Sub2 model ==========================

        ov_pem_sub3_model_compiled = self.ov_model_list[3]
        ov_pem_sub3_input = {
                    "dense_pm": dense_pm, 
                    "dense_fm": dense_fm, 
                    "geo_embedding_m": geo_embedding_m, 
                    "fps_idx_m": fps_idx_m,
                    "dense_po_out": dense_po_out, 
                    "dense_fo_out": dense_fo_out, 
                    "geo_embedding_o": geo_embedding_o, 
                    "fps_idx_o": fps_idx_o,
                    "radius": radius, 
                    "model": model, 
                    "init_R": init_R, 
                    "init_t": init_t}
        time_start = time.time()
        ov_pem_sub3_results = ov_pem_sub3_model_compiled(ov_pem_sub3_input)
        ov_pem_sub3_results = list(ov_pem_sub3_results.values())
        pem_sub3_time = time.time() - time_start
        # print(f"    [OpenVINO {device}] ov_pem_sub3 inference time: {pem_sub3_time*1000:.2f} ms")

        fine_Rt_atten = ov_pem_sub3_results[0]
        fine_Rt_model_pts = ov_pem_sub3_results[1]
        # ==============[Pass]OpenVINO Sub3 model ============================
        torch_pem_sub4_input = (torch.from_numpy(fine_Rt_atten), torch.from_numpy(dense_pm), \
                                torch.from_numpy(dense_po_out), torch.from_numpy(fine_Rt_model_pts))
        torch_flag = False
        if torch_flag:
            pem_sub4_model = OVPEM_Sub4(self.pem_cfg.model)
            torch_device = torch.device("cpu")
            pem_sub4_model.to(torch_device)
            torch_pem_input_new = []
            for tmp_tensor in torch_pem_sub4_input:
                torch_pem_input_new.append(tmp_tensor.to(torch_device))
            time_start = time.time()
            with torch.no_grad():
                pred_R, pred_t, pred_pose_score = pem_sub4_model(*torch_pem_input_new)
                pred_R = pred_R.cpu().numpy()
                pred_t = pred_t.cpu().numpy()
                pred_pose_score = pred_pose_score.cpu().numpy()
            pem_sub4_time = time.time() - time_start
            # print(f"    [Pytorch cpu] pem_sub4 inference time: {pem_sub4_time*1000:.2f} ms")
        else:
            ov_pem_sub4_model_compiled = self.ov_model_list[4]
            ov_pem_sub4_input = {
                "fine_Rt_atten": fine_Rt_atten,
                "dense_pm": dense_pm,
                "dense_po_out": dense_po_out,
                "fine_Rt_model_pts": fine_Rt_model_pts
            }
            time_start = time.time()
            ov_pem_sub4_results = ov_pem_sub4_model_compiled(ov_pem_sub4_input)
            ov_pem_sub4_results = list(ov_pem_sub4_results.values())
            pem_sub4_time = time.time() - time_start
            # print(f"    [OpenVINO {device}] ov_pem_sub4 inference time: {pem_sub4_time*1000:.2f} ms")
            
            pred_R = ov_pem_sub4_results[0]
            pred_t = ov_pem_sub4_results[1]
            pred_pose_score = ov_pem_sub4_results[2]
        # ==============[Pass]OpenVINO Sub4 model ============================
        pred_t = pred_t * (radius.reshape(-1, 1)+1e-6)
        pem_model_time = time.time() - pem_time_start
        if not self.first_run:
            print(f"[OpenVINO {self.pem_cfg.device}] OpenVINO PEM model inference time: {pem_model_time*1000:.2f} ms")
            print(f"    ov_pem_sub1 inference time: {pem_sub1_time*1000:.2f} ms")
            print(f"    ov_pem_sub2 inference time: {pem_sub2_time*1000:.2f} ms")
            print(f"    ov_pem_sub3 inference time: {pem_sub3_time*1000:.2f} ms")
            print(f"    ov_pem_sub4 inference time: {pem_sub4_time*1000:.2f} ms")

        self.pose_scores = pred_pose_score * input_data['score'].detach().cpu().numpy()
        self.pred_rot = pred_R
        self.pred_trans = pred_t * 1000  # m to mm

        if self.visualize:
            for idx in range(ninstance):
                K = input_data['K'][idx:idx+1].detach().cpu().numpy()  # shape (1, 3, 3)
                vis_img = visualize_pem(
                    img,
                    self.pred_rot[idx:idx+1],
                    self.pred_trans[idx:idx+1],
                    model_points*1000,
                    K
                )
                print(f"    pred_rot[{idx}]: ", self.pred_rot[idx])
                print(f"    pred_trans[{idx}]: ", self.pred_trans[idx])
                print(f"    Pose score[{idx}]: ", self.pose_scores[idx])
                # Publish vis_img as a ROS2 Image message
                # Convert PIL Image to numpy array
                vis_img_np = np.array(vis_img)
                # Create ROS2 Image message
                ros_img_msg = RosImage()
                ros_img_msg.header.stamp = self.node.get_clock().now().to_msg()
                ros_img_msg.header.frame_id = "pose_estimation"
                ros_img_msg.height = vis_img_np.shape[0]
                ros_img_msg.width = vis_img_np.shape[1]
                ros_img_msg.encoding = "rgb8"
                ros_img_msg.is_bigendian = False
                ros_img_msg.step = vis_img_np.shape[1] * 3
                ros_img_msg.data = vis_img_np.tobytes()
                self.image_pub.publish(ros_img_msg)

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
