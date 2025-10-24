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
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

import time
import logging

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

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
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    print(f"[OpenVINO] OV ISM pipeline inference start...")
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    start_model_init = time.perf_counter()
    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    logging.info("Initializing template")
    template_dir = os.path.join(output_dir, 'templates')
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

    model.ref_data = {}
    start_compute_features = time.perf_counter()
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    end_compute_features = time.perf_counter()
    print(f"    [Timing] compute_features time: {(end_compute_features - start_compute_features)*1000:.2f} ms")

    start_compute_masked_patch_feature = time.perf_counter()
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    end_compute_masked_patch_feature = time.perf_counter()
    print(f"    [Timing] compute_masked_patch_feature time: {(end_compute_masked_patch_feature - start_compute_masked_patch_feature)*1000:.2f} ms")
    logging.info(f"** compute_features input: {templates.size()}")
    logging.info(f"** compute_features output: {model.ref_data['descriptors'].size()}")
    logging.info(f"** compute_masked_patch_feature input: {templates.size()}, {masks_cropped[:, 0, :, :].size()}")
    logging.info(f"** compute_masked_patch_feature output: {model.ref_data['appe_descriptors'].size()}")

    # run inference
    ### Export to onnx
    export_model = model.descriptor_model.model.cpu()
    dummy_input = torch.rand_like(templates).cpu()
    dummy_flag = False
    start_infer = time.perf_counter()
    # torch.onnx.export(export_model, dummy_input, "descriptor_dinov2_model.onnx", )
    # torch.onnx.export(
    #     export_model,
    #     (dummy_input),
    #     "descriptor_dinov2_model_infer.onnx",
    #     export_params=True,
    #     opset_version=17,
    #     do_constant_folding=False,  # Keep both branches
    #     input_names=["images"],
    #     dynamic_axes={"input": {0: "batch"}}
    # )
    rgb = Image.open(rgb_path).convert("RGB")
    ### warm up
    detections_w = model.segmentor_model.generate_masks(np.array(rgb))
    logging.info(detections_w['masks'].size())
    logging.info(detections_w['boxes'].size())

    detections_w['masks'] = detections_w['masks'][0:2, :, :]
    detections_w['boxes'] = detections_w['boxes'][0:2, :]

    logging.info(detections_w['masks'].size())
    logging.info(detections_w['boxes'].size())
    
    detections_w = Detections(detections_w)

    logging.info(detections_w.masks.size())
    logging.info(detections_w.boxes.size())

    ov_infer_start = time.time()
    start_gen_masks = time.perf_counter()
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    end_gen_masks = time.perf_counter()
    print(f"    [Timing] generate_masks time: {(end_gen_masks - start_gen_masks)*1000:.2f} ms")
    # print(f"detections: {detections}")
    logging.info(f"** generate_masks input: {np.array(rgb).shape}") 
    logging.info(f"** generate_masks output dict: mask {detections['masks'].size()}, boxes {detections['boxes'].size()}")
    
    #print(rgb.size)
    # # export model: model.segmentor_model.model.predictor.model
    # # export input: images [n, 3, h, w] --> /home/intel/miniforge3/envs/sam6d/lib/python3.9/site-packages/ultralytics/yolo/engine/predictor.py line 124
    # # export input shape [n, 3, h, w], n=1, np.array(rgb) [h, w, 3]: h=np.array(rgb).shape[0], w=np.array(rgb).shape[1]
    # temp_rgb = np.array(rgb) # obtain h, w
    # height, width = temp_rgb.shape[0], temp_rgb.shape[1]
    # export_input = torch.rand(1, 3, height, width)
    # export_model = model.segmentor_model.model.predictor.model.cpu()
    # torch.onnx.export(export_model, export_input, "fastsam_yolo_v8_predictor.onnx")
    detections = Detections(detections)
    #print(np.array(rgb).shape)

    start_forward = time.perf_counter()
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)
    end_forward = time.perf_counter()
    print(f"    [Timing] descriptor_model.forward time: {(end_forward - start_forward)*1000:.2f} ms")

    start_sem = time.perf_counter()
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)
    end_sem = time.perf_counter()
    print(f"    [Timing] compute_semantic_score time: {(end_sem - start_sem)*1000:.2f} ms")

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    start_appe = time.perf_counter()
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)
    end_appe = time.perf_counter()
    print(f"    [Timing] compute_appearance_score time: {(end_appe - start_appe)*1000:.2f} ms")
    # compute the geometric score
    
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    start_project = time.perf_counter()
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
    end_project = time.perf_counter()
    print(f"    [Timing] compute_project_template_to_image time: {(end_project - start_project)*1000:.2f} ms")

    start_geo = time.perf_counter()
    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )
    end_geo = time.perf_counter()
    print(f"    [Timing] compute_geometric_score time: {(end_geo - start_geo)*1000:.2f} ms")
    
    total_stage_time = (
        (end_gen_masks - start_gen_masks)
        + (end_forward - start_forward)
        + (end_sem - start_sem)
        + (end_appe - start_appe)
        + (end_geo - start_geo)
    )
    ov_infer_end = time.time()
    print(f"[Timing] Sum of 5 core stages: {total_stage_time*1000:.2f} ms")
    # final score
    start_final = time.perf_counter()
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
    end_final = time.perf_counter()
    print(f"    [Timing] compute final_score time: {(end_final - start_final)*1000:.2f} ms")

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, f"{output_dir}/sam6d_results/vis_ism.png")
    vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")
    print(f"[OpenVINO] OpenVINO Instance_Segmentation_Model pipeline E2E Inference Time: {(ov_infer_end - ov_infer_start)*1000:.2f} ms, save_path : {output_dir}/sam6d_results/vis_ism.png")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )