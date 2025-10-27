import argparse
import os
import sys
import numpy as np
import random
import importlib

import torch
import torch.nn as nn

import openvino as ov
from openvino import Core

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
ROOT_DIR = os.path.join(BASE_DIR,)
print("[ROOT_DIR] : ",ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

from utils.model_utils import sample_pts_feats
# from run_inference_custom_pytorch import *
from run_inference_custom_openvino_gpu import *

DEBUG_FLAG = True

class OVPEM_Sub1(nn.Module):
    def __init__(self, cfg, model, npoint=2048):
        super(OVPEM_Sub1, self).__init__()
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint
        self.model = model

    def forward(self, pts,  rgb, rgb_choose, 
                    model, dense_po, dense_fo):
        # [pass] 1. get dense features
        dense_pm, dense_fm, dense_po_out, dense_fo_out, radius = self.model.feature_extraction(pts, rgb, rgb_choose, dense_po, dense_fo)

        # [pass] 2. sample sparse features
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100

        # 1th-sparse_pm, 13th-sparse_fm, 7th-fps_idx_m
        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(dense_pm, dense_fm, self.coarse_npoint, return_index=True)

        # self.geo_embedding in ov model with error result -> -nan
        # 6th-geo_embedding_m
        geo_embedding_m = self.model.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(dense_po_out, dense_fo_out, self.coarse_npoint, return_index=True)
        
        # 10th-geo_embedding_o
        geo_embedding_o = self.model.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        # 0th-coarse_Rt_atten
        coarse_Rt_atten, coarse_Rt_model_pts = self.model.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, model
        )
        np.save("output/radius.npy", radius)
        np.save("output/model.npy", model)
        print(f"[Torch Debug] radius :", radius.shape, radius.type(), radius.dtype)
        print(f"[Torch Debug] model :", model.shape, model.type(), model.dtype)


        return coarse_Rt_atten, sparse_pm, sparse_po, coarse_Rt_model_pts, \
               dense_pm, dense_fm, geo_embedding_m, fps_idx_m, \
               dense_po_out, dense_fo_out, geo_embedding_o, fps_idx_o, radius

def prepare_pem_data(cfg, torch_model, device, npoint=None):
    tem_path = os.path.join(cfg.output_dir, 'templates')
    print("device", device, device.type)
    all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, cfg.test_dataset, device)
    with torch.no_grad():
        all_tem_pts, all_tem_feat = torch_model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    input_data, img, whole_pts, model_points, detections = get_test_data(
        cfg.rgb_path, cfg.depth_path, cfg.cam_path, cfg.cad_path, cfg.seg_path, 
        cfg.det_score_thresh, cfg.test_dataset, device
    )
    # print(f"[Pytorch] input_data.keys: {input_data.keys()}")
    ninstance = input_data['pts'].size(0)

    #pytorch infer verification
    with torch.no_grad():
        input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
        input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)
    
    print("prepare data for model convert...")
    torch_pem_input =  (input_data['pts'], 
                        input_data['rgb'], input_data['rgb_choose'], 
                      input_data['model'], 
                      input_data['dense_po'], input_data['dense_fo'])

    onnx_pem_input_name = ["pts", 
                           "rgb", 
                           "rgb_choose", 
                           "model", 
                           "dense_po", 
                           "dense_fo"]

    onnx_pem_output_name = ["coarse_Rt_atten", "sparse_pm", "sparse_po", "coarse_Rt_model_pts", \
               "dense_pm", "dense_fm", "geo_embedding_m", "fps_idx_m", \
               "dense_po_out", "dense_fo_out", "geo_embedding_o", "fps_idx_o", "radius"]

    onnx_pem_input = (
        input_data['pts'], 
        input_data['rgb'], 
        input_data['rgb_choose'], 
        input_data['model'], 
        input_data['dense_po'], 
        input_data['dense_fo'],
        )
    

    batch_size = -1
    ov_pem_input_name = {"pts":[batch_size,2048,3], 
                        "rgb":[batch_size,3,224,224], 
                        "rgb_choose":[batch_size,2048], 
                        "model":[batch_size,1024,3], 
                        "dense_po":[batch_size,2048,3], 
                        "dense_fo":[batch_size,2048,256],
                        }

    ov_pem_output_name = ["coarse_Rt_atten", "sparse_pm", "sparse_po", "coarse_Rt_model_pts", \
               "dense_pm", "dense_fm", "geo_embedding_m", "fps_idx_m", \
               "dense_po_out", "dense_fo_out", "geo_embedding_o", "fps_idx_o", "radius"]

    ov_pem_input = {
        "pts": input_data['pts'],
        "rgb": input_data['rgb'],
        "rgb_choose": input_data['rgb_choose'],
        "model": input_data['model'],
        "dense_po": input_data['dense_po'],
        "dense_fo": input_data['dense_fo'],
    }

    return (torch_pem_input, 
            onnx_pem_input_name, 
            onnx_pem_input,
            ov_pem_input_name, 
            ov_pem_input)

def onnx_model_convert_pose_estimation_submodel(model, onnx_pem_input_name, onnx_pem_input, onnx_model_path):
    try:
        torch.onnx.export(
            model,
            onnx_pem_input,
            onnx_model_path,
            opset_version=20,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=onnx_pem_input_name,
            # output_names=onnx_pem_output_name,
            dynamic_axes={k: {0: "batch"} for k in onnx_pem_input_name},
            do_constant_folding=False,
            verbose=False,
            export_params=True,
            keep_initializers_as_inputs=False
        )
        print(f"[ONNX] PEM onnx model export success: {onnx_model_path}")
            
    except Exception as e:
        print(f"[ONNX] export failed : {e}")

def openvino_model_convert_pose_estimation_submodel(core, ov_pem_input_name, ov_pem_input, onnx_pem_model_path, ov_pem_model_path, ov_extension_lib_path):
    
    ov_pem_model = ov.convert_model(onnx_pem_model_path, 
                                    input=ov_pem_input_name,
                                    # output=ov_pem_output_name,
                                    example_input=ov_pem_input,
                                    extension=ov_extension_lib_path,
                                    )
    compiled_model = core.compile_model(ov_pem_model, 'CPU')
    ov.save_model(ov_pem_model, ov_pem_model_path, compress_to_fp16=False)
    print(f"[OpenVINO] pose estimation submodel convert success: {ov_pem_model_path}")

def openvino_infer_pose_estimation_submodel(core, ov_fe_input, ov_fe_model_path, ov_gpu_kernel_path, device):
    ov_fe_model = core.read_model(ov_fe_model_path)

    if device == "GPU":
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        core.set_property("GPU", {"CONFIG_FILE": ov_gpu_kernel_path})

    if device == "HETERO:GPU,CPU":
        for op in ov_fe_model.get_ops():
            rt_info = op.get_rt_info()
            rt_info["CustomSVDu"] = "CPU"
            rt_info["CustomSVDv"] = "CPU"

    ov_pem_compiled_model = core.compile_model(ov_fe_model, device)

    time_start = time.time()
    ov_pem_results = ov_pem_compiled_model(ov_fe_input)
    ov_pem_results_list = list(ov_pem_results.values())
    pem_time = time.time() - time_start
    print(f"[OpenVINO {device}] pem (feature extraction) inference time: {pem_time*1000:.2f} ms")
    return ov_pem_results_list

def torch_infer_pose_estimation_submodel(model, input_data, save_flag=True):
    time_start = time.time()
    with torch.no_grad():
        coarse_Rt_atten, sparse_pm, sparse_po, coarse_Rt_model_pts, \
               dense_pm, dense_fm, geo_embedding_m, fps_idx_m, \
               dense_po_out, dense_fo_out, geo_embedding_o, fps_idx_o, radius = model(*input_data)
    fe_time = time.time() - time_start
    print(f"[PyTorch] feature extraction inference time: {fe_time*1000:.2f} ms")
    torch_output_list = [coarse_Rt_atten, sparse_pm, sparse_po, coarse_Rt_model_pts, \
               dense_pm, dense_fm, geo_embedding_m, fps_idx_m, \
               dense_po_out, dense_fo_out, geo_embedding_o, fps_idx_o, radius]
    if save_flag:
        torch_save_result(torch_output_list)
    return torch_output_list

def torch_save_result(torch_output_list):
    print("[Torch Debug] torch_save_result ===============")
    np.save("output/coarse_Rt_atten.npy", torch_output_list[0])
    np.save("output/sparse_pm.npy", torch_output_list[1])
    np.save("output/sparse_po.npy", torch_output_list[2])
    np.save("output/coarse_Rt_model_pts.npy", torch_output_list[3])
    np.save("output/dense_pm.npy", torch_output_list[4])
    np.save("output/dense_fm.npy", torch_output_list[5])
    np.save("output/geo_embedding_m.npy", torch_output_list[6])
    np.save("output/fps_idx_m.npy", torch_output_list[7])
    np.save("output/dense_po_out.npy", torch_output_list[8])
    np.save("output/dense_fo_out.npy", torch_output_list[9])
    np.save("output/geo_embedding_o.npy", torch_output_list[10])
    np.save("output/fps_idx_o.npy", torch_output_list[11])
    np.save("output/radius.npy", torch_output_list[12])
    for i in range(len(torch_output_list)):
        print(f"[Torch Debug] {i}th output :", torch_output_list[i].shape, torch_output_list[i].type(), torch_output_list[i].dtype)
    
    print("[Torch Debug] torch_save_result Done ===============")

def compare_result(torch_out, ov_out, device, atol=1e-4, return_indices=True, top_k=10):
    # Compare
    print("Torch output shape:", torch_out.shape)
    ov_tensor = torch.from_numpy(ov_out)
    print("OV output shape:", ov_tensor.shape)

    ov_tensor = ov_tensor.type(torch.float32)
    torch_out = torch_out.type(torch.float32)

    diff = torch.abs(ov_tensor - torch_out)
    max_diff = diff.max()
    min_diff = diff.min()
    mse = (diff ** 2).mean()
    
    print(f"[OV {device}] + Pytorch  Max diff: {max_diff}")
    print(f"[OV {device}] + Pytorch  Min diff: {min_diff}")
    print(f"[OV {device}] + Pytorch  MSE: {mse}")
    
    # 初始化返回字典
    compare_result = {
        'max_diff': max_diff.item() if torch.is_tensor(max_diff) else max_diff,
        'min_diff': min_diff.item() if torch.is_tensor(min_diff) else min_diff,
        'mse': mse.item() if torch.is_tensor(mse) else mse,
        'passed': torch.allclose(ov_tensor, torch_out, atol=atol)
    }
    
    if return_indices:
        significant_diff_mask = diff > atol
        print("[significant_diff_mask] shape:", significant_diff_mask.shape)
        
        if significant_diff_mask.any():
            # 直接获取多维索引: shape [N, 3]
            multi_indices = torch.nonzero(significant_diff_mask, as_tuple=False)
            print("[multi_indices] shape:", multi_indices.shape)  # 应该是 [394274, 3]
            
            # 获取这些位置的差异值
            diff_values = diff[significant_diff_mask]  # shape [394274]
            print("[diff_values] shape:", diff_values.shape)
            
            # 按差异值排序（从大到小）
            if top_k is not None:
                k = min(top_k, len(diff_values))
                sorted_indices_in_flat = torch.argsort(diff_values, descending=True)[:k]
            else:
                sorted_indices_in_flat = torch.argsort(diff_values, descending=True)
            
            # 取出排序后的索引和值
            top_multi_indices = multi_indices[sorted_indices_in_flat]  # shape [k, 3]
            top_diff_values = diff_values[sorted_indices_in_flat]
            
            # 转换为列表用于输出
            indices_list = top_multi_indices.tolist()
            diff_list = top_diff_values.detach().cpu().numpy().tolist()
            
            compare_result['significant_diff_values'] = diff_list
            
            print(f"\n[DIFF INDICES] Found {significant_diff_mask.sum().item()} elements with |diff| > {atol}")
            print(f"[DIFF INDICES] Top {len(indices_list)} largest differences:")
            
            with open('output/fps_node_debug_node.txt', 'w') as f:
                for i, (idx, val) in enumerate(zip(indices_list, diff_list)):
                    # idx 是类似 [0, point_idx, channel] 的 list
                    torch_val = torch_out[tuple(idx)].item()
                    ov_val = ov_tensor[tuple(idx)].item()
                    print(f"  Rank {i+1}: Index {idx}, Diff = {val:.5f}, torch_val={torch_val:.5f}, ov_val={ov_val:.5f}")
                    f.write(f"{idx}\n")

    import traceback

    try:
        torch.testing.assert_close(ov_tensor, torch_out, atol=1e-4, rtol=1e-5, check_dtype=False)
        print(f"✅ [COMPARE Result {device}] and Pytorch PASSED")
    except Exception as e:
        print(f"❌ [COMPARE Result {device}], Failed")
        print(f"Error: {e}")
        traceback.print_exc() 
    return compare_result


def main():
    cfg = init()

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # device setting
    device = torch.device(cfg.device.lower())
    print(f"Using device: {device}")

    # model loading
    print("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    torch_model = MODEL.Net(cfg.model)
    torch_model = torch_model.to(device)
    torch_model.eval()
    checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'checkpoints', 'sam-6d-pem-base.pth')
    # load checkpoint
   # load checkpoint with map_location
    if gorilla_module is not None:
        gorilla_module.solver.load_checkpoint(model=torch_model, filename=checkpoint, map_location=device)
    else:
        # Fallback: load checkpoint manually using PyTorch (partial/non-strict)
        print("Loading checkpoint using PyTorch fallback method...")
        try:
            checkpoint_data = torch.load(checkpoint, map_location=device)
            state = checkpoint_data.get('state_dict', checkpoint_data)
            if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
                state = state['model']
            # strip common prefixes
            def _strip_prefix(k, prefix):
                return k[len(prefix):] if k.startswith(prefix) else k
            normalized_state = {}
            for k, v in state.items():
                nk = k
                for prefix in ('module.', 'model.', 'net.'):
                    nk = _strip_prefix(nk, prefix)
                normalized_state[nk] = v
            model_state = torch_model.state_dict()
            # filter by matching keys and shapes
            filtered_state = {k: v for k, v in normalized_state.items() if k in model_state and v.shape == model_state[k].shape}
            missing_keys = [k for k in model_state.keys() if k not in normalized_state]
            unexpected_keys = [k for k in normalized_state.keys() if k not in model_state]
            print(f"Partial checkpoint load -> matched: {len(filtered_state)}/{len(model_state)}, missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")
            model_state.update(filtered_state)
            torch_model.load_state_dict(model_state, strict=False)
            print("Checkpoint loaded with partial matching")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Continuing with uninitialized model weights")

    model_save_path = "model_save"
    os.makedirs(model_save_path, exist_ok=True)
    onnx_pem_model_path = os.path.join(model_save_path, 'onnx_pem_sub1_model.onnx')
    ov_pem_model_path = os.path.join(model_save_path, 'ov_pem_sub1_model.xml')
    ov_gpu_kernel_path = "./model/ov_pointnet2_op/ov_gpu_custom_op.xml"
    ov_extension_lib_path = './model/ov_pointnet2_op/build/libopenvino_operation_extension.so'

    core = Core()
    core.add_extension(ov_extension_lib_path)

    # data pre-process
    torch_pem_input, \
        onnx_pem_input_name, onnx_pem_input, \
        ov_pem_input_name,  ov_pem_input = prepare_pem_data(cfg, torch_model, device)

    pem_sub_model_1 = OVPEM_Sub1(cfg.model, model=torch_model)

    # torch model infer
    torch_output = torch_infer_pose_estimation_submodel(pem_sub_model_1, torch_pem_input)

    # onnx model convert
    onnx_model_convert_pose_estimation_submodel(pem_sub_model_1, onnx_pem_input_name, onnx_pem_input, onnx_pem_model_path)

    # openvino model convert
    openvino_model_convert_pose_estimation_submodel(core, ov_pem_input_name, ov_pem_input, onnx_pem_model_path, ov_pem_model_path, ov_extension_lib_path)

    # openvino model cpu infer
    ov_device = "CPU"
    DEBUG_FLAG = False # True / False
    ov_output_cpu = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    if DEBUG_FLAG:
        for i in range(len(ov_output_cpu)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_cpu[i], ov_device)

    # # openvino model gpu infer
    ov_device = "GPU"
    DEBUG_FLAG = True # True / False
    ov_output_gpu = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    ov_output_gpu = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    if DEBUG_FLAG:
        for i in range(len(ov_output_gpu)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_gpu[i], ov_device)

    # openvino model HETERO:GPU,CPU infer
    ov_device = "HETERO:GPU,CPU"
    DEBUG_FLAG = False # True / False
    ov_output_hetero = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    if DEBUG_FLAG:
        for i in range(len(ov_output_hetero)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_hetero[i], ov_device)

if __name__ == "__main__":
    main()
    """
    This submodel includes: 
        ViTEncoder -> GeometricStructureEmbedding -> CoarsePointMatching (without compute_coarse_Rt).
    Currently, there are CPU infer pass, GPU infer pass, and Hetero infer pass.
    However, the output differs slightly from Torch, so debugging the output is necessary.

    Consider using OV GPU for actual deployment.
    """