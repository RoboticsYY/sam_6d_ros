import argparse
import os
import sys
from PIL import Image
import os.path as osp
import numpy as np
import random
import importlib
import json
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms

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

from utils.model_utils import compute_fine_Rt
# from run_inference_custom_pytorch import *
from run_inference_custom_openvino_gpu import *

class OVPEM_Sub4(nn.Module):
    def __init__(self, cfg, npoint=2048):
        super(OVPEM_Sub4, self).__init__()
        self.cfg = cfg

    def forward(self, fine_Rt_atten, dense_pm, dense_po_out, fine_Rt_model_pts):
        pred_R, pred_t, pred_pose_score = compute_fine_Rt(fine_Rt_atten, dense_pm, 
                                        dense_po_out, fine_Rt_model_pts)

        return pred_R, pred_t, pred_pose_score

def prepare_pem_data(cfg, torch_model, device, npoint=None):
    fine_Rt_atten_file_path = "output/fine_Rt_atten.npy"
    dense_pm_file_path = "output/dense_pm.npy"
    dense_po_out_file_path = "output/dense_po_out.npy"
    fine_Rt_model_pts_file_path= "output/fine_Rt_model_pts.npy"
    flag_real_input = False
    if flag_real_input:
        fine_Rt_atten = torch.from_numpy(np.load(fine_Rt_atten_file_path))
        dense_pm = torch.from_numpy(np.load(dense_pm_file_path))
        dense_po_out = torch.from_numpy(np.load(dense_po_out_file_path)) 
        fine_Rt_model_pts = torch.from_numpy(np.load(fine_Rt_model_pts_file_path)) 
    else:
        fine_Rt_atten = torch.randn(7, 2049, 2049, dtype=torch.float32)
        dense_pm = torch.randn(7, 2048, 3, dtype=torch.float32) 
        dense_po_out = torch.randn(7, 2048, 3, dtype=torch.float32)
        fine_Rt_model_pts = torch.randn(7, 1024, 3, dtype=torch.float32) 
    
    print("prepare data for model convert...")
    torch_pem_input =  (fine_Rt_atten, dense_pm, dense_po_out, fine_Rt_model_pts)

    onnx_pem_input_name = ["fine_Rt_atten", "dense_pm", "dense_po_out", "fine_Rt_model_pts"]

    onnx_pem_input = (fine_Rt_atten, dense_pm, dense_po_out, fine_Rt_model_pts)

    batch_size = -1
    ov_pem_input_name = {"fine_Rt_atten":[batch_size, 2049, 2049], 
                        "dense_pm":[batch_size, 2048, 3], 
                        "dense_po_out":[batch_size, 2048, 3], 
                        "fine_Rt_model_pts":[batch_size, 1024, 3]
                        }
    ov_pem_input = {
        "fine_Rt_atten": fine_Rt_atten,
        "dense_pm": dense_pm,
        "dense_po_out": dense_po_out,
        "fine_Rt_model_pts": fine_Rt_model_pts
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
            rt_info["CustomDet"] = "CPU"

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
        pred_R, pred_t, pred_pose_score = model(*input_data)
    fe_time = time.time() - time_start
    print(f"[PyTorch] feature extraction inference time: {fe_time*1000:.2f} ms")
    torch_output_list = [pred_R, pred_t, pred_pose_score]
    if save_flag:
        torch_save_result(torch_output_list)
    return torch_output_list

def torch_save_result(torch_output_list):
    print("[Torch Debug] torch_save_result ===============")
    np.save("output/pred_R.npy", torch_output_list[0])
    np.save("output/pred_t.npy", torch_output_list[1])
    np.save("output/pred_pose_score.npy", torch_output_list[2])
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
    onnx_pem_model_path = os.path.join(model_save_path, 'onnx_pem_sub4_model.onnx')
    ov_pem_model_path = os.path.join(model_save_path, 'ov_pem_sub4_model.xml')
    ov_gpu_kernel_path = "./model/ov_pointnet2_op/ov_gpu_custom_op.xml"
    ov_extension_lib_path = './model/ov_pointnet2_op/build/libopenvino_operation_extension.so'

    core = Core()
    core.add_extension(ov_extension_lib_path)

    # data pre-process
    torch_pem_input, \
        onnx_pem_input_name, onnx_pem_input, \
        ov_pem_input_name, ov_pem_input = prepare_pem_data(cfg, torch_model, device)

    pem_sub4_model = OVPEM_Sub4(cfg.model)

    # torch model infer
    torch_output = torch_infer_pose_estimation_submodel(pem_sub4_model, torch_pem_input)

    # onnx model convert
    onnx_model_convert_pose_estimation_submodel(pem_sub4_model, onnx_pem_input_name, onnx_pem_input, onnx_pem_model_path)

    # openvino model convert
    openvino_model_convert_pose_estimation_submodel(core, ov_pem_input_name, ov_pem_input, onnx_pem_model_path, ov_pem_model_path, ov_extension_lib_path)
   
    # openvino model cpu infer
    ov_device = "CPU"
    DEBUG_FLAG = True # True / False
    ov_output_cpu = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    if DEBUG_FLAG:
        for i in range(len(ov_output_cpu)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_cpu[i], ov_device)


    # openvino model gpu infer
    ov_device = "GPU"
    DEBUG_FLAG = False # True / False
    '''
    [GPU infer failed] Exceeded max size of memory object allocation: requested 77446656000 bytes, but max alloc size supported by device is 24385683456 bytes.Please try to reduce batch size or use lower precision.
    Compute_coarse_Rt contains a large number of GPU-unfriendly ops, causing OOM errors. This part of the model structure should be restructured.
    The specific reason is unclear, but the CPU performance is sufficient, so there is no need to consider GPU deployment.
    '''
    # ov_output_gpu = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    # ov_output_gpu = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    # if DEBUG_FLAG:
    #     for i in range(len(ov_output_gpu)):
    #         print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
    #         compare_result(torch_output[i], ov_output_gpu[i], ov_device)


    # openvino model HETERO:GPU,CPU infer
    ov_device = "HETERO:GPU,CPU"
    DEBUG_FLAG = False # True / False
    '''
    [HETERO infer failed] 
    [GPU] CL_OUT_OF_RESOURCES exception.
    [GPU] clFinish, error code: -5 CL_OUT_OF_RESOURCES
    The specific reason is unclear, but the CPU performance is sufficient, so there is no need to consider Hetero deployment.
    '''
    # ov_output_hetero = openvino_infer_pose_estimation_submodel(core, ov_pem_input, ov_pem_model_path, ov_gpu_kernel_path, ov_device)
    # if DEBUG_FLAG:
    #     for i in range(len(ov_output_hetero)):
    #         print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
    #         compare_result(torch_output[i], ov_output_hetero[i], ov_device)

if __name__ == "__main__":
    main()
    """
    This submodel includes: 
        CoarsePointMatching.compute_coarse_Rt.
    Result compare(with pytorch), 
      - CPU infer pass,  accuracy pass
      - GPU infer failed, 
      - Hetero infer failed.

    Consider using OV CPU for actual deployment.
    """