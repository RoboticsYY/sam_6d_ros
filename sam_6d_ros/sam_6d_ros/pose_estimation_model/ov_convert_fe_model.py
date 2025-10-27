import argparse
import time
import os
import sys
import random
import importlib

import torch
import torch.nn as nn
import openvino as ov
from openvino import Core

from run_inference_custom_openvino_gpu import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', '..', 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

def prepare_fe_data(cfg, device, npoint=None):
    tem_path = os.path.join(cfg.output_dir, 'templates')
    print("device", device, device.type)
    tem_rgb_list, tem_pts_list, tem_choose_list = get_templates(tem_path, cfg.test_dataset, device)
    torch_fe_input = (tem_rgb_list, tem_pts_list, tem_choose_list)

    rgb_input = torch.cat(tem_rgb_list, dim=1)            # (B, T*3, H, W)
    pts_input = torch.cat(tem_pts_list, dim=1)            # (B, T*N, 3)
    choose_input = torch.cat(tem_choose_list, dim=1)      # (B, T*N)
    
    # 使用 stack，保留 template 维度
    # rgb_input = torch.stack(tem_rgb_list, dim=1)     # [1, T, 3, H, W]
    # pts_input = torch.stack(tem_pts_list, dim=1)     # [1, T, N, 3]
    # choose_input = torch.stack(tem_choose_list, dim=1) # [1, T, N]

    # rgb_input = torch.stack([t for t in tem_rgb_list], dim=1)       # [1, T, 3, H, W]
    # pts_input = torch.stack([t for t in tem_pts_list], dim=1)       # [1, T, N, 3]
    # choose_input = torch.stack([t for t in tem_choose_list], dim=1) # [1, T, N]

    print("[OV debug] rgb_input shape:", rgb_input.shape)
    print("[OV debug] pts_input shape:", pts_input.shape)
    print("[OV debug] choose_input shape:", choose_input.shape)
    
    onnx_fe_input_name = ["rgb_input", "pts_input", "choose_input"]
    onnx_fe_input = (rgb_input, pts_input, choose_input)

    batch_size = 1
    ov_fe_input_name = {"rgb_input"   :[batch_size,126,224,224], 
                        "pts_input"   :[batch_size,210000,3], 
                        "choose_input":[batch_size,210000]}

    ov_fe_input = {
        "rgb_input": rgb_input,
        "pts_input": pts_input,
        "choose_input": choose_input,
    }

    return (torch_fe_input, 
            onnx_fe_input_name, onnx_fe_input,
            ov_fe_input_name, ov_fe_input)


class FeatureExtractionWrapper(nn.Module):
    """feature_extraction get_obj_feats wrapper for onnx export"""
    def __init__(self, feature_extraction):
        super().__init__()
        self.feature_extraction = feature_extraction
    
    def forward(self, rgb_input, pts_input, choose_input):
        """
        export get_obj_feats method
        input: concatenated tensors
        rgb_input: (B, n_template_view*3, H, W)
        pts_input: (B, n_template_view*n_sample_template_point, 3)
        choose_input: (B, n_template_view*n_sample_template_point)
        """
        # debug info
        print(f"FeatureExtractionWrapper输入形状: rgb_input={rgb_input.shape}, pts_input={pts_input.shape}, choose_input={choose_input.shape}")
        
        # split concatenated tensors
        n_template_view = 42  # from config
        n_sample_template_point = pts_input.size(1) // n_template_view
        
        print(f"split parameters: n_template_view={n_template_view}, n_sample_template_point={n_sample_template_point}")
        
        tem_rgb_batch = rgb_input.view(rgb_input.size(0), n_template_view, 3, *rgb_input.shape[2:])
        tem_pts_batch = pts_input.view(pts_input.size(0), n_template_view, n_sample_template_point, 3)
        tem_choose_batch = choose_input.view(choose_input.size(0), n_template_view, n_sample_template_point)

        # call original method
        return self.feature_extraction.get_obj_feats(tem_rgb_batch, tem_pts_batch, tem_choose_batch)

def onnx_model_convert_feature_extraction_submodel(model, onnx_fe_input_name, onnx_fe_input, onnx_model_path):
    feature_wrapper = FeatureExtractionWrapper(model.feature_extraction)
    try:
        with torch.no_grad():
            torch.onnx.export(
                feature_wrapper,
                onnx_fe_input,
                onnx_model_path,
                opset_version=20,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                input_names=onnx_fe_input_name,
                dynamic_axes={k: {0: "batch"} for k in onnx_fe_input_name},
                do_constant_folding=False,
                verbose=False,  # True , for detailed output
                export_params=True,
                keep_initializers_as_inputs=False
            )
        print(f"[ONNX] feature extraction submodel export success: {onnx_model_path}")
        
    except Exception as e:
        print(f"[ONNX] feature extraction submodel export failed: {e}")

def openvino_model_convert_feature_extraction_submodel(core, ov_fe_input_name, ov_fe_input, onnx_fe_model_path, ov_fe_model_path, ov_extension_lib_path):
    # ov_fe_model = core.read_model(onnx_fe_model_path)
    ov_fe_model = ov.convert_model(onnx_fe_model_path, 
                                    input=ov_fe_input_name,
                                    example_input=ov_fe_input,
                                    extension=ov_extension_lib_path,
                                    )
    compiled_model = core.compile_model(ov_fe_model, 'CPU')
    ov.save_model(ov_fe_model, ov_fe_model_path, compress_to_fp16=False)
    print(f"[OpenVINO] feature extraction submodel convert success: {ov_fe_model_path}")

def openvino_infer_feature_extraction_submodel(core, ov_fe_input, ov_fe_model_path, ov_gpu_kernel_path, device):
    ov_fe_model = core.read_model(ov_fe_model_path)

    if device == "GPU":
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        core.set_property("GPU", {"CONFIG_FILE": ov_gpu_kernel_path})

    if device == "HETERO:GPU,CPU":
        for op in ov_fe_model.get_ops():
            rt_info = op.get_rt_info()
            rt_info["FurthestPointSampling"] = "CPU"
            rt_info["GroupingOperation"] = "CPU"

    ov_fe_compiled_model = core.compile_model(ov_fe_model, device)

    time_start = time.time()
    ov_fe_results = ov_fe_compiled_model(ov_fe_input)
    ov_fe_results_list = list(ov_fe_results.values())
    fe_time = time.time() - time_start
    print(f"[OpenVINO {device}] fe (feature extraction) inference time: {fe_time*1000:.2f} ms")
    return ov_fe_results_list

def torch_infer_feature_extraction_submodel_list(model, input_data):
    all_tem = input_data[0]
    all_tem_pts = input_data[1]
    all_tem_choose = input_data[2]
    time_start = time.time()
    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)
    fe_time = time.time() - time_start
    print(f"[PyTorch] feature extraction inference time: {fe_time*1000:.2f} ms")
    return all_tem_pts, all_tem_feat

def torch_infer_feature_extraction_submodel_batched(model, input_data):
    all_tem = input_data[0]
    all_tem_pts = input_data[1]
    all_tem_choose = input_data[2]
    feature_wrapper = FeatureExtractionWrapper(model.feature_extraction)
    time_start = time.time()
    with torch.no_grad():
        all_tem_pts, all_tem_feat = feature_wrapper(all_tem, all_tem_pts, all_tem_choose)
    fe_time = time.time() - time_start
    print(f"[PyTorch] feature extraction inference time: {fe_time*1000:.2f} ms")
    torch_output_list = [all_tem_pts, all_tem_feat]
    return torch_output_list

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
    model = MODEL.Net(cfg.model)
    model = model.to(device)
    model.eval()
    checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'checkpoints', 'sam-6d-pem-base.pth')
    # load checkpoint
   # load checkpoint with map_location
    if gorilla_module is not None:
        gorilla_module.solver.load_checkpoint(model=model, filename=checkpoint, map_location=device)
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
            model_state = model.state_dict()
            # filter by matching keys and shapes
            filtered_state = {k: v for k, v in normalized_state.items() if k in model_state and v.shape == model_state[k].shape}
            missing_keys = [k for k in model_state.keys() if k not in normalized_state]
            unexpected_keys = [k for k in normalized_state.keys() if k not in model_state]
            print(f"Partial checkpoint load -> matched: {len(filtered_state)}/{len(model_state)}, missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")
            model_state.update(filtered_state)
            model.load_state_dict(model_state, strict=False)
            print("Checkpoint loaded with partial matching")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Continuing with uninitialized model weights")


    model_save_path = "model_save"
    os.makedirs(model_save_path, exist_ok=True)
    onnx_fe_model_path = os.path.join(model_save_path, 'onnx_fe_model.onnx')
    ov_fe_model_path = os.path.join(model_save_path, 'ov_fe_model.xml')
    ov_gpu_kernel_path = "./model/ov_pointnet2_op/ov_gpu_custom_op.xml"
    ov_extension_lib_path = './model/ov_pointnet2_op/build/libopenvino_operation_extension.so'

    core = Core()
    core.add_extension(ov_extension_lib_path)

    # data pre-process
    torch_fe_input, onnx_fe_input_name, onnx_fe_input, ov_fe_input_name, ov_fe_input = prepare_fe_data(cfg, device)

    # onnx model convert
    onnx_model_convert_feature_extraction_submodel(model, onnx_fe_input_name, onnx_fe_input, onnx_fe_model_path)

    # openvino model convert
    openvino_model_convert_feature_extraction_submodel(core, ov_fe_input_name, ov_fe_input, onnx_fe_model_path, ov_fe_model_path, ov_extension_lib_path)

    # torch model infer
    """
    [Draft] Using batch type output instead of list input can effectively improve the performance of FE model
    Currently, there are some differences in the output results of get_obj_feats batch and list.
    """
    torch_output = torch_infer_feature_extraction_submodel_list(model, torch_fe_input)
    # torch_output = torch_infer_feature_extraction_submodel_batched(model, onnx_fe_input)

    # openvino model cpu infer
    ov_device = "CPU"
    DEBUG_FLAG = False # True / False
    ov_output_cpu = openvino_infer_feature_extraction_submodel(core, ov_fe_input, ov_fe_model_path, ov_gpu_kernel_path, ov_device)
    if DEBUG_FLAG:
        for i in range(len(ov_output_cpu)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_cpu[i], ov_device)

    # openvino model gpu infer
    ov_device = "GPU"
    DEBUG_FLAG = True # True / False
    ov_output_gpu = openvino_infer_feature_extraction_submodel(core, ov_fe_input, ov_fe_model_path, ov_gpu_kernel_path, ov_device)
    # ov_output_gpu = openvino_infer_feature_extraction_submodel(core, ov_fe_input, ov_fe_model_path, ov_gpu_kernel_path, ov_device)
    if DEBUG_FLAG:
        for i in range(len(ov_output_gpu)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_gpu[i], ov_device)

    # openvino model HETERO:GPU,CPU infer
    ov_device = "HETERO:GPU,CPU"
    DEBUG_FLAG = False # True / False
    ov_output_hetero = openvino_infer_feature_extraction_submodel(core, ov_fe_input, ov_fe_model_path, ov_gpu_kernel_path, "HETERO:GPU,CPU")
    if DEBUG_FLAG:
        for i in range(len(ov_output_hetero)):
            print(f"=====================[{ov_device} Result Compare :The {i}th output]======================")
            compare_result(torch_output[i], ov_output_hetero[i], ov_device)

if __name__ == "__main__":
    main()
    """
    This submodel includes: 
        FeatureExtraction model (feature_extraction.get_obj_feats).
    Currently, there are CPU infer pass, GPU infer pass, and Hetero infer pass.

    Using batch type output instead of list input can effectively improve the performance of FE model
    Currently, there are some differences in the output results of get_obj_feats batch and list.
    Need to debug the difference between 'get_obj_feats batch and list' structure.

    Consider using OV GPU for actual deployment.
    """