import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import openvino as ov
import torch
from scipy.spatial.transform import Rotation


def request_device_time_ms(request):
    return sum(
        item.real_time.total_seconds() * 1000
        for item in request.profiling_info
        if item.status.name == "EXECUTED" and "wait_for_events" not in item.exec_type
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam6d-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cad-path", required=True)
    parser.add_argument("--templates-dir", required=True)
    parser.add_argument("--camera-path", required=True)
    parser.add_argument("--detector-path", required=True)
    parser.add_argument("--rgb-path", required=True)
    parser.add_argument("--depth-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--detector-threshold", type=float, default=0.2)
    parser.add_argument("--pipeline", choices=("yolo-sam", "ism"), required=True)
    parser.add_argument(
        "--sam2-variant",
        choices=(
            "sam2_hiera_tiny",
            "sam2_hiera_small",
            "sam2_hiera_base_plus",
            "sam2_hiera_large",
        ),
        default="sam2_hiera_small",
    )
    return parser.parse_args()


def nms(boxes, scores, threshold=0.2):
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        index = order[0]
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        intersection = (
            np.maximum(0.0, np.minimum(boxes[index, 2], boxes[rest, 2]) -
                       np.maximum(boxes[index, 0], boxes[rest, 0])) *
            np.maximum(0.0, np.minimum(boxes[index, 3], boxes[rest, 3]) -
                       np.maximum(boxes[index, 1], boxes[rest, 1])))
        union = areas[index] + areas[rest] - intersection
        order = rest[intersection / np.maximum(union, 1e-6) <= threshold]
    return keep


class OVSam2ImagePredictor:
    def __init__(self, encoder, decoder, transform):
        self.encoder_request = encoder.create_infer_request()
        self.decoder_request = decoder.create_infer_request()
        self.transform = transform
        self.image_embeddings = None
        self.high_res_256 = None
        self.high_res_128 = None
        self.original_size = None

    def set_image(self, image):
        resized = self.transform.apply_image(image)
        tensor = (resized.astype(np.float32) - np.asarray(
            [123.675, 116.28, 103.53], dtype=np.float32)) / np.asarray(
                [58.395, 57.12, 57.375], dtype=np.float32)
        tensor = tensor.transpose(2, 0, 1)[None]
        height, width = tensor.shape[-2:]
        tensor = np.pad(
            tensor,
            ((0, 0), (0, 0), (0, 1024 - height), (0, 1024 - width)),
        )
        start = time.perf_counter()
        self.encoder_request.infer({"image": tensor})
        host_ms = (time.perf_counter() - start) * 1000
        device_ms = request_device_time_ms(self.encoder_request)
        self.image_embeddings = self.encoder_request.get_output_tensor(0).data.copy()
        self.high_res_256 = self.encoder_request.get_output_tensor(1).data.copy()
        self.high_res_128 = self.encoder_request.get_output_tensor(2).data.copy()
        self.original_size = image.shape[:2]
        return host_ms, device_ms

    def predict_box(self, box):
        transformed_box = self.transform.apply_boxes(
            box.reshape(1, 4), self.original_size).reshape(1, 2, 2).astype(np.float32)
        labels = np.asarray([[2, 3]], dtype=np.float32)
        inputs = {
            "image_embeddings": self.image_embeddings,
            "point_coords": transformed_box,
            "point_labels": labels,
            "high_res_feats_256": self.high_res_256,
            "high_res_feats_128": self.high_res_128,
        }
        start = time.perf_counter()
        self.decoder_request.infer(inputs)
        host_ms = (time.perf_counter() - start) * 1000
        device_ms = request_device_time_ms(self.decoder_request)
        masks = self.decoder_request.get_output_tensor(0).data.copy()
        scores = self.decoder_request.get_output_tensor(1).data.copy()
        resized_height, resized_width = self.transform.get_preprocess_shape(
            self.original_size[0], self.original_size[1], 1024)
        masks = masks[..., :resized_height, :resized_width]
        masks = torch.nn.functional.interpolate(
            torch.from_numpy(masks),
            size=self.original_size,
            mode="bilinear",
            align_corners=False,
        ).numpy()
        index = int(np.argmax(scores[0]))
        return masks[0, index] > 0, float(scores[0, index]), host_ms, device_ms


def compile_sam2(core, args):
    ism_root = os.path.join(args.sam6d_root, "Instance_Segmentation_Model")
    if ism_root not in sys.path:
        sys.path.insert(0, ism_root)
    from infer_ism_ov import ResizeLongestSide

    model_dir = os.path.join(
        ism_root, "checkpoints", "ov_models", args.sam2_variant)
    encoder_config = {"PERF_COUNT": "YES"}
    decoder_config = {"PERF_COUNT": "YES"}
    if args.device.upper().startswith("GPU"):
        encoder_config["INFERENCE_PRECISION_HINT"] = (
            "f16" if args.precision == "fp16" else "f32")
        decoder_config["INFERENCE_PRECISION_HINT"] = "f32"
    encoder = core.compile_model(
        core.read_model(os.path.join(model_dir, "ov_image_encoder.xml")),
        args.device,
        encoder_config,
    )
    decoder = core.compile_model(
        core.read_model(os.path.join(model_dir, "ov_mask_predictor.xml")),
        args.device,
        decoder_config,
    )
    return OVSam2ImagePredictor(encoder, decoder, ResizeLongestSide(1024))


def detect(compiled_model, rgb, threshold):
    input_height, input_width = compiled_model.input(0).shape[2:]
    tensor = cv2.resize(rgb, (input_width, input_height)).astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]
    request = compiled_model.create_infer_request()
    inference_start = time.perf_counter()
    request.infer({0: tensor})
    output = np.asarray(request.get_output_tensor(0).data).squeeze().T
    inference_ms = (time.perf_counter() - inference_start) * 1000
    device_ms = request_device_time_ms(request)
    candidates = output[output[:, 4] >= threshold]
    if not len(candidates):
        raise RuntimeError("YOLO found no object above detector threshold")
    boxes = candidates[:, :4].copy()
    boxes[:, 0] = candidates[:, 0] - candidates[:, 2] / 2
    boxes[:, 1] = candidates[:, 1] - candidates[:, 3] / 2
    boxes[:, 2] = candidates[:, 0] + candidates[:, 2] / 2
    boxes[:, 3] = candidates[:, 1] + candidates[:, 3] / 2
    boxes[:, [0, 2]] *= rgb.shape[1] / input_width
    boxes[:, [1, 3]] *= rgb.shape[0] / input_height
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, rgb.shape[1] - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, rgb.shape[0] - 1)
    best = max(nms(boxes, candidates[:, 4]), key=lambda index: candidates[index, 4])
    return boxes[best], float(candidates[best, 4]), inference_ms, device_ms


def save_detection(args, mask, score):
    import pycocotools.mask as mask_utils

    results_dir = os.path.join(args.output_dir, "sam6d_results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "detection_yolo_sam.json")
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump([{"scene_id": 0, "image_id": 0, "category_id": 1,
                    "score": score, "segmentation": encoded}], output_file)
    return path


def save_top_mask_visualization(rgb, mask, score, output_path, box=None):
    visualization = rgb.copy()
    visualization[mask] = (
        visualization[mask].astype(np.float32) * 0.35 +
        np.asarray([0, 220, 70], dtype=np.float32) * 0.65
    ).astype(np.uint8)
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 70, 40), 2)
        label_origin = (x1, max(y1 - 8, 20))
    else:
        label_origin = (12, 28)
    cv2.putText(
        visualization, f"score={score:.4f}", label_origin,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


def run_pem(args, segmentation_path):
    pem_dir = os.path.join(args.sam6d_root, "Pose_Estimation_Model")
    output_templates = os.path.join(args.output_dir, "templates")
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(output_templates):
        os.symlink(args.templates_dir, output_templates)
    topk = 5 if args.pipeline == "ism" else 1
    command = [
        sys.executable, "run_inference_custom_openvino.py",
        "--device", args.device, "--precision", args.precision,
        "--output_dir", args.output_dir, "--cad_path", args.cad_path,
        "--rgb_path", args.rgb_path, "--depth_path", args.depth_path,
        "--cam_path", args.camera_path, "--seg_path", segmentation_path,
        "--topk_ism_score", str(topk),
        "--max_batch_size", str(topk),
        "--gt_path", "",
    ]
    if args.pipeline == "ism":
        command.extend(["--select_result_index", "0"])
    subprocess.run(command, cwd=pem_dir, check=True)
    results_dir = os.path.join(args.output_dir, "sam6d_results")
    result_path = os.path.join(
        results_dir, f"detection_pem_ov_{args.device}_{args.precision}.json")
    visualization_path = os.path.join(
        results_dir, f"vis_pem_ov_{args.device}_{args.precision}.png")
    pipeline_name = args.pipeline.replace("-", "_")
    pipeline_result_path = os.path.join(
        results_dir,
        f"detection_pem_{pipeline_name}_ov_{args.device}_{args.precision}.json")
    pipeline_visualization_path = os.path.join(
        results_dir,
        f"vis_pem_{pipeline_name}_ov_{args.device}_{args.precision}.png")
    shutil.copy2(result_path, pipeline_result_path)
    shutil.copy2(visualization_path, pipeline_visualization_path)
    os.remove(result_path)
    os.remove(visualization_path)
    with open(pipeline_result_path, encoding="utf-8") as result_file:
        results = json.load(result_file)
    return results[0] if args.pipeline == "ism" else max(
        results, key=lambda item: item["score"])


def run_ism(args):
    ism_dir = os.path.join(args.sam6d_root, "Instance_Segmentation_Model")
    frame_dir = os.path.dirname(args.rgb_path)
    staged_camera = os.path.join(frame_dir, "camera.json")
    if os.path.realpath(args.camera_path) != os.path.realpath(staged_camera):
        shutil.copy2(args.camera_path, staged_camera)
    ism_output = os.path.join(args.output_dir, "sam6d_results", "ism")
    timing_path = os.path.join(ism_output, "timing.json")
    os.makedirs(ism_output, exist_ok=True)
    command = [
        sys.executable, "infer_ism_ov.py",
        "--ov_model_dir", os.path.join(ism_dir, "checkpoints", "ov_models"),
        "--ov_device", args.device,
        "--precision", args.precision,
        "--ext",
        "--segmentor_model", "fastsam",
        "--image", args.rgb_path,
        "--cad", args.cad_path,
        "--templates_dir", args.templates_dir,
        "--output_dir", ism_output,
        "--timing_json", timing_path,
    ]
    subprocess.run(command, cwd=ism_dir, check=True)
    segmentation_path = os.path.join(ism_output, "detection_ism.json")
    with open(segmentation_path, encoding="utf-8") as detection_file:
        detections = json.load(detection_file)
    if not detections:
        raise RuntimeError("OpenVINO ISM produced no detections")
    with open(timing_path, encoding="utf-8") as timing_file:
        timings = json.load(timing_file)
    shutil.copy2(
        os.path.join(ism_output, "top_detection.png"),
        os.path.join(args.output_dir, "sam6d_results", "top_mask_ism.png"))
    return (
        segmentation_path,
        float(max(item["score"] for item in detections)),
        timings,
    )


def main():
    args = parse_args()
    rgb = cv2.cvtColor(cv2.imread(args.rgb_path), cv2.COLOR_BGR2RGB)
    yolo_score = None
    sam_score = None
    ism_score = None
    latency_ms = {}
    box = None
    if args.pipeline == "yolo-sam":
        core = ov.Core()
        detector_config = {"PERF_COUNT": "YES"} if args.device.upper().startswith("GPU") else {}
        detector = core.compile_model(
            core.read_model(os.path.join(args.detector_path, "yolo.xml")),
            args.device,
            detector_config)
        predictor = compile_sam2(core, args)
        box, yolo_score, yolo_inference_ms, yolo_device_ms = detect(
            detector, rgb, args.detector_threshold)
        encoder_ms, encoder_device_ms = predictor.set_image(rgb)
        mask, sam_score, decoder_ms, decoder_device_ms = predictor.predict_box(box)
        sam_timings = {
            "sam2_encoder_ms": encoder_ms,
            "sam2_decoder_ms": decoder_ms,
            "sam2_inference_ms": encoder_ms + decoder_ms,
            "sam2_encoder_device_ms": encoder_device_ms,
            "sam2_decoder_device_ms": decoder_device_ms,
            "sam2_device_ms": encoder_device_ms + decoder_device_ms,
        }
        latency_ms = {
            "yolo_inference_ms": yolo_inference_ms,
            "yolo_device_ms": yolo_device_ms,
            **sam_timings,
        }
        mask_score = yolo_score * sam_score
        segmentation_path = save_detection(args, mask, mask_score)
        save_top_mask_visualization(
            rgb, mask, mask_score,
            os.path.join(args.output_dir, "sam6d_results", "top_mask_yolo_sam.png"),
            box=box)
    else:
        segmentation_path, ism_score, latency_ms = run_ism(args)
    pem_result = run_pem(args, segmentation_path)
    rotation = np.asarray(pem_result["R"], dtype=float)
    translation_m_array = np.asarray(pem_result["t"], dtype=float) / 1000.0
    if float(pem_result["score"]) <= 0 or np.linalg.norm(translation_m_array) <= 1e-6:
        raise RuntimeError("PEM produced an invalid zero-score pose")
    translation_m = translation_m_array.tolist()
    result = {
        "pipeline": args.pipeline,
        "translation_m": translation_m,
        "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
        "yolo_score": yolo_score,
        "sam_score": sam_score,
        "ism_score": ism_score,
        "pose_score": float(pem_result["score"]),
        "box_xyxy": None if box is None else box.tolist(),
        "latency_ms": latency_ms,
    }
    with open(args.result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2)


if __name__ == "__main__":
    main()