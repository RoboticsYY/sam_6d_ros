from ultralytics import YOLO
from pathlib import Path
from typing import Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.utils.amg import MaskData
import logging
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import pytorch_lightning as pl
from ultralytics import yolo  # noqa
from ultralytics.nn.autobackend import AutoBackend


class CustomYOLO(YOLO):
    def __init__(
        self,
        model,
        iou,
        conf,
        max_det,
        segmentor_width_size,
        selected_device="cpu",
        verbose=False,
    ):
        YOLO.__init__(
            self,
            model,
        )
        self.overrides["iou"] = iou
        self.overrides["conf"] = conf
        self.overrides["max_det"] = max_det
        self.overrides["verbose"] = verbose
        self.overrides["imgsz"] = segmentor_width_size

        self.overrides["conf"] = 0.25
        self.overrides["mode"] = "predict"
        self.overrides["save"] = False

        self.predictor = yolo.v8.segment.SegmentationPredictor(
            overrides=self.overrides, _callbacks=self.callbacks
        )

        self.not_setup = True
        self.selected_device = selected_device
        logging.info(f"Init CustomYOLO done!")

    def setup_model(self, device, verbose=False):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        model = self.predictor.model or self.predictor.args.model
        self.predictor.args.half &= (
            device.type != "cpu"
        )  # half precision only supported on CUDA
        self.predictor.model = AutoBackend(
            model,
            device=device,
            dnn=self.predictor.args.dnn,
            data=self.predictor.args.data,
            fp16=self.predictor.args.half,
            fuse=True,
            verbose=verbose,
        )
        self.predictor.device = device
        self.predictor.model.eval()
        logging.info(f"Setup model at device {device} done!")

    def __call__(self, source=None, stream=False):
        return self.predictor(source=source, stream=stream)


class FastSAM(object):

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: dict = None,
        segmentor_width_size=None,
        device=None,
    ):
        self.model = CustomYOLO(
            model=checkpoint_path,
            iou=config.iou_threshold,
            conf=config.conf_threshold,
            max_det=config.max_det,
            selected_device=device,
            segmentor_width_size=segmentor_width_size,
        )
        self.segmentor_width_size = segmentor_width_size
        self.current_device = device
        logging.info(f"Init FastSAM done!")

    def postprocess_resize(self, detections, orig_size, update_boxes=False):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        if update_boxes:
            scale = orig_size[1] / self.segmentor_width_size
            detections["boxes"] = detections["boxes"].float() * scale
            detections["boxes"][:, [0, 2]] = torch.clamp(
                detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
            )
            detections["boxes"][:, [1, 3]] = torch.clamp(
                detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
            )
        return detections

    @torch.no_grad()
    def generate_masks(self, image) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
        detections = self.model(image)

        masks = detections[0].masks.data
        boxes = detections[0].boxes.data[:, :4]  # two lasts:  confidence and class

        # define class data
        mask_data = {
            "masks": masks.to(self.current_device),
            "boxes": boxes.to(self.current_device),
        }
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data
    
    @torch.no_grad()
    def generate_masks_from_bbox(self, image, bbox) -> Dict[str, Any]:
        """
        Generate masks only within the given bounding box region.
        Args:
            image (np.ndarray): Input image (H, W, C)
            bbox (tuple or list): (x1, y1, x2, y2) coordinates
        Returns:
            Dict[str, Any]: Mask data for the region
        """
        x_center, y_center, w, h = map(int, bbox)
        img_h, img_w = image.shape[:2]
        x1 = int(x_center - w // 2)
        y1 = int(y_center - h // 2)
        x2 = int(x_center + w // 2)
        y2 = int(y_center + h // 2)
        print(f"Generating masks for bbox: {bbox} -> crop: ({x1}, {y1}), ({x2}, {y2})")
        crop_img = image[y1:y2, x1:x2]
        # Ensure crop is valid and has 3 channels
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            raise ValueError(f"Cropped image has zero height or width: bbox={bbox}, crop_img.shape={crop_img.shape}")
        if crop_img.ndim == 2:
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
        elif crop_img.shape[-1] != 3:
            crop_img = crop_img[..., :3]
        # Resize crop to model input size
        expected_h, expected_w = 480, 640  # adjust to your model's input size if needed
        crop_img = cv2.resize(crop_img, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
        # Run model on cropped image
        detections = self.model(crop_img)

        masks = detections[0].masks.data  # shape: (N, H_model, W_model)
        boxes = detections[0].boxes.data[:, :4]  # two lasts: confidence and class

        # Rescale mask back to crop size
        crop_h = y2 - y1
        crop_w = x2 - x1
        if masks.ndim == 3:
            # masks: (N, H_model, W_model)
            masks_rescaled = F.interpolate(masks.unsqueeze(1).float(), size=(crop_h, crop_w), mode="bilinear", align_corners=False)[:, 0, :, :]
        else:
            masks_rescaled = masks

        # Adjust boxes to original image coordinates
        boxes = boxes + torch.tensor([x1, y1, x1, y1], dtype=boxes.dtype, device=boxes.device)

        mask_data = {
            "masks": masks_rescaled.to(self.current_device),
            "boxes": boxes.to(self.current_device),
        }
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data
