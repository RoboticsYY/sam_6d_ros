import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from ..utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from .utils import BatchedData
from copy import deepcopy
import os
import os.path as osp
import openvino as ov


def get_ov_model(model_path, device='CPU'):
    core = ov.Core()
    ov_model = core.read_model(model_path)
    # ov_model.reshape({"x": ov.PartialShape([-1, 3, 224, 224])})
    compiled_model = ov.compile_model(ov_model, device)
    return compiled_model

def ov_to_torch(x):
    ### Convert x which is a ov output to torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        return x
    elif isinstance(x, list):
        out_list = []
        for item in x:
            if isinstance(item, np.ndarray):
                out_list.append(torch.from_numy(item))
            else:
                out_list.append(item)
        return out_list
    else:
        raise NotImplementedError

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

descriptor_map = {
    "dinov2_vits14": "vit_small",
    "dinov2_vitb14": "vit_base",
    "dinov2_vitl14": "vit_large",
    "dinov2_vitg14": "vit_giant2",
}

from enum import Enum
from typing import Union


class Weights(Enum):
    LVD142M = "LVD142M"

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from . import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model

class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        checkpoint_dir,
        patch_size=14,
        validpatch_thresh=0.5,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = _make_dinov2_model(arch_name=descriptor_map[model_name], pretrained=False)
        # self.model.load_state_dict(torch.load(osp.join(checkpoint_dir, f"{model_name}_pretrain.pth")))
        ## SUBSTITUE dino v2 model with openvino compiled dino v2 model
        # self.ov_model = get_ov_model(model_path="/home/test/SAM-6D/SAM-6D/Instance_Segmentation_Model/descriptor_dinov2_model_infer.xml", device="GPU")
        print(f"checkpoint_dir: {checkpoint_dir}")
        current_dir = os.getcwd()
        print(f"current_dir: {current_dir}")
        ov_ism_path = os.path.normpath(os.path.join(current_dir, checkpoint_dir, "descriptor_dinov2_model_infer.xml"))
        self.ov_model = get_ov_model(model_path=ov_ism_path, device="GPU")

        self.validpatch_thresh = validpatch_thresh
        self.token_name = token_name
        self.chunk_size = chunk_size # chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )
        self.patch_kernel = torch.nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)
        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    def process_rgb_proposals(self, image_np, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs

    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                # features = self.model(images)
                # print(features[0]["x_norm_clstoken"], features[-1])
                # print("images.numpy()",images.numpy().shape)
                features = self.ov_model(images.numpy())[-1]
                features = ov_to_torch(features)
        else:  # get both features
            raise NotImplementedError
        return features

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            shape = batch_rgbs[idx_batch].shape
            cur_batch_size = shape[0]
            if cur_batch_size < self.chunk_size:
                pad = torch.zeros(self.chunk_size-cur_batch_size, *shape[1:])
                tmp_batch = torch.cat([batch_rgbs[idx_batch], pad], dim=0)
                feats = self.compute_features(
                tmp_batch, token_name="x_norm_clstoken"
            )
            else:
                feats = self.compute_features(
                    batch_rgbs[idx_batch], token_name="x_norm_clstoken"
                )
            feats = feats[:cur_batch_size]
            features.cat(feats)
        return features.data


    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)


    def process_masks_proposals(self, masks, boxes):
        """
        1. Normalize image with DINOv2 transfom
        2. Mask and crop each proposals
        3. Resize each proposals to predefined longest image size
        """
        num_proposals = len(masks)
        masks.unsqueeze_(1) # [N_proposal, 1, ImgH, ImgW]
        processed_masks = self.rgb_proposal_processor(
            masks, boxes
        ).squeeze_()  # [N, 1, target_size, target_size]
        return processed_masks

    @torch.no_grad()
    def forward_patch_tokens(self, image_np, proposals):
        # with preprocess
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        processed_masks = self.process_masks_proposals(proposals.masks, proposals.boxes)
        return self.forward_by_chunk_v2(processed_rgbs, processed_masks)

    @torch.no_grad()
    def forward_by_chunk_v2(self, processed_rgbs, masks):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs  # free memory
        del masks
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_masked_patch_feature(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            features.cat(feats)
        return features.data

    @torch.no_grad()
    def compute_masked_patch_feature(self, images, masks):
        # without preprocess
        if images.shape[0] > self.chunk_size:
            features = self.forward_by_chunk_v2(images, masks)
        else:
            # features = self.model(images, is_training=True)[0]["x_norm_patchtokens"]
            # print(features.shape)
            features = self.ov_model(images.numpy())[2] # x_norm_patchtokens is the third output
            features = ov_to_torch(features)
            features_mask = self.patch_kernel(masks).flatten(-2) > self.validpatch_thresh
            features_mask = features_mask.unsqueeze(-1).repeat(1, 1, features.shape[-1])
            features = F.normalize(features * features_mask, dim=-1)
        return features


    @torch.no_grad()
    def forward(self, image_np, proposals):
        # with preprocess
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        # print(image_np.shape, processed_rgbs.shape)
        processed_masks = self.process_masks_proposals(proposals.masks, proposals.boxes)

        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=processed_masks)
        del processed_rgbs  # free memory
        del processed_masks
        cls_features = BatchedData(batch_size=self.chunk_size)
        patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            shape = batch_rgbs[idx_batch].shape
            cur_batch_size = shape[0]
            # print(cur_batch_size, self.chunk_size)
            if cur_batch_size < self.chunk_size:
                pad = torch.zeros(self.chunk_size-cur_batch_size, *shape[1:])
                tmp_batch = torch.cat([batch_rgbs[idx_batch], pad], dim=0)
                cls_feats, patch_feats = self.compute_cls_and_patch_features(
                tmp_batch, batch_masks[idx_batch]
                )
            else:
                cls_feats, patch_feats = self.compute_cls_and_patch_features(
                    batch_rgbs[idx_batch], batch_masks[idx_batch]
                )
            cls_feats, patch_feats = cls_feats[:cur_batch_size], patch_feats[:cur_batch_size]
            cls_features.cat(cls_feats)
            patch_features.cat(patch_feats)
        
        return cls_features.data, patch_features.data

    def compute_cls_and_patch_features(self, images, masks):
        # features = self.model(images, is_training=True)[0]
        # for k, v in features.items():
        #     print(k, v.shape)
        # patch_features = features["x_norm_patchtokens"]
        # cls_features = features["x_norm_clstoken"]
        features = self.ov_model(images.numpy())
        patch_features = ov_to_torch(features[2]) # x_norm_patchtokens is the third output
        cls_features = ov_to_torch(features[0]) # x_norm_clstoken is the first output
        if images.shape[0] != masks.shape[0]:
            mask_shape = masks.shape
            pad_mask = torch.zeros(images.shape[0]-masks.shape[0], *mask_shape[1:])
            masks = torch.cat([masks, pad_mask], dim=0)

        features_mask = self.patch_kernel(masks).flatten(-2) > self.validpatch_thresh
        features_mask = features_mask.unsqueeze(-1).repeat(1, 1, patch_features.shape[-1])
        patch_features = F.normalize(patch_features * features_mask, dim=-1)

        return cls_features, patch_features
        
