import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import SparseToDenseTransformer
from model_utils import compute_feature_similarity, compute_fine_Rt
from loss_utils import compute_correspondence_loss
from pointnet2_utils import QueryAndGroup
from pytorch_utils import SharedMLP, Conv1d

DEBUG_TAG = False

class FinePointMatching(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(FinePointMatching, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * .02)
        self.PE = PositionalEncoding(cfg.hidden_dim, r1=cfg.pe_radius1, r2=cfg.pe_radius2)
        
        self.transformers = nn.ModuleList([
            SparseToDenseTransformer(
                cfg.hidden_dim,
                num_heads=4,
                sparse_blocks=['self', 'cross'],
                dropout=None,
                activation_fn='ReLU',
                focusing_factor=cfg.focusing_factor,
                with_bg_token=True,
                replace_bg_token=True
            ) for _ in range(self.nblock)
        ])

    def forward(self, p1, f1, geo1, fps_idx1, 
                      p2, f2, geo2, fps_idx2, 
                      radius, model, init_R, init_t):
        B = p1.size(0)
        
        p1_ = (p1 - init_t.unsqueeze(1)) @ init_R
        
        f1 = self.in_proj(f1) + self.PE(p1_)
        f1 = torch.cat([self.bg_token.repeat(B,1,1), f1], dim=1) # adding bg

        f2 = self.in_proj(f2) + self.PE(p2)
        f2 = torch.cat([self.bg_token.repeat(B,1,1), f2], dim=1) # adding bg
        
        atten_list = []
        for idx in range(self.nblock):

            if DEBUG_TAG:
                self.save_iteration_inputs(idx, f1, geo1, fps_idx1, f2, geo2, fps_idx2)
            
            f1, f2 = self.transformers[idx](f1, geo1, fps_idx1, f2, geo2, fps_idx2)
            
            if self.training or idx==self.nblock-1:
                atten_list.append(compute_feature_similarity(
                    self.out_proj(f1),
                    self.out_proj(f2),
                    self.cfg.sim_type,
                    self.cfg.temp,
                    self.cfg.normalize_feat
                ))

        fine_Rt_atten = atten_list[-1]
        fine_Rt_model_pts = model / (radius.reshape(-1, 1, 1) + 1e-6)
        return fine_Rt_atten, fine_Rt_model_pts

        # pred_R, pred_t, pred_pose_score = compute_fine_Rt(
        #     atten_list[-1], p1, p2,
        #     model / (radius.reshape(-1, 1, 1) + 1e-6),
        # )
        # pred_t = pred_t * (radius.reshape(-1, 1)+1e-6) 
        # return pred_R, pred_t, pred_pose_score

    def save_iteration_inputs(self, iteration_idx, f1, geo1, fps_idx1, f2, geo2, fps_idx2):
        """[For Debug] Save the real inputs of each transformer iteration"""
        save_dir = "transformer_iteration_inputs"
        os.makedirs(save_dir, exist_ok=True)
        
        iteration_file = os.path.join(save_dir, f"iteration_{iteration_idx}.npz")
        
        inputs = {
            'f1': f1.detach().cpu().numpy(),
            'geo1': geo1.detach().cpu().numpy(),
            'fps_idx1': fps_idx1.detach().cpu().numpy(),
            'f2': f2.detach().cpu().numpy(),
            'geo2': geo2.detach().cpu().numpy(),
            'fps_idx2': fps_idx2.detach().cpu().numpy(),
        }
        
        np.savez(iteration_file, **inputs)
        print(f"[For Debug] Save iteration {iteration_idx} inputs to: {iteration_file}")



class PositionalEncoding(nn.Module):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(PositionalEncoding, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3

        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        print(f"[PositionalEncoding] pts1 shape :{pts1.shape}")
        if pts2 is None:
            # pts2 = pts1
            pts2 = pts1+0.00000001
        """
        Error: [GPU] Different primitive with id 'parameter:pts1_/group1/BallQuery_cldnn_custom_preprocess' exists already
        
        When pts2 = pts1, the GPU kernel will display a different primitive already present during model compilation.
        The specific cause of this issue is unknown, but it is confirmed to be related to OV graph transformation optimization.
        This may be because the same source data, after OV transformation optimization, is considered a duplicate primitive when the GPU kernel primitive is executed.
        By adding a minimum value for pts2, computation accuracy is not affected and OV optimization prevents the data from being considered the same source.
        """
        # scale1
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat1 = self.mlp1(feat1)
        # feat1 = F.max_pool2d(feat1, kernel_size=[1, feat1.size(3)]) # onnx export error, due to ONNX operations not support F.max_pool2d
        feat1 = torch.amax(feat1, dim=3, keepdim=True)

        # scale2
        feat2 = self.group2(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat2 = self.mlp2(feat2)
        # feat2 = F.max_pool2d(feat2, kernel_size=[1, feat2.size(3)])
        feat2 = torch.amax(feat2, dim=3, keepdim=True)

        feat = torch.cat([feat1, feat2], dim=1).squeeze(-1)
        feat = self.mlp3(feat).transpose(1,2)
        return feat

