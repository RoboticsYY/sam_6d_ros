import torch
import torch.nn as nn

from feature_extraction import ViTEncoder
from ov_coarse_point_matching import CoarsePointMatching
from ov_fine_point_matching import FinePointMatching
from transformer import GeometricStructureEmbedding
from model_utils import sample_pts_feats, CustomDebugNode, compute_coarse_Rt, compute_fine_Rt

DEBUG_FLAG = False # True / False

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extraction = ViTEncoder(cfg.feature_extraction, self.fine_npoint)
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.coarse_point_matching = CoarsePointMatching(cfg.ov_coarse_point_matching)
        self.fine_point_matching = FinePointMatching(cfg.ov_fine_point_matching)

    # def forward(self, pts, rgb, rgb_choose, score, model, K, dense_po, dense_fo):
    def forward(self, pts, rgb, rgb_choose, model, dense_po, dense_fo):
        # 1. get dense features
        dense_pm, dense_fm, dense_po_out, dense_fo_out, radius = self.feature_extraction(pts, rgb, rgb_choose, dense_po, dense_fo)

        # 2. sample sparse features
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100
        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(dense_pm, dense_fm, self.coarse_npoint, return_index=True)

        # self.geo_embedding in ov model with error result -> -nan
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))
        if DEBUG_FLAG:
            geo_embedding_m = CustomDebugNode.apply(geo_embedding_m)
            print("[Deubg] geo_embedding_m ...")

        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(dense_po_out, dense_fo_out, self.coarse_npoint, return_index=True)

        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))
        
        # 3. coarse point matching
        coarse_Rt_atten, coarse_Rt_model_pts = self.coarse_point_matching(
            sparse_pm, sparse_fm, geo_embedding_m,
            sparse_po, sparse_fo, geo_embedding_o,
            radius, model
        )

        init_R, init_t = compute_coarse_Rt(
            coarse_Rt_atten, coarse_Rt_model_pts,
            self.cfg.ov_coarse_point_matching.nproposal1, self.cfg.ov_coarse_point_matching.nproposal2)

        # 4. fine point matching
        fine_Rt_atten, fine_Rt_model_pts = self.fine_point_matching(
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            dense_po_out, dense_fo_out, geo_embedding_o, fps_idx_o,
            radius, model, init_R, init_t
        )

        pred_R, pred_t, pred_pose_score = compute_fine_Rt(
            fine_Rt_atten, fine_Rt_model_pts,
        )
        pred_t = pred_t * (radius.reshape(-1, 1)+1e-6) 
        
        return pred_R, pred_t, pred_pose_score

