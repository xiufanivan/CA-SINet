import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvt import pvt_v2_b2
from lib.Module import NeighborConnectionDecoder, ReverseStage, MCCS


class Network(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- PVT Backbone ----
        path = r'./pvt_v2_b2.pth'
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

        # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

        # ---- my module ----
        self.mtem = MCCS(128, 320, 512, channel)

    def forward(self, x):
        # Feature Extraction

        pvt = self.backbone(x)
        x1 = pvt[0]  # 64x176x176
        x2 = pvt[1]  # 128x88x88
        x3 = pvt[2]  # 320x44x44
        x4 = pvt[3]  # 512x22x22

        # Receptive Field Block (enhanced)
        x2, x3, x4 = self.mtem(x2, x3, x4)

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4, x3, x2)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')  # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4, guidance_g)  # (bs, 1, 11, 11)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred
