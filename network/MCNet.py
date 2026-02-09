import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import clip
import numpy as np
from network.convnext import convnext_tiny, convnext_small, convnext_base

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNextModel(nn.Module):
    embed_dims = {
        "convnext_tiny": [96, 192, 384, 768],    # c1, c2, c3, c4
        "convnext_small": [96, 192, 384, 768],
        "convnext_base": [128, 256, 512, 1024]
    }
    def __init__(self, model_name='convnext_base', pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.cur_embed_dims = self.embed_dims[model_name]  # 当前模型的维度配置
        
        self.convnext = eval(model_name)(pretrained=pretrained)

        self.depth_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        nn.init.kaiming_normal_(self.depth_adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.depth_adapter.bias is not None:
            nn.init.constant_(self.depth_adapter.bias, 0)

    def forward(self, rgb, depth):
        depth_3ch = self.depth_adapter(depth)  # (B, 3, H, W)
        
        rgb_c1, rgb_c2, rgb_c3, rgb_c4 = self.convnext(rgb)
        depth_c1, depth_c2, depth_c3, depth_c4 = self.convnext(depth_3ch)
        
        return {
            'rgb': {'c2': rgb_c2, 'c3': rgb_c3, 'c4': rgb_c4},
            'depth': {'c2': depth_c2, 'c3': depth_c3, 'c4': depth_c4}
        }

class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=device)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.output_dim = 512  

    def forward(self, texts):
        text_tokens = clip.tokenize(texts, truncate=True).to(device)
        
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
        
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats

class CrossModalFusion(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.visual_dim = visual_dim 
        self.text_dim = text_dim    
        
        self.text_proj = nn.Linear(text_dim, visual_dim)
        nn.init.kaiming_normal_(self.text_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.text_proj.bias, 0)
        
        self.rgb_text_attn = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        self.depth_text_attn = nn.MultiheadAttention(visual_dim, num_heads=8, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1),
            nn.BatchNorm2d(visual_dim),  
            nn.Sigmoid()
        )
      
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feats, text_feat):
        rgb_c4 = visual_feats['rgb']['c4']  # (B, C, H, W)
        depth_c4 = visual_feats['depth']['c4']
        B, C, H, W = rgb_c4.shape
        
        text_proj = self.text_proj(text_feat)  # (B, C)
        text_expand = text_proj.unsqueeze(1)   # (B, 1, C) 
        
        rgb_flat = rgb_c4.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        rgb_text_attn, _ = self.rgb_text_attn(rgb_flat, text_expand, text_expand)
        rgb_text_feat = rgb_text_attn.permute(0, 2, 1).view(B, C, H, W) 
        
        depth_flat = depth_c4.flatten(2).permute(0, 2, 1)
        depth_text_attn, _ = self.depth_text_attn(depth_flat, text_expand, text_expand)
        depth_text_feat = depth_text_attn.permute(0, 2, 1).view(B, C, H, W)
        
        gate_input = torch.cat([rgb_text_feat, depth_text_feat], dim=1)  # (B, 2C, H, W)
        gate_weight = self.gate(gate_input)  # (B, C, H, W)
        fused_feat = gate_weight * rgb_text_feat + (1 - gate_weight) * depth_text_feat
       
        return fused_feat

class MultiScaleOptimizer(nn.Module):
    def __init__(self, c2_dim, c3_dim, c4_dim, out_dim=256):
        super().__init__()
        self.out_dim = out_dim 
        
        self.scale_c2 = self._build_scale_layer(c2_dim, out_dim)
        self.scale_c3 = self._build_scale_layer(c3_dim, out_dim)
        self.scale_c4 = self._build_scale_layer(c4_dim, out_dim)

        self.cross_attn = nn.MultiheadAttention(out_dim, num_heads=8, batch_first=True)
  
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim)
        )
 
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_scale_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, fused_c4, multi_scale_feats):
        rgb_c2 = multi_scale_feats['rgb']['c2']
        rgb_c3 = multi_scale_feats['rgb']['c3']
        depth_c2 = multi_scale_feats['depth']['c2']
        depth_c3 = multi_scale_feats['depth']['c3']
        
        H2, W2 = rgb_c2.shape[2:]

        rgb_c2 = self.scale_c2(rgb_c2)  # (B, 256, H2, W2)
        depth_c2 = self.scale_c2(depth_c2)
        
        rgb_c3 = self.scale_c3(rgb_c3)
        rgb_c3_up = F.interpolate(rgb_c3, size=(H2, W2), mode='bilinear', align_corners=True)
        depth_c3 = self.scale_c3(depth_c3)
        depth_c3_up = F.interpolate(depth_c3, size=(H2, W2), mode='bilinear', align_corners=True)
        
        fused_c4 = self.scale_c4(fused_c4)
        fused_c4_up = F.interpolate(fused_c4, size=(H2, W2), mode='bilinear', align_corners=True)
        
        B, C, H, W = fused_c4_up.shape
        feat_c4 = fused_c4_up.flatten(2).permute(0, 2, 1) # (B, H*W, 256) - 查询（语义主导）
        feat_c3 = (rgb_c3_up + depth_c3_up).flatten(2).permute(0, 2, 1)   # (B, H*W, 256)
        feat_c2 = (rgb_c2 + depth_c2).flatten(2).permute(0, 2, 1)      # (B, H*W, 256)
        
        attn_out, _ = self.cross_attn(feat_c4, feat_c3, feat_c2)  # (B, H*W, 256)
        attn_feat = attn_out.permute(0, 2, 1).view(B, C, H, W)  # 恢复空间维度
        
        combined_feat = attn_feat + feat_c4 + feat_c3 + feat_c2
        optimized_feat = self.refine(combined_feat)  # 减少冗余噪声
        
        return optimized_feat


class SalientPredictor(nn.Module):
    def __init__(self, in_dim=256, target_scale=4):
        super().__init__()

        self.predict = nn.Sequential(
            nn.Conv2d(in_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
   
        for m in self.predict.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, orig_size):
        pred = self.predict(feat)  
        sal_map = F.interpolate(pred, size=orig_size, mode='bilinear', align_corners=True)
        return sal_map


class MCNet(nn.Module):
    ### Multi-modal Complement Network (MCNet)
    def __init__(self, convnext_model_name='convnext_base'):
        super().__init__()

        self.visual_encoder = ConvNextModel(model_name=convnext_model_name)
        self.text_encoder = CLIPTextEncoder()
        
        embed_dims = self.visual_encoder.cur_embed_dims  # [c1, c2, c3, c4]
        visual_c4_dim = embed_dims[-1]  # c4维度（视觉高层特征）
        text_dim = self.text_encoder.output_dim  # CLIP文本维度（768）
        
        self.cross_modal_fusion = CrossModalFusion(visual_dim=visual_c4_dim, text_dim=text_dim)
        
        self.multi_scale_opt = MultiScaleOptimizer(
            c2_dim=embed_dims[1],
            c3_dim=embed_dims[2],
            c4_dim=embed_dims[3],
            out_dim=256
        )
        
        self.sal_predictor = SalientPredictor(in_dim=256)

        self.bce_loss = nn.BCELoss()  # 像素级分类损失
        self.iou_loss = lambda x, y: 1 - torch.mean(  # IoU损失（提升区域一致性）
            (x * y).sum(dim=[1,2,3]) / ((x + y - x * y).sum(dim=[1,2,3]) + 1e-8)
        )
        self.semantic_consist_loss = nn.CosineEmbeddingLoss()  # 语义-视觉一致性损失
        
        self.text2feat_proj = nn.Linear(text_dim, 256)
        nn.init.kaiming_normal_(self.text2feat_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.text2feat_proj.bias, 0)

    def forward(self, rgb, depth, texts, gt=None):
        B, _, H_orig, W_orig = rgb.shape
        outputs = {}
        
        visual_feats = self.visual_encoder(rgb, depth)
        text_feats = self.text_encoder(texts)  # (B, 512)
        
        text_feats = text_feats.float()  
    
        fused_c4 = self.cross_modal_fusion(visual_feats, text_feats)
        
        optimized_feat = self.multi_scale_opt(fused_c4, visual_feats)  # (B, 256, H2, W2)
        
        sal_map = self.sal_predictor(optimized_feat, orig_size=(H_orig, W_orig))
        outputs['sal_map'] = sal_map
        
        if self.training and gt is not None:
            bce_loss = self.bce_loss(sal_map, gt)
            iou_loss = self.iou_loss(sal_map, gt)
            base_loss = bce_loss + iou_loss
            
            feat_compress = F.adaptive_avg_pool2d(optimized_feat, (1, 1)).squeeze(-1).squeeze(-1)
            text_proj = self.text2feat_proj(text_feats)
            semantic_loss = self.semantic_consist_loss(feat_compress, text_proj, torch.ones(B, device=device))
            
            total_loss = (
                base_loss + 
                0.5 * semantic_loss  # 语义引导权重
            )
            
            outputs['losses'] = {
                'base_loss': base_loss,
                'semantic_loss': semantic_loss,
                'total_loss': total_loss
            }
        
        return outputs
