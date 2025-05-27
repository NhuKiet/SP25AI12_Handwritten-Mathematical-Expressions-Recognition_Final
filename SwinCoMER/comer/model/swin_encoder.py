import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from einops import rearrange
from typing import List, Tuple, Optional, Dict, Any
import timm
from timm.models.swin_transformer_v2 import SwinTransformerV2

from ..model.pos_enc import ImgPosEnc


class SwinV2PretrainedEncoder(pl.LightningModule):
    """
    Swin Transformer V2 Pretrained Encoder
    """
    def __init__(
        self,
        d_model: int = 512,
        pretrained: bool = True,
        model_name: str = 'swinv2_tiny_window16_256',
        use_pos_enc: bool = True,
        freeze_backbone: bool = False,
        output_layer: int = -1
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pos_enc = use_pos_enc
        self.output_layer = output_layer
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3) if output_layer == -1 else (output_layer,)
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        dummy_input = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        if output_layer == -1:
            self.feature_dim = features[-1].shape[1]
            self.output_size = features[-1].shape[2:]
        else:
            self.feature_dim = features[0].shape[1]
            self.output_size = features[0].shape[2:]
        
        self.projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        if self.use_pos_enc:
            self.pos_encoder = ImgPosEnc(d_model=d_model)
    
    def _extract_features(self, x: FloatTensor) -> FloatTensor:
        """
        Extract features from the pretrained backbone
        
        Parameters
        ----------
        x: FloatTensor
            [B, C, H, W]
            
        Returns
        -------
        FloatTensor
            [B, C', H', W']
        """
        features = self.backbone(x)
        
        if self.output_layer == -1:
            return features[-1]
        else:
            return features[0]
    
    def forward(self, x: FloatTensor, img_mask: Optional[LongTensor] = None) -> Tuple[FloatTensor, LongTensor]:
        """
        Parameters
        ----------
        x: FloatTensor
            [B, C, H, W]
        img_mask: Optional[LongTensor]
            [B, H, W]
            
        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            features: [B, H', W', d_model]
            mask: [B, H', W']
        """

        features = self._extract_features(x)  # [B, C', H', W']
        features = self.projection(features)  # [B, d_model, H', W']
        
        features = rearrange(features, 'b c h w -> b h w c')
        
        B, H, W, _ = features.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=features.device)
        
        if self.use_pos_enc:
            features = self.pos_encoder(features, mask)
        
        return features, mask