"""Swin Transformer V2 encoder for SwinCoMER.

## The channel-axis bug this module now handles

timm returns Swin feature maps in **NHWC** order: the last stage of
`swinv2_tiny_window16_256` comes out as `[B, 8, 8, 768]`, not `[B, 768, 8, 8]`.

The original implementation read the channel count as `features[-1].shape[1]`
and fed the tensor straight into a `Conv2d`. With NHWC that picks up the
*height* (8) as the channel count and treats `(H, C) = (8, 768)` as the spatial
grid. Nothing raises — `Conv2d` happily convolves over whatever two trailing
axes it is given — so the mistake is silent, and the decoder ends up
cross-attending over **6144 "positions" instead of 8x8 = 64**.

Two consequences:

* **Correctness.** Those 6144 positions are (row, channel) pairs, not spatial
  locations. The 2D layout the Swin encoder exists to capture is destroyed
  before the decoder ever sees it.
* **Cost.** Every cross-attention step does ~96x the intended work. Measured
  on CPU with the bundled CROHME sample: 107s at beam_size 2, 407s at beam 8.

## Backward compatibility

Checkpoints trained before this fix have `projection.0.weight` of shape
`(d_model, 8, 1, 1)` — they encode the bug in their weights and cannot run the
corrected path. Loading one still works: a state-dict pre-hook detects the
legacy shape, rebuilds `projection` to match and switches `forward` back to the
old behaviour, so existing checkpoints reproduce their previous results exactly.
`self.legacy_hw_as_channels` says which path is active, and a warning is printed
on load so the state is never silent.

New training runs get the corrected path and will need their own checkpoints.
"""
import warnings
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor
import timm

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

        # timm states the channel count per stage, so take it from there rather
        # than guessing which tensor axis holds it. Whether that count lands on
        # axis 1 (NCHW) or the last axis (NHWC) then tells us the layout, and
        # the layout varies by timm version - which is exactly what made the
        # original bug survive unnoticed.
        channels = self.backbone.feature_info.channels()
        self.feature_dim = channels[-1] if output_layer == -1 else channels[output_layer]

        dummy_input = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            features = self._select(self.backbone(dummy_input))

        if features.shape[1] == self.feature_dim:
            self.channels_last = False
            self.output_size = tuple(features.shape[2:])
        elif features.shape[-1] == self.feature_dim:
            self.channels_last = True
            self.output_size = tuple(features.shape[1:3])
        else:
            raise RuntimeError(
                f"Cannot locate the channel axis in a {tuple(features.shape)} "
                f"feature map: timm reports {self.feature_dim} channels for "
                f"'{model_name}' but neither axis 1 nor the last axis matches. "
                "This encoder needs updating for your timm version."
            )

        # Set by the load hook below when an older checkpoint is restored.
        self.legacy_hw_as_channels = False
        self.projection = self._build_projection(self.feature_dim)

        if self.use_pos_enc:
            self.pos_encoder = ImgPosEnc(d_model=d_model)

        self._register_load_state_dict_pre_hook(self._adapt_legacy_projection)

    def _build_projection(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, self.d_model, kernel_size=1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(inplace=True)
        )

    def _adapt_legacy_projection(self, state_dict, prefix, *args) -> None:
        """Reshape this module to match a pre-fix checkpoint, if that is what
        is being loaded.

        Runs before the weights are copied in, so `projection` can still be
        replaced with one whose `in_channels` matches what the checkpoint
        expects. Without this, restoring an old checkpoint fails with a bare
        size-mismatch error that says nothing about why.
        """
        weight = state_dict.get(prefix + "projection.0.weight")
        if weight is None or weight.dim() != 4:
            return

        in_channels = weight.shape[1]
        if in_channels == self.feature_dim:
            return  # Already a corrected checkpoint.

        self.legacy_hw_as_channels = True
        self.projection = self._build_projection(in_channels)
        warnings.warn(
            f"Loading a checkpoint trained before the NHWC channel-axis fix: "
            f"projection expects {in_channels} input channels, not "
            f"{self.feature_dim}. Running in legacy mode, which reproduces the "
            "original results but attends over ~96x more memory positions than "
            "intended. Retrain to get the corrected encoder.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _select(self, features):
        """Pick the stage this encoder was configured to read."""
        return features[-1] if self.output_layer == -1 else features[0]

    def _extract_features(self, x: FloatTensor) -> FloatTensor:
        """
        Extract features from the pretrained backbone, as NCHW.

        Parameters
        ----------
        x: FloatTensor
            [B, C, H, W]

        Returns
        -------
        FloatTensor
            [B, C', H', W']
        """
        features = self._select(self.backbone(x))

        if self.legacy_hw_as_channels:
            # Reproduce the original behaviour exactly: hand the backbone's
            # tensor to Conv2d untouched, whatever its layout, because that is
            # what the loaded weights were trained against.
            return features

        if self.channels_last:
            features = rearrange(features, 'b h w c -> b c h w')
        return features

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

        features = self._extract_features(x)   # [B, C', H', W']
        features = self.projection(features)   # [B, d_model, H', W']

        features = rearrange(features, 'b c h w -> b h w c')

        B, H, W, _ = features.shape
        # NOTE: img_mask is accepted but not used. Every image reaching this
        # encoder is resized to exactly 256x256 upstream, so no padding exists
        # to mask. Feeding variable-sized batches would need this to downsample
        # img_mask onto the [H', W'] grid instead - see the mask handling in
        # collate_fn, which already builds the real padding mask.
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=features.device)

        if self.use_pos_enc:
            features = self.pos_encoder(features, mask)

        return features, mask
