from typing import List
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from ..utils.utils import Hypothesis
from .decoder import Decoder
from .swin_encoder import SwinV2PretrainedEncoder

class CoMER(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.encoder = SwinV2PretrainedEncoder(d_model)

        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

    def forward(self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor) -> FloatTensor:
        feature, mask = self.encoder(img, img_mask)
        feature = torch.cat((feature, feature), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        out = self.decoder(feature, mask, tgt)
        return out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        feature, mask = self.encoder(img, img_mask)
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature, vocab_size=self.vocab_size
        )