import zipfile
from typing import List
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor
import time
import editdistance

from .datamodule import Batch, vocab
from .model.comer_swin import CoMER
from .utils.utils import (ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out)
from torchvision.transforms import transforms
from .utils import utils

class LitCoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        dc: int = 128,
        # Defaults enable the Attention Refinement Module. Checkpoints trained
        # before this change carry cross_coverage/self_coverage=False in their
        # saved hyperparameters, and load_from_checkpoint restores those, so
        # existing checkpoints keep loading with ARM disabled as they were
        # trained.
        cross_coverage: bool = True,
        self_coverage: bool = True,
        beam_size: int = 5,
        max_len: int = 200,
        alpha: float = 0.6,
        early_stopping: bool = True,
        temperature: float = 1.0,
        learning_rate: float = 0.0001,
        patience: int = 5,
        vocab_size: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.comer_model = CoMER(
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

        self.comer_model.decoder.word_embed[0] = torch.nn.Embedding(vocab_size, d_model)
        self.comer_model.decoder.proj = torch.nn.Linear(d_model, vocab_size)

        # Registered as a submodule so torchmetrics handles device placement
        # and distributed reduction. Drives checkpoint selection via val_ExpRate.
        self.val_exprate_recorder = ExpRateRecorder()

        # Khởi tạo ExpRateRecorder, test_outputs và test_gts cho từng tập test
        self.exprate_recorders = {}
        self.test_outputs_dict = {}
        self.test_gts_dict = {}

    def forward(self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor) -> FloatTensor:
        return self.comer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.token_ids, self.device)
        out_hat = self(batch.images, batch.mask, tgt)
        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(batch))

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.token_ids, self.device)
        out_hat = self(batch.images, batch.mask, tgt)
        loss = ce_loss(out_hat, out)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))

        # Teacher-forced sequence accuracy: the decoder is handed the ground
        # truth prefix at every step, so this is far more optimistic than what
        # the model produces when it decodes on its own. Useful as a cheap
        # training signal, but never as a checkpoint selection criterion.
        # out is [2b, l]: the first b rows are l2r, the next b are r2l. A sample
        # counts as correct only when both directions are exact.
        pad_mask = out != vocab.PAD_IDX
        errs = ((out_hat.argmax(-1) != out) & pad_mask).sum(-1)  # [2b]
        errs = errs.view(2, -1).sum(0)                           # [b]
        seq_acc = (errs == 0).float().mean()
        self.log(
            "val_token_seq_acc",
            seq_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(batch),
        )

        # The real metric: autoregressive beam search, exactly the procedure
        # test_step uses. This is what val_ExpRate must mean for checkpoint
        # selection to be meaningful. It runs a full beam search per validation
        # batch, so validation is now much slower than a forward pass - cap it
        # with Trainer(limit_val_batches=...) if that is too expensive.
        hyps = self.approximate_joint_search(batch.images, batch.mask)
        self.val_exprate_recorder([h.seq for h in hyps], batch.token_ids)
        self.log(
            "val_ExpRate",
            self.val_exprate_recorder,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

    def test_step(self, batch: Batch, batch_idx, dataloader_idx=0):
        # Xác định tập test hiện tại
        test_splits = ["2014", "2016", "2019"]  # Chỉ sử dụng các tập test của CROHME
        split = test_splits[dataloader_idx]

        # Khởi tạo ExpRateRecorder, test_outputs và test_gts nếu chưa có
        if split not in self.exprate_recorders:
            self.exprate_recorders[split] = ExpRateRecorder()
            self.test_outputs_dict[split] = []
            self.test_gts_dict[split] = []

        # Đo thời gian suy luận
        start_time = time.time()
        hyps = self.approximate_joint_search(batch.images, batch.mask)
        inference_time = time.time() - start_time

        # Tính ExpRate
        self.exprate_recorders[split]([h.seq for h in hyps], batch.token_ids)

        # Lưu dự đoán và ground truth
        preds = [self.trainer.test_dataloaders[dataloader_idx].dataset.decode(h.seq) for h in hyps]
        for img_id, pred in zip(batch.img_ids, preds):
            self.test_outputs_dict[split].append((img_id, pred))
        self.test_gts_dict[split].extend(batch.token_ids)

        self.log(f'batch_inference_time_{split}', inference_time)
        return split, batch.img_ids, preds, inference_time

    def on_test_epoch_end(self) -> None:
        # Xử lý kết quả cho từng tập test
        for split in self.exprate_recorders.keys():
            # Tính ExpRate
            exprate = self.exprate_recorders[split].compute()
            self.log(f"test_ExpRate_{split}", exprate, prog_bar=True, logger=True, sync_dist=True)

            print(f"ExpRate for {split}: {exprate}")
            print(f"Length of total files in {split}: {len(self.test_outputs_dict[split])}")

            # Tính số lỗi ≤1, ≤2, ≤3
            le1, le2, le3 = 0, 0, 0
            total = len(self.test_outputs_dict[split])

            for (img_id, pred), gt_indices in zip(self.test_outputs_dict[split], self.test_gts_dict[split]):
                pred_tokens = pred.strip().split()
                gt_tokens = self.trainer.test_dataloaders[list(self.exprate_recorders.keys()).index(split)].dataset.decode(gt_indices).strip().split()
                dist = editdistance.eval(pred_tokens, gt_tokens)

                if dist <= 1:
                    le1 += 1
                if dist <= 2:
                    le2 += 1
                if dist <= 3:
                    le3 += 1

            self.log(f"test_≤1_error_{split}", le1 / total if total > 0 else 0, prog_bar=True, logger=True)
            self.log(f"test_≤2_error_{split}", le2 / total if total > 0 else 0, prog_bar=False, logger=True)
            self.log(f"test_≤3_error_{split}", le3 / total if total > 0 else 0, prog_bar=False, logger=True)

            # Ghi kết quả vào file ZIP riêng cho từng tập test
            with zipfile.ZipFile(f"result_{split}.zip", "w") as zip_f:
                for img_id, pred in self.test_outputs_dict[split]:
                    content = f"%{img_id}\n${pred}$".encode()
                    with zip_f.open(f"{img_id}.txt", "w") as f:
                        f.write(content)

    def approximate_joint_search(self, img: FloatTensor, mask: LongTensor) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}