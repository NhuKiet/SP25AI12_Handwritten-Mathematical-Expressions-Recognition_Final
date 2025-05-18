import os
from typing import List, Optional
import zipfile
from dataclasses import dataclass
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from .dataset import LatexDataset
from .vocab import vocab

@dataclass
class Batch:
    img_ids: List[str]
    images: torch.Tensor  # [B, 1, H, W]
    mask: torch.Tensor  # [B, H, W]
    token_ids: List[List[int]]  # [B, L]

    def __len__(self) -> int:
        return len(self.img_ids)

    def to(self, device) -> "Batch":
        return Batch(
            img_ids=self.img_ids,
            images=self.images.to(device),
            mask=self.mask.to(device),
            token_ids=self.token_ids,
        )

def collate_fn(batch):
    img_ids, images, token_ids = zip(*batch)

    max_height = max(img.size(1) for img in images)
    max_width = max(img.size(2) for img in images)
    n_samples = len(images)

    padded_images = torch.zeros(n_samples, 3, max_height, max_width)
    mask = torch.ones(n_samples, max_height, max_width, dtype=torch.bool)

    for idx, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        padded_images[idx, :, :h, :w] = img
        mask[idx, :h, :w] = 0

    return Batch(
        img_ids=list(img_ids),
        images=padded_images,
        mask=mask,
        token_ids=list(token_ids),
    )

class LatexDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = "data.zip",
        train_batch_size: int = 16,
        eval_batch_size: int = 4,
        num_workers: int = 5,
    ) -> None:
        super().__init__()
        self.zipfile_path = zipfile_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        self.temp_dir = "temp_data"
        os.makedirs(self.temp_dir, exist_ok=True)
        with zipfile.ZipFile(self.zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

        if stage == "fit" or stage is None:
            self.train_dataset = LatexDataset(
                root_dir=os.path.join(self.temp_dir, "data", "train"),
                split="train",
            )
            self.val_dataset = LatexDataset(
                root_dir=os.path.join(self.temp_dir, "data", "val"),
                split="val",
            )

        if stage == "test" or stage is None:
            # Chỉ tải các tập test của CROHME
            self.test_datasets = {}
            test_splits = ["2014", "2016", "2019"]  # Bỏ "test"
            for split in test_splits:
                test_dir = os.path.join(self.temp_dir, "data", split)
                if os.path.exists(test_dir):
                    self.test_datasets[split] = LatexDataset(
                        root_dir=test_dir,
                        split=split,
                    )
                else:
                    print(f"Test split {split} not found in {self.zipfile_path}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        # Trả về danh sách các DataLoader cho từng tập test
        dataloaders = []
        for split, dataset in self.test_datasets.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=collate_fn,
                )
            )
        return dataloaders

    def teardown(self, stage: Optional[str] = None) -> None:
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def get_vocab_size(self):
        return self.train_dataset.get_vocab_size()