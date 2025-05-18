import os
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from .transforms import CustomAugmentation
from .vocab import LatexVocab, vocab

class LatexDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        # Sử dụng thư mục "img" cho CROHME splits, "image" cho các splits khác
        self.image_dir = os.path.join(root_dir, "img" if split in ["2014", "2016", "2019"] else "image")
        self.caption_path = os.path.join(root_dir, "caption.txt")
        self.split = split
        self.transform = transform if transform else CustomAugmentation(is_train=(split == "train"))
        self.vocab = vocab

        # Xác định định dạng hình ảnh dựa trên split
        self.image_ext = ".bmp" if split in ["2014", "2016", "2019"] else ".png"

        self.samples = []
        with open(self.caption_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        invalid_lines = 0
        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                invalid_lines += 1
                print(f"Skipping line {line_num} in {self.caption_path}: Empty line")
                continue
            if '\t' not in line:
                invalid_lines += 1
                print(f"Skipping line {line_num} in {self.caption_path}: No tab separator - {line.strip()}")
                continue
            parts = line.strip().split('\t', 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                invalid_lines += 1
                print(f"Skipping line {line_num} in {self.caption_path}: Invalid format - {line.strip()}")
                continue
            img_id = parts[0].strip().split()[0]
            caption = parts[1].strip()
            img_path = os.path.join(self.image_dir, img_id + self.image_ext)
            if os.path.exists(img_path):
                try:
                    Image.open(img_path).verify()
                    self.samples.append((img_id, caption))
                except Exception as e:
                    invalid_lines += 1
                    print(f"Skipping line {line_num} in {self.caption_path}: Corrupted image {img_path} - {str(e)}")
            else:
                invalid_lines += 1
                print(f"Skipping line {line_num} in {self.caption_path}: Image not found {img_path}")

        print(f"Loaded {len(self.samples)} samples from {split} split")
        print(f"Skipped {invalid_lines} invalid lines in {self.caption_path}")

    def __len__(self):
        return len(self.samples)

    def tokenize_latex(self, caption: str) -> list:
        tokens = []
        i = 0
        while i < len(caption):
            char = caption[i]
            if char == '\\':
                j = i + 1
                while j < len(caption) and (caption[j].isalpha() or caption[j].isdigit()):
                    j += 1
                token = caption[i:j]
                tokens.append(token)
                i = j
            elif char.isspace():
                i += 1
            else:
                tokens.append(char)
                i += 1
        return tokens

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, list]:
        img_id, caption = self.samples[idx]
        img_name = img_id + self.image_ext
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("L", (256, 256), 255)

        image = self.transform(image)

        tokens = self.tokenize_latex(caption)
        token_ids = self.vocab.words2indices(tokens)

        return img_id, image, token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.vocab.indices2label(token_ids)

    def get_vocab_size(self):
        return len(self.vocab)