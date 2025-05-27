import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from comer.datamodule import LatexDataModule
from comer.lit_comer_swin import LitCoMER

# Kiểm tra file tồn tại
zipfile_path = "/kaggle/working/Comer/CoMER/data.zip"
ckp_path = "/kaggle/working/Comer/CoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4245.ckpt"
assert os.path.exists(zipfile_path), f"data.zip not found at {zipfile_path}"
assert os.path.exists(ckp_path), f"Checkpoint not found at {ckp_path}"

# Khởi tạo DataModule
dm = LatexDataModule(
    zipfile_path=zipfile_path,
    train_batch_size=6,
    eval_batch_size=4,
    num_workers=5,
)

# Khởi tạo Trainer với logger
logger = TensorBoardLogger("logs/", name="comer_test")
trainer = Trainer(logger=logger, devices=1)

if __name__ == "__main__":
    # Tải mô hình từ checkpoint
    model = LitCoMER.load_from_checkpoint(ckp_path, beam_size=8)
    
    # Chạy kiểm tra
    results = trainer.test(model, datamodule=dm)
    
    # In kết quả
    print("Test results:")
    for result in results:
        print(result)