
from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

test_year = "2014"
ckp_path = "checkpoints/epoch=197-step=69300-val_ExpRate=0.4477.ckpt" 

if __name__ == "__main__":
    trainer = Trainer(logger=False, devices=1)

    dm = CROHMEDatamodule(test_year=test_year)

    model = LitBTTR.load_from_checkpoint(ckp_path, beam_size=5)

    results = trainer.test(model, datamodule=dm)
    