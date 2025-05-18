from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Define the datamodule
dm = CROHMEDatamodule()

# Define the model
model = LitBTTR(
    d_model=64,
    growth_rate=32,
    num_layers=3,
    nhead=4,
    num_decoder_layers=3,
    dim_feedforward=256,
    dropout=0.1,
    beam_size=5,
    max_len=4,
    alpha=0.9,
    learning_rate=1e-3,
    patience=5,
)

# Define the logger
logger = TensorBoardLogger("logs", name="bttr")


# Define the trainer
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="bttr-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
        ,
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ],
    fast_dev_run=False,
    check_val_every_n_epoch=2,

)

# Train the model
trainer.fit(model, dm, ckpt_path="/teamspace/studios/this_studio/BTTR/checkpoints/epoch=197-step=69300-val_ExpRate=0.4477.ckpt")
