import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch
from comer.datamodule import LatexDataModule
from comer.lit_comer_swin import LitCoMER

class LoggingCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get("val_loss", float("nan"))
        val_exp_rate = metrics.get("val_ExpRate", float("nan"))
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_ExpRate={val_exp_rate:.4f}")

def main():
    wandb.init(project="math-recognizer", name="comer-swin-")
    wandb_logger = WandbLogger(project="math-recognizer", log_model="all")

    dm = LatexDataModule(
        zipfile_path="/content/drive/MyDrive/Lab1/CoMER/data.zip",
        train_batch_size=16,
        eval_batch_size=4,
        num_workers=5,
    )
    dm.setup(stage="fit")
    vocab_size = dm.get_vocab_size()

    model = LitCoMER(
        d_model=768,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        dc=128,
        # Both must be True for the Attention Refinement Module to be built at
        # all - see _build_transformer_decoder in comer/model/decoder.py. With
        # both False the decoder is a plain Transformer and the coverage
        # mechanism that defines CoMER is never applied.
        cross_coverage=True,
        self_coverage=True,
        beam_size=8,
        max_len=200,
        alpha=0.6,
        early_stopping=False,  
        temperature=1.0,
        learning_rate=0.0001,
        patience=5,
        vocab_size=vocab_size,  
    )

    model.hparams.vocab_size = vocab_size
    model.comer_model.decoder.word_embed[0] = torch.nn.Embedding(vocab_size, model.hparams.d_model)
    model.comer_model.decoder.proj = torch.nn.Linear(model.hparams.d_model, vocab_size)

    wandb.config.update({
        "d_model": model.hparams.d_model,
        "nhead": model.hparams.nhead,
        "num_decoder_layers": model.hparams.num_decoder_layers,
        "dim_feedforward": model.hparams.dim_feedforward,
        "dropout": model.hparams.dropout,
        "learning_rate": model.hparams.learning_rate,
        "train_batch_size": dm.train_batch_size,
        "eval_batch_size": dm.eval_batch_size,
        "vocab_size": vocab_size,
        "beam_size": model.hparams.beam_size,
        "max_len": model.hparams.max_len,
        "alpha": model.hparams.alpha,
    })

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ExpRate",
        dirpath="checkpoints",
        filename="ComerSwin-{epoch:02d}-{val_ExpRate:.4f}",
        save_top_k=5,
        mode="max",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        verbose=True,
    )

    logging_callback = LoggingCallback()

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=200,
        max_steps=-1,
        logger=wandb_logger,
        default_root_dir='checkpoints',
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            logging_callback,
        ],
        fast_dev_run=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)

    #trainer.test(model, dm)

    wandb.finish()

if __name__ == "__main__":
    main()