import os
import hydra
import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

from src.lightning.model import SequenceModel
from src.lightning.data import SequenceDataModule


class MetricsCallback(Callback):
    """Custom callback for logging additional metrics to WandB."""
    
    def __init__(self):
        super().__init__()
        self.metrics = {}
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log arbitrary metrics to WandB.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for logging
        """
        wandb.log(metrics, step=step)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Example of logging at the end of each training epoch."""
        if hasattr(pl_module, 'get_metrics'):
            metrics = pl_module.get_metrics()
            self.log_metrics(metrics, step=trainer.current_epoch)


@hydra.main(config_path="../../config", config_name="train")
def train(config: DictConfig):
    """Main training routine.
    
    Args:
        config: Hydra configuration
    """
    # Set up W&B logging
    wandb_logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config.paths.output_dir, "checkpoints"),
            filename="model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.patience,
            mode="min",
        ),
        MetricsCallback(),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=callbacks,
    )

    # Initialize model
    model = SequenceModel(config)

    # Initialize data module
    # TODO add a way to generate thse datasets
    train_set, val_set, test_set = None, None, None
    data_module = hydra.utils.instantiate(config.data, datasets=(train_set, val_set, test_set))

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    train()
    