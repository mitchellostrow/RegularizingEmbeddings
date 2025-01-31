import os
import hydra
import torch
import wandb
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

from RegularizingEmbeddings.lightning.model import SequenceModel
from RegularizingEmbeddings.lightning.data import SequenceDataModule
from RegularizingEmbeddings.data.data_generation import make_trajectories, postprocess_data, generate_train_and_test_sets

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


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train(config: DictConfig):
    """Main training routine.

    Args:
        config: Hydra configuration
    """
    # Set up W&B logging
    wandb_logger = WandbLogger(
        project=config.training.wandb.project,
        name=config.training.wandb.name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config.training.paths.output_dir, "checkpoints"),
            filename="model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.training.patience,
            mode="min",
        ),
        MetricsCallback(),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.training.max_epochs,
        accelerator='auto',
        devices='auto',
        logger=wandb_logger,
        callbacks=callbacks,
    )

    # Initialize data module
    torch.random.manual_seed(config.data.flow.random_state)
    eq, sol, dt = make_trajectories(config)
    values = postprocess_data(config, sol)

    config.model.input_dim = values.shape[-1]

    # Create train and test sets
    train_dataset, val_dataset, test_dataset, trajs = generate_train_and_test_sets(values, **config.data.train_test_params)

    # Initialize model
    model = SequenceModel(config)

    data_module = SequenceDataModule(
        datasets=(train_dataset, val_dataset, test_dataset),
        batch_size=config.training.data.batch_size,
        num_workers=config.training.data.num_workers,
        pin_memory=config.training.data.pin_memory,
    )
    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    train()
