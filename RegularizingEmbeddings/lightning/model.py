import pytorch_lightning as pl
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from RegularizingEmbeddings.models.lru import LRUMinimal
from RegularizingEmbeddings.models.mamba import MinimalMamba


class SequenceModel(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Args:
            model: Hydra config for the model
            optimizer: Hydra config for the optimizer
            scheduler: Optional Hydra config for the scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate model using Hydra
        self.model = hydra.utils.instantiate(config.model)
        self.config = config
        self.init_criterion()
        
    # ------------------------------ training setup ------------------------------ #
    def configure_optimizers(self) -> dict:
<<<<<<< HEAD:src/lightning/model.py
        self.optimizer = hydra.utils.instantiate(self.config.optimizer, params=self.model.parameters())
        self.scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer=self.optimizer)
=======
        self.optimizer = hydra.utils.instantiate(self.config.training.optimizer, params=self.model.parameters())
        self.scheduler = hydra.utils.instantiate(self.config.training.scheduler, optimizer=self.optimizer)
>>>>>>> ec4c5c534c930bef028bb7176ed8bb053fac76f9:RegularizingEmbeddings/lightning/model.py
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def init_criterion(self) -> None:
        self.loss_fn = hydra.utils.instantiate(self.config.training.criterion)


    # ------------------------------ forward pass ------------------------------ #

    def forward(self, x) -> torch.Tensor:
        out, _ = self.model(x)
        return out
    
    def _step(self, batch:tuple[torch.Tensor, torch.Tensor], stage:str) -> torch.Tensor:
        """
        Generic forward pass for a batch of data

        Args:
            batch: tuple of (x, y)
            stage: "train", "val", or "test"
        """
        # x, y = batch
        x = batch[..., :-1, :]
        y = batch[..., 1:, :]
        y_hat, _ = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "train")
        
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = batch
        y_hat, hidden_states = self(x)
        return y_hat, hidden_states
