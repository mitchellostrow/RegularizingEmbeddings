import pytorch_lightning as pl
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from RegularizingEmbeddings.models.lru import LRUMinimal
from RegularizingEmbeddings.models.mamba import MinimalMamba

from RegularizingEmbeddings.amplification import compute_noise_amp_k
from RegularizingEmbeddings.metrics import predict_hidden_dims
from RegularizingEmbeddings.tangling import TanglingRegularization
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

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
        self.hidden_states_buffer = []
        self.y_hat_buffer = []
        self.data_buffer = []
        self.init_criterion()
        
    # ------------------------------ training setup ------------------------------ #
    def configure_optimizers(self) -> dict:
        self.optimizer = hydra.utils.instantiate(self.config.training.optimizer, params=self.model.parameters())
        self.scheduler = hydra.utils.instantiate(self.config.training.scheduler, optimizer=self.optimizer)
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

        y_hat, z_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        if stage == "val":
            # Save hidden states for custom metric computation.
            self.hidden_states_buffer.append(z_hat.detach())
            self.data_buffer.append(batch)

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

    def on_validation_epoch_end(self):
        if self.hidden_states_buffer:
            hidden_states = torch.cat(self.hidden_states_buffer, dim=0)
            data = torch.cat(self.data_buffer, dim=0)
        #if hidden_states is complex, then we need to concatenate the real and imaginary parts
        if hidden_states.is_complex():
            hidden_states = torch.cat([hidden_states.real, hidden_states.imag], dim=-1)
        #compute noise amplification
        if "noise_amplification" in self.config.evals.metrics:
            noise_amp, E_k, eps_k = compute_noise_amp_k(data, hidden_states, 
                self.config.evals.noise_amplification.n_neighbors,
                self.config.evals.noise_amplification.max_T,
                self.config.evals.noise_amplification.normalize)
            self.log("noise_amplification (sigma)", noise_amp, on_epoch=True, prog_bar=True)
            self.log("conditional variance (E_k)", E_k, on_epoch=True, prog_bar=True) #conditional var
            self.log("embedding volume (eps_k)", eps_k, on_epoch=True, prog_bar=True)

        self.hidden_states_buffer = []
        self.data_buffer = []

        #TODO: debug
        # if "tangling" in self.config.evals.metrics:
        #     tangling = TanglingRegularization(**self.config.evals.tangling)
        #     Q = tangling(hidden_states).mean()
        #     self.log("tangling", Q, on_epoch=True, prog_bar=True)

        #TODO: for these metrics, we need to have the full data, not just the partially observed. Right now, data only has the partially observed
        # if "predict_hidden_dims_lm" in self.config.evals.metrics:
        #     train_score, test_score, classifier = predict_hidden_dims(
        #             data,
        #             hidden_states,
        #             self.config.data.postprocessing.dims_to_observe,
        #             model=ElasticNet,
        #             **self.config.evals.predict_hiddens.linear_model_kwargs,
        #             )
        #     self.log("predict_hidden_dims_lm_R2", test_score, on_epoch=True, prog_bar=True)

        # if "predict_hidden_dims_mlp" in self.config.evals.metrics:
        #     train_score, test_score, classifier = predict_hidden_dims(
        #             data,
        #             hidden_states,
        #             self.config.data.postprocessing.dims_to_observe,
        #             model=MLPRegressor,
        #             **self.config.evals.predict_hiddens.mlp_kwargs,
        #             )
        #     self.log("predict_hidden_dims_mlp_R2", test_score, on_epoch=True, prog_bar=True)
        #compute participation ratio of hidden dimensions
    
