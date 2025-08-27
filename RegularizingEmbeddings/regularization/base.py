from abc import ABC, abstractmethod
import torch


class AbstractRegularization(ABC):
    """Abstract base class for regularization terms that can be applied to embedding sequences.
    
    This class defines the interface for regularization terms that operate on batches of
    sequential embedding data. Implementations should compute a scalar loss value based
    on the properties of the input embeddings.
    """
    
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the regularization loss on a batch of embedding sequences.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim) containing
               the embedding sequences to regularize.
               
        Returns:
            torch.Tensor: A scalar tensor containing the regularization loss value.
        """
        pass


    
class NoRegularization(AbstractRegularization):
    """Regularization term that does nothing."""
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0)

