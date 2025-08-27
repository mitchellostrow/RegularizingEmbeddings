import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from RegularizingEmbeddings.regularization.base import AbstractRegularization


class AmplificationRegularization(AbstractRegularization):
    """
    AmplificationRegularization computes noise amplification in delay embeddings.
    
    This regularization term measures how much noise gets amplified in the embedding space
    by comparing the variance of neighboring trajectories over time with the distance
    between neighbors in the embedding space.
    
    The regularization encourages embeddings where nearby points in embedding space
    remain close over time, reducing noise amplification.
    """
    
    def __init__(self, n_neighbors: int = 10, max_T: int = 5, normalize: bool = False, 
                 epsilon: float = 1e-8):
        """
        Initialize the amplification regularization.
        
        Args:
            n_neighbors: Number of nearest neighbors to consider
            max_T: Maximum number of time steps to look ahead
            normalize: Whether to normalize the amplification by the sum of 1/eps_k
            epsilon: Small constant to avoid division by zero
        """
        self.n_neighbors = n_neighbors
        self.max_T = max_T
        self.normalize = normalize
        self.epsilon = epsilon
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the amplification regularization loss.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            torch.Tensor: Scalar regularization loss
        """
        if torch.is_complex(x):
            x = torch.concatenate([torch.real(x), torch.imag(x)], dim=-1)
        
        batch_size, seq_len, embedding_dim = x.shape
        
        # Ensure we have enough time steps
        if seq_len <= self.max_T:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Flatten the embedding for neighbor computation (excluding last max_T steps)
        embedding_flat = x[:, :-self.max_T, :].reshape(-1, embedding_dim)
        
        # Find neighbors in the embedding space
        neighbor_distances, neighbor_indices = self._find_neighbors(embedding_flat, self.n_neighbors)
        
        # Compute future trajectory variance (E_k)
        E_k = self._compute_Ek(x, neighbor_indices, batch_size, seq_len)
        
        # Compute embedding space distances (eps_k)
        eps_k = self._compute_eps_k(embedding_flat, neighbor_indices)
        
        # Compute amplification
        amplification = torch.mean(E_k / (eps_k + self.epsilon))
        
        if self.normalize:
            norm_factor = torch.sum(1 / (eps_k + self.epsilon))
            amplification /= norm_factor
        
        return amplification
    
    def _find_neighbors(self, embedding, n_neighbors):
        """Find nearest neighbors in embedding space."""
        # Convert to numpy for sklearn
        embedding_np = embedding.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding_np)
        distances, indices = nbrs.kneighbors(embedding_np)
        
        # Convert back to torch tensors
        distances = torch.tensor(distances, device=embedding.device, dtype=embedding.dtype)
        indices = torch.tensor(indices, device=embedding.device, dtype=torch.long)
        
        return distances, indices
    
    def _compute_eps_k(self, embedding, neighbor_indices):
        """Compute average pairwise distances between neighbors."""
        embedding_neighbors = embedding[neighbor_indices]  # (n_points, n_neighbors, dim)
        
        # Compute all pairwise differences between neighbors
        diff = embedding_neighbors[:, :, None, :] - embedding_neighbors[:, None, :, :]
        squared_dists = torch.sum(diff ** 2, dim=-1)
        
        # Average over all pairwise distances
        sum_squared_dists = torch.sum(squared_dists, dim=(1, 2))
        n_neighbors = squared_dists.shape[1]
        mdists = sum_squared_dists / (n_neighbors * (n_neighbors - 1))
        
        return mdists
    
    def _compute_EkT(self, data_T, neighbor_indices):
        """Compute variance of neighbors at time T."""
        # Get neighbor trajectories at time T
        neighbor_data = data_T[neighbor_indices]  # (n_pts, n_neighbors, dim)
        
        # Compute mean of neighbors
        mu_kT = torch.mean(neighbor_data, dim=1)  # (n_pts, dim)
        
        # Expand for broadcasting
        mu_kT = mu_kT.unsqueeze(1).expand(-1, neighbor_indices.shape[-1], -1)
        
        # Compute variance
        variance = torch.mean((neighbor_data - mu_kT) ** 2, dim=1)  # (n_pts, dim)
        E_kT = torch.sum(variance, dim=-1)  # (n_pts,)
        
        return E_kT
    
    def _compute_Ek(self, x, neighbor_indices, batch_size, seq_len):
        """Compute E_k over multiple time steps."""
        # Prepare data for different time steps ahead
        embedding_dim = x.shape[-1]
        n_points = (seq_len - self.max_T) * batch_size
        
        E_k_list = []
        
        for T in range(1, self.max_T + 1):
            # Get data T steps ahead, flattened
            data_T = x[:, T:seq_len-self.max_T+T, :].reshape(-1, embedding_dim)
            
            # Only use points that have valid neighbor indices
            if data_T.shape[0] >= n_points:
                data_T = data_T[:n_points]
                E_kT = self._compute_EkT(data_T, neighbor_indices)
                E_k_list.append(E_kT)
        
        if not E_k_list:
            return torch.zeros(n_points, device=x.device, dtype=x.dtype)
        
        E_k = torch.stack(E_k_list, dim=0)  # (max_T, n_pts)
        return torch.mean(E_k, dim=0)  # Average over time steps
