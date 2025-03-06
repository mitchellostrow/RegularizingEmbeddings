import torch

from RegularizingEmbeddings.regularization.base import AbstractRegularization

class TanglingRegularization(AbstractRegularization):
    """
        TanglingRegularization

    Given a batch size of shape (batch_size, sequence_length, embedding_dim), for each sequence and for 
    each time point, we compute the tangling regularization (i.e. we estimate tangling)

    let x^s_t be the embedding of sequence s at time t, we compute:

        max_(s', t') ||dx^s_t / dx^s'_t'|| / (||x^s_t - x^s'_t'|| + epsilon)

    where epsilon is a small constant to avoid division by zero and dx is the finite difference time derivative.

    Let B,S,D be the batch size, sequence length, and embedding dimension respectively.
    The naive implementation would look something like this:

    for s in range(B):
        for t in range(S-1):
            x_s_t = x[s, t]
            dx_s_t = x[s, t+1] - x[s, t]
            for s' in range(B):
                for t' in range(S-1):
                    x_s'_t' = x[s', t']
                    dx_s'_t' = x[s', t'+1] - x[s', t']
                    Q_s_s'_t_t' = norm(dx_s_t / dx_s'_t') / (norm(x_s_t - x_s'_t') + epsilon)

    This is O(B^2 S^2) and we can do better.
    """

    def __init__(self, mode: str = 'efficient', epsilon: float = 1e-6, dt: float = 1e-6, normalize: bool = True):
        self.mode = mode
        self.epsilon = epsilon
        self.dt = dt
        self.normalize = normalize


    def __call__(self, x):
        if self.normalize:
            x = self.normalize_data(x)
            
        if self.mode == 'naive':
            return self.naive_implementation(x)
        elif self.mode == 'efficient':
            return self.efficient_implementation(x)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def normalize_data(self, x):
       B,S,D = x.shape
       # reshape x to (B*S, D)
       x = x.reshape(-1, D)
       
       x_min = x.min(dim=0)[0]  # Shape: (D,)
       x_max = x.max(dim=0)[0]  # Shape: (D,)
       x_range = x_max - x_min
       x_range[x_range == 0] = 1.0
       x = (x - x_min) / x_range
       x = x.reshape(B, S, D)

       return x


    def naive_implementation(self, x):
        dx = (x[:, 1:, :] - x[:, :-1, :]) / self.dt
        x = x[:, 1:, :]

        assert x.shape == dx.shape, f"x.shape: {x.shape}, dx.shape: {dx.shape}"

        B, S, D = x.shape
        
        Q = torch.full((B, S), -float('inf'))
        
        for s in range(S):
            for b in range(B):
                x_s_t = x[b, s, :]  # current position
                dx_s_t = dx[b, s, :]  # current velocity

                for s_prime in range(S):
                    for b_prime in range(B):
                        if s_prime == s and b_prime == b:
                            continue
                        x_s_prime_t = x[b_prime, s_prime, :]  # other position
                        dx_s_prime_t = dx[b_prime, s_prime, :]  # other velocity

                        num = torch.norm(dx_s_t - dx_s_prime_t)**2
                        den = torch.norm(x_s_t - x_s_prime_t)**2 + self.epsilon
                        Q[b, s] = max(Q[b, s], num/den)

       
        return Q

    def efficient_implementation(self, x):
        # Compute time derivatives
        dx = (x[:, 1:, :] - x[:, :-1, :]) / self.dt
        x = x[:, 1:, :]  # Use same positions as naive implementation
        
        B, S, D = x.shape
        
        # Reshape to (B*S, D) to treat each point independently
        x_flat = x.reshape(-1, D)    # Shape: (B*S, D)
        dx_flat = dx.reshape(-1, D)  # Shape: (B*S, D)
        
        # Compute all pairwise differences
        x_diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)   # Shape: (B*S, B*S, D)
        dx_diff = dx_flat.unsqueeze(1) - dx_flat.unsqueeze(0) # Shape: (B*S, B*S, D)
        
        # Compute norms (removed squared from here)
        x_norms = torch.norm(x_diff, dim=-1)   # Shape: (B*S, B*S)
        dx_norms = torch.norm(dx_diff, dim=-1) # Shape: (B*S, B*S)
        
        # Compute ratios (using non-squared norms)
        ratios = (dx_norms / (x_norms + self.epsilon))**2  # Shape: (B*S, B*S)
        
        # Get maximum ratio for each point (excluding self-comparisons)
        mask = torch.eye(B*S, device=x.device)
        ratios.masked_fill_(mask.bool(), -float('inf'))
        max_ratios = torch.max(ratios, dim=1)[0]  # Shape: (B*S,)
        
        # Reshape back to (B, S)
        Q = max_ratios.reshape(B, S)
        
        return Q
