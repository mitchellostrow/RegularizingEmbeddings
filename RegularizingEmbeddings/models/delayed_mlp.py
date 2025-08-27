import torch
import torch.nn as nn
import numpy as np


def embed_signal_torch(data, n_delays, delay_interval=1):
    """
    Create a delay embedding from the provided tensor data.

    Parameters
    ----------
    data : torch.tensor
        The data from which to create the delay embedding. Must be either:
        a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    n_delays : int
        Parameter that controls the size of the delay embedding. Explicitly,
        the number of delays to include.

    delay_interval : int
        The number of time steps between each delay in the delay embedding. Defaults
        to 1 time step.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    device = data.device

    b, time, dim = data.shape
    if data.shape[int(data.ndim == 3)] - (n_delays - 1) * delay_interval < 1:
        raise ValueError(
            "The number of delays is too large for the number of time points in the data!"
        )

    # initialize the embedding
    embedding = torch.zeros((b, time, dim * n_delays)).to(device)

    for d in range(n_delays):

        ddelay = d * delay_interval

        ddata = d * dim
        delayed_data = data[:, : time - ddelay]

        embedding[:, ddelay:, ddata : ddata + data.shape[2]] = delayed_data

    return embedding

class MLP(nn.Module):
    def __init__(self, d_model, mlp_hidden, output_dim=None,dropout=0.0):
        super().__init__()
        if output_dim is None:
            output_dim = d_model
        self.c_fc = nn.Linear(d_model, mlp_hidden, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.c_proj = nn.Linear(mlp_hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x


class DelayedMLP(nn.Module):
    def __init__(self, input_dim, mlp_hidden, d_model, delay_interval):
        super().__init__()

        # self.ln = nn.LayerNorm(input_dim * n_delays)

        self.mlp = MLP(input_dim * d_model, mlp_hidden, input_dim)
        
        self.n_delays = d_model
        self.delay_interval = delay_interval

    def forward(self, x):
        # x should have shape (B, T, D)
        # first, reshape to have B,T-delay, D*delay
        # then add the first delays as well to the beginning of the sequence, concatenated with zeros
        with torch.no_grad():
            x = embed_signal_torch(x, self.n_delays, delay_interval=self.delay_interval)
        # x = self.ln(x)
        hidden = x.clone()

        x = self.mlp(x)
       
        return x, hidden
