from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch


def find_neighbors(embedding, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    # indices = indices[:, 1:]  # exclude present point
    return distances, indices

def get_dists_bw_all_neighbors(embedding, indices):
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding)
    embedding_neighbors = embedding[indices]  #(n_points, n_neighbors, dim)
    diff = embedding_neighbors[:, :, np.newaxis, :] - embedding_neighbors[:, np.newaxis, :, :] #(n_points, n_neighbors, n_neighbors, dim)

    squared_dists = torch.sum(diff ** 2, dim=-1)

    sum_squared_dists = torch.sum(squared_dists, dim=(1, 2))

    mdists = sum_squared_dists / (squared_dists.shape[1] * (squared_dists.shape[1] - 1))

    return mdists

def compute_eps_k(embedding, n_neighbors, thresh=10):
    if embedding.ndim == 3:
        embedding = embedding[
            :, :-thresh-1
        ]  # pick the data to go along with the E_k cutoff (can't look at the last T points because we need to look T steps ahead)
        embedding = embedding.reshape(-1, embedding.shape[-1])
    else:
        embedding = embedding[:-thresh]

    _, indices = find_neighbors(embedding, n_neighbors)

    eps_k = get_dists_bw_all_neighbors(embedding, indices)
    
    return eps_k
    

def compute_EkT(data, embedding, n_neighbors, T):
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding)
    # data should be 1 dimensional in the last axis
    if data.ndim == 3:
        #given the indices, we need to find the index of the point T steps after the point of a given index
        data_T = data[:,T:].reshape(-1, data.shape[-1])
        embedding_0 = embedding[:, :-T].reshape(-1, embedding.shape[-1])
    else:
        embedding_0 = embedding[:-T]
        data_T = data[T:]

    _, indices = find_neighbors(embedding_0, n_neighbors)
    #compute the variance over these points in the future
    mu_kT = torch.mean(
        data_T[indices], dim=1
    )
    mu_kT = mu_kT[:, None].repeat(1,n_neighbors,1)
    E_kT = torch.mean((data_T[indices] - mu_kT) ** 2,dim=1)  # shape
    E_kT = torch.sum(E_kT,dim=-1)
    return E_kT


def compute_Ek(data, embedding, n_neighbors, max_T):
    E_k = []
    for T in range(1, max_T + 1):
        E_kT = compute_EkT(data, embedding, n_neighbors, T)[:-(max_T + 1 - T)]
        E_k.append(E_kT)
    E_k = torch.stack(E_k, dim = 0)
    return E_k


def compute_noise_amp_k(
    data, embedding, n_neighbors, max_T, normalize=False
):  # noise_res=0.0,
    print("computing noise amp")
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    if isinstance(embedding, torch.Tensor):
        if embedding.device.type == "cuda":
            embedding = embedding.detach().cpu()

    if torch.is_complex(embedding):
        # hidden = np.real(embedding)
        embedding = torch.concatenate([torch.real(embedding), torch.imag(embedding)], dim=-1)
    # if noise_res >= 0:
    # embedding += np.random.uniform(-noise_res, noise_res, embedding.shape)
    E_k = compute_Ek(data, embedding, n_neighbors, max_T)

    eps_k = compute_eps_k(embedding, n_neighbors, max_T)

    sig = torch.mean(E_k / eps_k)

    if normalize:
        norm_factor = torch.sum(1/eps_k)
        sig /= norm_factor

    return sig, torch.mean(E_k), torch.mean(eps_k)

