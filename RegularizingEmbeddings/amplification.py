from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch

def find_neighbors(embedding, n_neighbors):
    #we need to implement a torch friendly version of knn
    dists = torch.cdist(embedding, embedding, p=2)
    distances , indices = torch.topk(dists, n_neighbors+1, largest=False)
    
    # nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding)
    # distances, indices = nbrs.kneighbors(embedding)
    # # indices = indices[:, 1:]  # exclude present point
    return distances, indices

def get_dists_bw_all_neighbors(embedding, indices):
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding)
    embedding_neighbors = embedding[indices]  #(n_points, n_neighbors, dim)
    diff = embedding_neighbors[:, :, np.newaxis, :] - embedding_neighbors[:, np.newaxis, :, :] #(n_points, n_neighbors, n_neighbors, dim)

    squared_dists = torch.sum(diff ** 2, dim=-1)

    sum_squared_dists = torch.sum(squared_dists, dim=(1, 2))

    # no missing two because already counted twice in distance matrix
    mdists = sum_squared_dists / (squared_dists.shape[1] * (squared_dists.shape[1] - 1))

    return mdists

def compute_eps_k(embedding, neighbor_indices):
    # if embedding.ndim == 3:
    #     embedding = embedding[
    #         :, :-thresh-1
    #     ]  # pick the data to go along with the E_k cutoff (can't look at the last T points because we need to look T steps ahead)
    #     embedding = embedding.reshape(-1, embedding.shape[-1])
    # else:
    #     embedding = embedding[:-thresh]

    # _, indices = find_neighbors(embedding, n_neighbors)

    eps_k = get_dists_bw_all_neighbors(embedding, neighbor_indices)
    
    return eps_k
    

def compute_EkT(data_T, neighbor_indices):

    mu_kT = torch.mean(
        data_T[neighbor_indices], dim=1 # average over neighbors
    ) # shape (n_pts x dim)
    
    mu_kT = mu_kT[:, None].repeat(1,neighbor_indices.shape[-1],1) # shape (n_pts x n_neighbors x dim)
    E_kT = torch.mean((data_T[neighbor_indices] - mu_kT) ** 2,dim=1)  # shape (n_pts x dim)
    E_kT = torch.sum(E_kT,dim=-1) # shape (n_pts)
    return E_kT


def compute_Ek(data_T_steps_ahead, embedding_flat, neighbor_indices):
    max_T = data_T_steps_ahead.shape[0]
    E_k = []
    for T in range(1, max_T + 1):
        E_kT = compute_EkT(data_T_steps_ahead[T-1], neighbor_indices)
        E_k.append(E_kT)
    E_k = torch.stack(E_k, dim = 0) # shape (max_T, n_pts)
    return E_k


def compute_noise_amp_k(
    data, embedding, n_neighbors, max_T, normalize=False
):  # noise_res=0.0,
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)

    # if isinstance(embedding, torch.Tensor):
    #     if embedding.device.type == "cuda":
    #         embedding = embedding.detach().cpu()

    if torch.is_complex(embedding):
        # hidden = np.real(embedding)
        embedding = torch.concatenate([torch.real(embedding), torch.imag(embedding)], dim=-1)
    # if noise_res >= 0:
    # embedding += np.random.uniform(-noise_res, noise_res, embedding.shape)

    # neighbor_distances, neighbor_indices = find_neighbors(embedding, n_neighbors)

    # embedding_0 = embedding[:, :-max_T]
    # neighbor_distances, neighbor_indices = find_neighbors(embedding_0, n_neighbors)

    data_T_steps_ahead = []
    for T in range(max_T):
        data_T_steps_ahead.append(data[..., T:-(max_T - T), :].reshape(-1, data.shape[-1]))
    data_T_steps_ahead = torch.stack(data_T_steps_ahead, dim=0)
    embedding_flat = embedding[..., :-max_T, :].reshape(-1, embedding.shape[-1])
    neighbor_distances, neighbor_indices = find_neighbors(embedding_flat, n_neighbors)

    E_k = compute_Ek(data_T_steps_ahead, embedding_flat, neighbor_indices) # shape (max_T, n_pts)

    eps_k = compute_eps_k(embedding_flat, neighbor_indices) # shape (n_pts)

    # import ipdb; ipdb.set_trace()
    sig = torch.mean(E_k / eps_k) # mean over (max_T, n_pts)

    if normalize:
        norm_factor = torch.sum(1/eps_k)
        sig /= norm_factor

    return sig, torch.mean(E_k), torch.mean(eps_k)

