from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch


def find_neighbors(embedding, n_neighbors, algorithm="ball_tree"):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)
    # indices = indices[:, 1:]  # exclude present point
    return distances, indices

def get_dists_bw_all_neighbors(embedding, indices):
    dists = []
    for inds in indices:
        dists.append(np.linalg.norm(embedding[inds] - embedding[inds][:, np.newaxis], axis=-1))
    
    dists = np.array(dists)
    mdists = np.sum(dists ** 2, axis=(1,2)) 
    mdists /= (dists.shape[1] * (dists.shape[1] - 1)) / 2
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
    mu_kT = np.mean(
        data_T[indices], axis=1
    )
    mu_kT = mu_kT[:, np.newaxis].repeat(n_neighbors, axis=1)
    E_kT = np.mean((data_T[indices] - mu_kT) ** 2,axis=1)  # shape
    E_kT = np.sum(E_kT,axis=-1)
    return E_kT


def compute_Ek(data, embedding, n_neighbors, max_T):
    E_k = []
    for T in range(1, max_T + 1):
        E_kT = compute_EkT(data, embedding, n_neighbors, T)[:-(max_T + 1 - T)]
        E_k.append(E_kT)
        print(E_kT.shape)
    E_k = np.array(E_k)
    E_k = np.mean(E_k, axis = 0)
    return E_k


def compute_noise_amp_k(
    data, embedding, n_neighbors, max_T, normalize=False
):  # noise_res=0.0,
    print("computing noise amp")
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if isinstance(embedding, torch.Tensor):
        if embedding.device.type == "cuda":
            embedding = embedding.detach().cpu().numpy()

    if np.iscomplex(embedding).any():
        # hidden = np.real(embedding)
        embedding = np.concatenate([np.real(embedding), np.imag(embedding)], axis=-1)
    # if noise_res >= 0:
    # embedding += np.random.uniform(-noise_res, noise_res, embedding.shape)
    E_k = compute_Ek(data, embedding, n_neighbors, max_T)

    eps_k = compute_eps_k(embedding, n_neighbors, max_T)

    sig = np.mean(E_k / eps_k)

    return sig, np.mean(E_k), np.mean(eps_k)
