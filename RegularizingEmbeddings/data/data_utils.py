import numpy as np
import scipy.signal as signal
from scipy.signal import argrelextrema
import torch
from tqdm.auto import tqdm


# ----------------------------------------
# Data Organization
# ----------------------------------------

def get_min_period_lengthscale(x, max_time=None, verbose=False):
    """
    Given a set of points, compute the minimum period lengthscale as defined in the methods of:
    Ahamed, T., Costa, A. C., & Stephens, G. J. (2020). Capturing the continuous complexity of behaviour in Caenorhabditis elegans. Nature Physics, 17(2), 275–283.
    
    Args:
        x (ndarray, torch.tensor): a set of points
    """
    if max_time is None:
        max_time = int(x.shape[-2]/6)
    
    num_rs = x.shape[-2] - max_time
    epsilon_vals = torch.zeros(num_rs, max_time).to(x.device)
    for t in tqdm(range(1, max_time + 1), desc='Computing Epsilon Function', disable=not verbose):
        if len(x.shape) == 2:
            epsilon_vals[:, t - 1] = torch.sort(torch.linalg.norm(x[:-t] - x[t:], axis=-1)).values[:num_rs]
        else:
            epsilon_vals[:, t - 1] = torch.sort(torch.linalg.norm(x[:, :-t] - x[:, t:], axis=-1)).values.mean(axis=0)[:num_rs]

    epsilon_mean = epsilon_vals.mean(axis=0).cpu().numpy()
    min_ind = argrelextrema(epsilon_mean, np.less)[0][0]
    return epsilon_mean[min_ind]
    # return epsilon_vals[0, min_ind]

def weighted_jacobian_lstsq(x, lengthscales, iterator=None, verbose=False):
    # lengthscales is a tensor of shape batch x time so lengthscales can vary by point
    seq_length = x.shape[-2]
    if len(x.shape) == 2:
        Js = torch.zeros(seq_length, x.shape[-1], x.shape[-1]).type(x.dtype).to(x.device)
    else:
        Js = torch.zeros(x.shape[0], seq_length, x.shape[-1], x.shape[-1]).type(x.dtype).to(x.device)

    iterator_passed = True
    if iterator is None:
        iterator = tqdm(total=seq_length, disable = not verbose, desc='Computing Weighted Jacobians')
        iterator_passed = False

    for i in range(seq_length):
        if len(x.shape) == 2:
            weighting = torch.exp(-torch.linalg.norm(x[i] - x, axis=-1)/lengthscales[i])
        else:
            weighting = torch.exp(-torch.linalg.norm(x[:, [i]] - x, axis=-1)/lengthscales[:, [i]])
        weighted_x = x*weighting.unsqueeze(-1)

        if len(weighted_x.shape) == 2:
            weighted_x_plus = weighted_x[1:]
            weighted_x_minus = weighted_x[:-1]
            Js[i] = torch.linalg.lstsq(weighted_x_minus, weighted_x_plus).solution.transpose(-2, -1)
        else:
            weighted_x_plus = weighted_x[:, 1:]
            weighted_x_minus = weighted_x[:, :-1]
            weighted_x_minus = torch.cat((weighted_x_minus, torch.ones(weighted_x_minus.shape[0], weighted_x_minus.shape[1]).unsqueeze(-1).to(x.device)), dim=-1)
            Js[:,i] = torch.linalg.lstsq(weighted_x_minus, weighted_x_plus).solution.transpose(-2, -1)[:, :, :-1]

        iterator.update()
    
    if not iterator_passed:
        iterator.close()

    return Js

def estimate_weighted_jacobians(x, max_time=None, sweep=False, thetas=None, return_losses=False, device='cpu', discrete=False, dt=None,verbose=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)

    if not sweep:
        eps = get_min_period_lengthscale(x, max_time=max_time, verbose=verbose)
        eps = torch.ones(x.shape[0], x.shape[1]).to(x.device)*eps
        Js = weighted_jacobian_lstsq(x, eps, verbose=verbose)
    else:
        pairwise_dists = torch.cdist(x, x)
        d_vals = pairwise_dists.mean(axis=-1)
        if thetas is None:
            # thetas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
            thetas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        iterator = tqdm(total=x.shape[-2]*len(thetas), disable = not verbose, desc='Computing Weighted Jacobians')
        losses = torch.zeros(len(thetas))
        for theta_ind, theta in enumerate(thetas):
            if theta > 0:
                lengthscales = d_vals/theta
            else: 
                lengthscales = torch.ones(d_vals.shape).to(x.device)*torch.inf
            Js = weighted_jacobian_lstsq(x, lengthscales, iterator=iterator, verbose=verbose)

            # get predictions
            preds = torch.zeros(x.shape).type(x.dtype).to(x.device)
            if len(x.shape) == 2:
                preds[:2] = x[:2]
            else:
                preds[:, :2] = x[:, :2]

            for t in range(preds.shape[-2] - 2):
                if len(x.shape) == 2:
                    preds[t + 2] = x[t + 1] + torch.matmul(Js[t], x[t + 1] - x[t])
                else:
                    preds[:, t + 2] = x[:, t + 1] + torch.matmul(Js[:, t], (x[:, t + 1] - x[:, t]).unsqueeze(-1)).squeeze(-1)

            losses[theta_ind] = torch.linalg.norm(preds - x).mean().cpu()
        iterator.close()
        theta = np.array(thetas)[torch.argmin(losses)]
        if theta > 0:
                lengthscales = d_vals/theta
        else: 
            lengthscales = torch.ones(d_vals.shape).to(x.device)*torch.inf
        Js = weighted_jacobian_lstsq(x, lengthscales, verbose=verbose)
    
    if not discrete:
        if dt is None:
            raise ValueError('dt must be provided for continuous data')
        Js = (Js - torch.eye(Js.shape[-1]).type(Js.dtype).to(Js.device))/dt

    if return_losses and sweep:
        return Js, losses
    else:
        return Js

def compute_lyaps_orig(Js, dt=1, k=None, worker_num=None, message_queue=None, verbose=False):
    T, n = Js.shape[0], Js.shape[1]
    old_Q = np.eye(n)
    if k is None:
        k = n
    old_Q = old_Q[:, :k]
    lexp = np.zeros(k)
    lexp_counts = np.zeros(k)
    for t in tqdm(range(T), disable=not verbose):
        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = np.linalg.qr(np.dot(Js[t], old_Q))
        # force diagonal of R to be positive
        # (if QR = A then also QLL'R = A with L' = L^-1)
        sign_diag = np.sign(np.diag(mat_R))
        sign_diag[np.where(sign_diag == 0)] = 1
        sign_diag = np.diag(sign_diag)
#         print(sign_diag)
        mat_Q = np.dot(mat_Q, sign_diag)
        mat_R = np.dot(sign_diag, mat_R)
        old_Q = mat_Q
        # successively build sum for Lyapunov exponents
        diag_R = np.diag(mat_R)

#         print(diag_R)
        # filter zeros in mat_R (would lead to -infs)
        idx = np.where(diag_R > 0)
        lexp_i = np.zeros(diag_R.shape, dtype="float32")
        lexp_i[idx] = np.log(diag_R[idx])
#         lexp_i[np.where(diag_R == 0)] = np.inf
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1

        if message_queue is not None:
            message_queue.put((worker_num, "task complete", "DEBUG"))
    
    return np.divide(lexp, lexp_counts)*(1/dt)

def compute_lyaps(Js, dt=1, k=None, verbose=False):
    squeeze = False
    if len(Js.shape) == 3:
        Js = Js.unsqueeze(0)
        squeeze = True

    T, n, _ = Js.shape[-3], Js.shape[-2], Js.shape[-1]
    old_Q = torch.eye(n, device=Js.device, dtype=Js.dtype)
    
    if k is None:
        k = n

    old_Q = old_Q[:, :k]
    lexp = torch.zeros(*Js.shape[:-3], k, device=Js.device, dtype=Js.dtype)
    lexp_counts = torch.zeros(*Js.shape[:-3], k, device=Js.device, dtype=Js.dtype)

    for t in tqdm(range(T), disable=not verbose):
            
        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = torch.linalg.qr(torch.matmul(Js[..., t, :, :], old_Q))
        
        # force diagonal of R to be positive
        # sign_diag = torch.sign(torch.diag(mat_R))
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)
        sign_diag = torch.sign(diag_R)
        sign_diag[sign_diag == 0] = 1
        sign_diag = torch.diag_embed(sign_diag)
        
        mat_Q = mat_Q @ sign_diag
        mat_R = sign_diag @ mat_R
        old_Q = mat_Q
        
        # Successively build sum for Lyapunov exponents
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)

        # Filter zeros in mat_R (would lead to -infs)
        idx = diag_R > 0
        lexp_i = torch.zeros_like(diag_R, dtype=Js.dtype, device=Js.device)
        lexp_i[idx] = torch.log(diag_R[idx])
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    if squeeze:
        lexp = lexp.squeeze(0)
        lexp_counts = lexp_counts.squeeze(0)
    
    return torch.flip(torch.sort((lexp / lexp_counts) * (1 / dt), axis=-1)[0], dims=[-1])

import scipy.signal as signal
def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    # return b, a
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def butter_highpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    # b, a = butter_highpass(cutoff, fs, order=order)
    # y = signal.filtfilt(b, a, data)
    sos = butter_highpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # return b, a
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    # b, a = butter_lowpass(cutoff, fs, order=order)
    # y = signal.filtfilt(b, a, data)
    sos = butter_lowpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

# Define the bandstop filter function
def butter_bandstop_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # b, a = signal.butter(order, [low, high], btype='bandstop')
    # y = signal.filtfilt(b, a, data)
    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # b, a = signal.butter(order, [low, high], btype='band')
    # return b, a
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    # b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = signal.lfilter(b, a, data)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def filter_data(data, low_pass=None, high_pass=None, dt=0.001, order=2, bidirectional=True):
    if low_pass is None and high_pass is None:
        return data
    elif low_pass is None and high_pass is not None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_highpass_filter(data[:, i], high_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    elif low_pass is not None and high_pass is None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_lowpass_filter(data[:, i], low_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    else:
        if low_pass == high_pass:
            return data
        elif low_pass > high_pass:
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandpass_filter(data[:, i], high_pass, low_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
        else: # low_pass < high_pass
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandstop_filter(data[:, i], low_pass, high_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
