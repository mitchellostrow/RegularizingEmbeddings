from hydra.utils import instantiate
import numpy as np
import torch
from torch import utils
from tqdm.auto import tqdm

from .data_utils import filter_data

def make_trajectories(cfg, verbose=False):
    # Make trajectories

    eq, sol, dt = make_dysts_trajectories(cfg, verbose=verbose)

    return eq, sol, dt

def make_dysts_trajectories(cfg, verbose=False):
    # ----------------------------------------
    # MAKE TRAJECTORIES
    # ----------------------------------------
    eq = instantiate(cfg.data.flow)  # Ensure eq is instantiated here
    cfg.data.trajectory_params.verbose = verbose
    sol = eq.make_trajectory(**cfg.data.trajectory_params)
    dt = sol['dt']
    return eq, sol, dt

def postprocess_data(cfg, sol):
    # ----------------------------------------
    # POSTPROCESS
    # ----------------------------------------
    cfg.data.postprocessing.obs_noise = cfg.data.postprocessing.obs_noise * float(sol['values'].std())
    values = sol['values']
    if cfg.data.postprocessing.obs_noise > 0:
        if isinstance(values, torch.Tensor):
            values += torch.randn_like(values) * cfg.data.postprocessing.obs_noise
        else:
            values += np.random.normal(0, cfg.data.postprocessing.obs_noise, values.shape)
    if cfg.data.postprocessing.filter_data:
        values_filtered = np.zeros(values.shape)
        for traj_num in range(values.shape[0]):
            values_filtered[traj_num] = filter_data(values[traj_num], low_pass=cfg.data.postprocessing.low_pass, high_pass=cfg.data.postprocessing.high_pass, dt=sol['dt'])
        values = values_filtered
    if cfg.data.postprocessing.dims_to_observe != 'all':
        values = values[:, :, cfg.data.postprocessing.dims_to_observe]
    return values

def create_dataloaders(cfg, values, use_test=False, verbose=False):
    # ----------------------------------------
    # MAKE TRAIN AND TEST SETS
    # ----------------------------------------
    cfg.data.train_test_params.verbose = verbose
    train_dataset, val_dataset, test_dataset, trajs = generate_train_and_test_sets(values, **cfg.data.train_test_params)

    num_workers = cfg.training.data.num_workers
    persistent_workers = cfg.training.data.persistent_workers
    pin_memory = cfg.training.data.pin_memory
    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=cfg.training.data.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=cfg.training.data.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)
    
    if trajs['test_trajs'].sequence.shape[0] == 0 or trajs['test_trajs'].sequence.shape[1] == 0:
        test_dataloader = utils.data.DataLoader(trajs['val_trajs'], batch_size=trajs['val_trajs'].sequence.shape[0], shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)
    else:
        test_dataloader = utils.data.DataLoader(trajs['test_trajs'], batch_size=trajs['test_trajs'].sequence.shape[0], shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory)

    dataloader_dict = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader,
        'trajs': trajs
    }

    return train_dataloader, val_dataloader, test_dataloader, dataloader_dict

def embed_signal_torch(data, n_delays, delay_interval=1):
    """
    Create a delay embedding from the provided tensor data.

    Parameters
    ----------
    data : torch.tensor
        The data from which to create the delay embedding. Must be either: (1) a
        2-dimensional array/tensor of shape T x N where T is the number
        of time points and N is the number of observed dimensions
        at each time point, or (2) a 3-dimensional array/tensor of shape
        K x T x N where K is the number of "trials" and T and N are
        as defined above.

    n_delays : int
        Parameter that controls the size of the delay embedding. Explicitly,
        the number of delays to include.

    delay_interval : int
        The number of time steps between each delay in the delay embedding. Defaults
        to 1 time step.
    """
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        device = data.device

        # initialize the embedding
        if data.ndim == 3:
            embedding = torch.zeros((data.shape[0], data.shape[1] - (n_delays - 1)*delay_interval, data.shape[2]*n_delays)).to(device)
        else:
            embedding = torch.zeros((data.shape[0] - (n_delays - 1)*delay_interval, data.shape[1]*n_delays)).to(device)
        
        for d in range(n_delays):
            index = (n_delays - 1 - d)*delay_interval
            ddelay = d*delay_interval

            if data.ndim == 3:
                ddata = d*data.shape[2]
                embedding[:,:, ddata: ddata + data.shape[2]] = data[:,index:data.shape[1] - ddelay]
            else:
                ddata = d*data.shape[1]
                embedding[:, ddata:ddata + data.shape[1]] = data[index:data.shape[0] - ddelay]
        
        return embedding

def convert_to_trajs_needed(pct):
    if pct == 0:
        return 0
    else:
        return 1/pct
    
def get_start_indices(seq_length, seq_spacing, T):
    if T == 0:
        return []

    if seq_length > T:
            raise ValueError(f'seq_length ({seq_length}) must be less than or equal to the number of time points ({T})')
    if seq_length == T:
        start_indices = [0]
    else:
        if seq_spacing == 0:
            raise ValueError('seq_spacing must be greater than 0 if seq_length != pts.shape[1]')
        start_indices = np.arange(0, T - seq_length, seq_spacing)
    
    return start_indices

    
# Dataset class for time series prediction
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        return self.sequence[index]

def generate_train_and_test_sets(pts, seq_length, seq_spacing=1, train_percent=0.8, test_percent=0.05, split_by='random', dtype='torch.FloatTensor', delay_embedding_params=None, verbose=False):
    val_percent = 1 - train_percent - test_percent

    if train_percent + test_percent >= 1:
        raise ValueError('train_percent + test_percent must be less than 1')

    if split_by == 'trajectory':
        # select start indices
        start_indices = get_start_indices(seq_length, seq_spacing, pts.shape[1])

        if convert_to_trajs_needed(train_percent) > pts.shape[0]:
            raise ValueError(f'With split_by==trajectory, not enough trajectories ({pts.shape[0]}) to satisfy train_percent ({train_percent:.4f})')
        if convert_to_trajs_needed(test_percent) > pts.shape[0]:
            raise ValueError(f'With split_by==trajectory, not enough trajectories ({pts.shape[0]}) to satisfy test_percent ({test_percent:.4f})')
        if convert_to_trajs_needed(val_percent) > pts.shape[0]:
            raise ValueError(f'With split_by==trajectory, not enough trajectories ({pts.shape[0]}) to satisfy val_percent ({val_percent:.4f})')

        train_inds = np.random.choice(pts.shape[0], int(train_percent*pts.shape[0]), replace=False)
        remaining_inds = np.array([i for i in np.arange(pts.shape[0]) if i not in train_inds])
        test_inds = np.random.choice(remaining_inds, int(test_percent*pts.shape[0]), replace=False)
        val_inds = np.array([i for i in np.arange(pts.shape[0]) if i not in train_inds and i not in test_inds])

        train_trajs = pts[train_inds]
        val_trajs = pts[val_inds]
        test_trajs = pts[test_inds]

        # generate training examples and labels
        n_train = train_trajs.shape[0]
        n_val = val_trajs.shape[0]
        n_test = test_trajs.shape[0]

        train_examples = np.zeros((n_train*len(start_indices), seq_length, train_trajs.shape[2]))
        val_examples = np.zeros((n_val*len(start_indices), seq_length, val_trajs.shape[2]))
        test_examples = np.zeros((n_test*len(start_indices), seq_length, test_trajs.shape[2]))

        for i, start_ind in tqdm(enumerate(start_indices), total=len(start_indices), disable=not verbose, desc='Sequence Indices'):
            train_examples[i*n_train:(i + 1)*n_train] = train_trajs[:, start_ind:start_ind + seq_length]
            val_examples[i*n_val:(i + 1)*n_val] = val_trajs[:, start_ind:start_ind + seq_length]
            test_examples[i*n_test:(i + 1)*n_test] = test_trajs[:, start_ind:start_ind + seq_length]

    # elif split_by == 'random':
    #     all_examples = np.zeros((pts.shape[0]*len(start_indices), seq_length, pts.shape[2]))
    #     for i, start_ind in tqdm(enumerate(start_indices), total=len(start_indices), disable=not verbose, desc='Sequence Indices'):
    #         all_examples[i*pts.shape[0]:(i + 1)*pts.shape[0]] = pts[:, start_ind:start_ind + seq_length]
        
    #     train_inds = np.random.choice(all_examples.shape[0], int(train_percent*all_examples.shape[0]), replace=False)
    #     remaining_inds = np.array([i for i in np.arange(all_examples.shape[0]) if i not in train_inds])
    #     test_inds = np.random.choice(remaining_inds, int(test_percent*all_examples.shape[0]), replace=False)
    #     val_inds = np.array([i for i in np.arange(all_examples.shape[0]) if i not in train_inds and i not in test_inds])

    #     train_examples = all_examples[train_inds]
    #     val_examples = all_examples[val_inds]
    #     test_examples = all_examples[test_inds]
    elif split_by == 'time':
        
        train_trajs = pts[:, np.arange(0, int(train_percent*pts.shape[1]))]
        val_trajs = pts[:, np.arange(int(train_percent*pts.shape[1]), int((train_percent + val_percent)*pts.shape[1]))]
        test_trajs = pts[:, np.arange(int((train_percent + val_percent)*pts.shape[1]), pts.shape[1])]

        # generate examples
        start_indices_train = get_start_indices(seq_length, seq_spacing, train_trajs.shape[1])
        start_indices_val = get_start_indices(seq_length, seq_spacing, val_trajs.shape[1])
        start_indices_test = get_start_indices(seq_length, seq_spacing, test_trajs.shape[1])

        n_trajs = train_trajs.shape[0]

        train_examples = np.zeros((n_trajs*len(start_indices_train), seq_length, train_trajs.shape[2]))
        val_examples = np.zeros((n_trajs*len(start_indices_val), seq_length, val_trajs.shape[2]))
        test_examples = np.zeros((n_trajs*len(start_indices_test), seq_length, test_trajs.shape[2]))

        iterator = tqdm(total=len(start_indices_train) + len(start_indices_val) + len(start_indices_test), disable=not verbose, desc='Sequence Indices')

        for i, start_ind in enumerate(start_indices_train):
            train_examples[i*n_trajs:(i + 1)*n_trajs] = train_trajs[:, start_ind:start_ind + seq_length]
            iterator.update()
        
        for i, start_ind in enumerate(start_indices_val):
            val_examples[i*n_trajs:(i + 1)*n_trajs] = val_trajs[:, start_ind:start_ind + seq_length]
            iterator.update()
        
        for i, start_ind in enumerate(start_indices_test):
            test_examples[i*n_trajs:(i + 1)*n_trajs] = test_trajs[:, start_ind:start_ind + seq_length]
            iterator.update()
        
        iterator.close()

    train_dataset = TimeSeriesDataset(torch.from_numpy(train_examples).type(dtype))
    val_dataset = TimeSeriesDataset(torch.from_numpy(val_examples).type(dtype))
    test_dataset = TimeSeriesDataset(torch.from_numpy(test_examples).type(dtype))

    if isinstance(train_trajs, np.ndarray):
        train_trajs = torch.from_numpy(train_trajs).type(dtype)
    if isinstance(val_trajs, np.ndarray):
        val_trajs = torch.from_numpy(val_trajs).type(dtype)
    if isinstance(test_trajs, np.ndarray):
        test_trajs = torch.from_numpy(test_trajs).type(dtype)

    if delay_embedding_params is not None:
        if delay_embedding_params['observed_indices'] != 'all':
            train_trajs = train_trajs[:, :, delay_embedding_params['observed_indices']]
            val_trajs = val_trajs[:, :, delay_embedding_params['observed_indices']]
            test_trajs = test_trajs[:, :, delay_embedding_params['observed_indices']]
        if delay_embedding_params['n_delays'] > 1:
            train_trajs = embed_signal_torch(train_trajs, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])
            val_trajs = embed_signal_torch(val_trajs, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])
            test_trajs = embed_signal_torch(test_trajs, delay_embedding_params['n_delays'], delay_embedding_params['delay_spacing'])

    trajs = dict(
        train_trajs=TimeSeriesDataset(train_trajs),
        val_trajs=TimeSeriesDataset(val_trajs),
        test_trajs=TimeSeriesDataset(test_trajs)
    )

    # train_dataset = TimeSeriesDataset(torch.from_numpy(train_examples).type(dtype), torch.from_numpy(train_labels))
    # test_dataset = TimeSeriesDataset(torch.from_numpy(test_examples).type(dtype), torch.from_numpy(test_labels))

    if verbose:
        print(f"Train dataset shape: {train_dataset.sequence.shape}")
        print(f"Validation dataset shape: {val_dataset.sequence.shape}")
        print(f"Test dataset shape: {test_dataset.sequence.shape}")

        print('Train trajectories dataset shape: {}'.format(trajs['train_trajs'].sequence.shape))
        print('Validation trajectories dataset shape: {}'.format(trajs['val_trajs'].sequence.shape))
        print('Test trajectories dataset shape: {}'.format(trajs['test_trajs'].sequence.shape))

    return train_dataset, val_dataset, test_dataset, trajs