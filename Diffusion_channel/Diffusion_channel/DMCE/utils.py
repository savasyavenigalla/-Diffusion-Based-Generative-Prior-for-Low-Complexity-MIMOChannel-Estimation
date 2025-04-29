import numpy as np
import scipy.linalg
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms

from typing import Union
from collections.abc import Collection
from matplotlib import pyplot as plt
from datetime import datetime
import os
import os.path as path
from scipy.io import loadmat
import json
import argparse


def exists(x) -> bool:
    """
    Checks, whether a variable is None
    """
    return x is not None


def default(val, d):
    """
    Return 'val' if it exists, else a default value or function is returned
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(x, *args, **kwargs):
    """
    callable that directly returns its parameter
    """
    return x


def get_timestamp() -> str:
    """
    Returns a timestamp in the format '%year%-%month%-%day%-%hour%h-%minute%m-$second%s'
    """
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%Hh%Mm%Ss')
    return timestamp


def rm_tree(pth: str):
    """
    Recursively removes all directories and files from the given 'pth', including the root directory itself
    """
    if path.isfile(pth):
        os.remove(pth)
    elif path.isdir(pth):
        contents = os.listdir(pth)
        for child in contents:
            rm_tree(path.join(pth, child))
        os.rmdir(pth)
    else:
        raise FileNotFoundError(f'{pth} is not a file or directory')


def parse_gpu_ids() -> list:
    """
    parses the command line argument "--gpu_ids" and returns a list of the requested integer GPU IDs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', '-g', nargs='+', required=False, type=int)
    args, _ = parser.parse_known_args()
    gpu_ids = args.gpu_ids
    return gpu_ids


def pretty_print_dict(d: dict):
    """
    prints a dictionary to the command line with an indentation of 4
    """
    pretty = json.dumps(d, indent=4)
    print(pretty)


def equal_iterables(it1: Collection, it2: Collection) -> bool:
    """
    checks, whether the contents of 'it1' and 'it2' are identical. Consider overloading the == operator for a certain
    data structure
    """
    assert len(it1) == len(it2)
    for first, second in zip(it1, it2):
        if not first == second:
            return False
    return True


class set_num_threads_context:

    def __init__(self, num_threads: int = 1):
        self.outer_threads = torch.get_num_threads()
        self.inner_threads = num_threads

    def __enter__(self):
        self.outer_threads = torch.get_num_threads()
        torch.set_num_threads(self.inner_threads)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_num_threads(self.outer_threads)


def load_dataset(bin_dir: str, which_dataset: str = 'MNIST', transformed=True) -> (torch.Tensor, int):
 
    if which_dataset == 'MNIST':
        dataset = torchvision.datasets.MNIST(root=bin_dir, download=True)
        data_size = 28
        num_channels = 1
        if transformed:
            dataset = transform_images_forward(dataset.data, num_channels=num_channels)
    elif which_dataset == 'FASHION_MNIST':
        dataset = torchvision.datasets.FashionMNIST(root=bin_dir, download=True)
        data_size = 32
        num_channels = 1
        if transformed:
            dataset = transform_images_forward(dataset.data, num_channels=num_channels)
    elif which_dataset == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=bin_dir, download=True)
        data_size = 32
        num_channels = 3
        if transformed:
            dataset = transform_images_forward(dataset.data, num_channels=num_channels)
    elif which_dataset == 'NOKIA':
        dataset = loadmat(file_name=path.join(bin_dir, 'nokia_measurements', 'total_measurement.mat'))['H_total']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'NOKIA_train':
        dataset = loadmat(file_name=path.join(bin_dir, 'nokia_measurements', 'train_measurement.mat'))['H_train']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'NOKIA_test':
        dataset = loadmat(file_name=path.join(bin_dir, 'nokia_measurements', 'test_measurement.mat'))['H_test']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=1':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=1_N=64', 'total_data.mat'))['H_total']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=1_train':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=1_N=64', 'train_data.mat'))['H_train']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=1_test':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=1_N=64', 'test_data.mat'))['H_test']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=1_cov':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=1_N=64', 'total_data.mat'))['t_total']
        if transformed:
            cov = np.asarray(dataset)
            dataset = np.zeros((cov.shape[0], cov.shape[1], cov.shape[1]), dtype=cov.dtype)
            for ind, t in enumerate(cov):
                dataset[ind] = scipy.linalg.toeplitz(t).conj()
        data_size = 64
    elif which_dataset == '3GPP_cluster=1_train_cov':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=1_N=64', 'train_data.mat'))['t_train']
        if transformed:
            cov = np.asarray(dataset)
            dataset = np.zeros((cov.shape[0], cov.shape[1], cov.shape[1]), dtype=cov.dtype)
            for ind, t in enumerate(cov):
                dataset[ind] = scipy.linalg.toeplitz(t).conj()
        data_size = 64
    elif which_dataset == '3GPP_cluster=1_test_cov':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=1_N=64', 'test_data.mat'))['t_test']
        if transformed:
            cov = np.asarray(dataset)
            dataset = np.zeros((cov.shape[0], cov.shape[1], cov.shape[1]), dtype=cov.dtype)
            for ind, t in enumerate(cov):
                dataset[ind] = scipy.linalg.toeplitz(t).conj()
        data_size = 64
    elif which_dataset == '3GPP_cluster=3':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=3_N=64', 'total_data.mat'))['H_total']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=3_train':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=3_N=64', 'train_data.mat'))['H_train']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=3_test':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=3_N=64', 'test_data.mat'))['H_test']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == '3GPP_cluster=3_cov':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=3_N=64', 'total_data.mat'))['t_total']
        if transformed:
            cov = np.asarray(dataset)
            dataset = np.zeros((cov.shape[0], cov.shape[1], cov.shape[1]), dtype=cov.dtype)
            for ind, t in enumerate(cov):
                dataset[ind] = scipy.linalg.toeplitz(t).conj()
        data_size = 64
    elif which_dataset == '3GPP_cluster=3_train_cov':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=3_N=64', 'train_data.mat'))['t_train']
        if transformed:
            cov = np.asarray(dataset)
            dataset = np.zeros((cov.shape[0], cov.shape[1], cov.shape[1]), dtype=cov.dtype)
            for ind, t in enumerate(cov):
                dataset[ind] = scipy.linalg.toeplitz(t).conj()
        data_size = 64
    elif which_dataset == '3GPP_cluster=3_test_cov':
        dataset = loadmat(file_name=path.join(bin_dir, '3GPP_SIMO_cluster=3_N=64', 'test_data.mat'))['t_test']
        if transformed:
            cov = np.asarray(dataset)
            dataset = np.zeros((cov.shape[0], cov.shape[1], cov.shape[1]), dtype=cov.dtype)
            for ind, t in enumerate(cov):
                dataset[ind] = scipy.linalg.toeplitz(t).conj()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_mixed':
        dataset = loadmat(file_name=path.join(bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa',
                                              'total_data.mat'))['H_total']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_mixed_train':
        dataset = loadmat(file_name=path.join(bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa',
                                              'train_data.mat'))['H_train']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_mixed_test':
        dataset = loadmat(file_name=path.join(bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa',
                                              'test_data.mat'))['H_test']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_LOS':
        dataset = loadmat(file_name=path.join(
            bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa_LOS', 'total_data.mat'))['H_total']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_LOS_train':
        dataset = loadmat(file_name=path.join(
            bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa_LOS', 'train_data.mat'))['H_train']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_LOS_test':
        dataset = loadmat(file_name=path.join(
            bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa_LOS', 'test_data.mat'))['H_test']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_NLOS':
        dataset = loadmat(file_name=path.join(
            bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa_NLOS', 'total_data.mat'))['H_total']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_NLOS_train':
        dataset = loadmat(file_name=path.join(
            bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa_NLOS', 'train_data.mat'))['H_train']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    elif which_dataset == 'QuaDRiGa_NLOS_test':
        dataset = loadmat(file_name=path.join(
            bin_dir, 'QuaDRiGa_SIMO_N=64_f=6GHz_bw=120kHz_scenario=3GPP_38901_UMa_NLOS', 'test_data.mat'))['H_test']
        if transformed:
            dataset = torch.from_numpy(np.asarray(dataset))
            dataset = cmplx2real(dataset, dim=1, new_dim=False).float()
        data_size = 64
    else:
        raise ValueError(f'Dataset {which_dataset} not available')
    return dataset, data_size


def get_random_subset(data: Union[torch.Tensor, Dataset], num_samples: int) -> torch.Tensor:


    subset, _ = random_split(data, [num_samples, len(data) - num_samples])
    dataloader = DataLoader(subset, batch_size=num_samples)
    subset_data = next(iter(dataloader))
    return subset_data


def transform_images_forward(data, num_channels: int = 1) -> torch.Tensor:

    data_transforms = [
        transforms.Lambda(lambda t: torch.as_tensor(t)),
        transforms.Lambda(lambda t: torch.permute(t, (0, 3, 1, 2)) if num_channels != 1 else t[:, None, :, :]),
        transforms.Lambda(lambda t: ((t / 255.) * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = torchvision.transforms.Compose(data_transforms)
    return data_transform(data)


def transform_images_reverse(data: torch.Tensor, num_channels: int = 1) -> np.ndarray:

    permute_list = list(range(len(data.shape)))
    if num_channels != 1:
        permute_list[-3] = -2
        permute_list[-2] = -1
        permute_list[-1] = -3
    permute_tuple = tuple(permute_list)

    data_transforms = [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: torch.permute(t, permute_tuple) if num_channels != 1 else torch.squeeze(t)),
        transforms.Lambda(lambda t: torch2np(t).astype(np.uint8)),
    ]
    data_transform = torchvision.transforms.Compose(data_transforms)
    return data_transform(data)


def torch2np(t: torch.Tensor) -> np.ndarray:

    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def np2torch(arr: np.ndarray) -> torch.Tensor:

    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(arr)


def real2cmplx(t: Union[torch.Tensor, np.ndarray], dim: int = 1,
               squeezed: bool = True) -> Union[torch.Tensor, np.ndarray]:
 

    if isinstance(t, np.ndarray):
        is_np = True
    elif isinstance(t, torch.Tensor):
        is_np = False
    else:
        raise NotImplementedError(f'Operation not implemented for class type {type(t)}.')
    n_dim = len(t.shape)
    assert t.shape[dim] == 2, f'Dimension {dim} must be of size 2'

    # create a permutation list, such that the first dimension contains the real and complex parts
    permutation = list(range(n_dim))
    permutation[0] = dim
    permutation[dim] = 0
    permutation = list(permutation)

    # permute and then add real and imaginary parts
    if is_np:
        t = t.transpose(permutation)
    else:
        t = torch.permute(t, permutation)
    t = t[0] + 1j * t[1]
    if squeezed:
        return t

    # Expand the resulting torch Tensor or numpy array
    if is_np:
        t = np.expand_dims(t, dim)
    else:
        t = torch.unsqueeze(t, dim)
    return t


def cmplx2real(t: Union[torch.Tensor, np.ndarray], dim: int = 1,
               new_dim: bool = True) -> Union[torch.Tensor, np.ndarray]:
    

    # for numpy Arrays
    if isinstance(t, np.ndarray):
        if not np.iscomplexobj(t):
            raise TypeError('Input array must be complex')
        t_real = np.real(t)
        t_imag = np.imag(t)
        if new_dim:
            t = np.stack([t_real, t_imag], axis=dim)
        else:
            t = np.concatenate([t_real, t_imag], axis=dim)

    # for torch Tensors
    elif isinstance(t, torch.Tensor):
        if not torch.is_complex(t):
            raise TypeError('Input array must be complex')
        t_real = torch.real(t)
        t_imag = torch.imag(t)
        if new_dim:
            t = torch.stack([t_real, t_imag], dim=dim)
        else:
            t = torch.cat([t_real, t_imag], dim=dim)
    else:
        raise NotImplementedError(f'Operation not implemented for class type {type(t)}.')
    assert t.shape[dim] == 2, f'Dimension {dim} must be of size 2 after conversion but has size {t.shape[dim]}'
    return t


def count_params(model: nn.Module, only_trainable: bool = True) -> int:

    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad) if only_trainable else
        sum(p.numel() for p in model.parameters())
    )


def extract(arr: torch.Tensor, indices: torch.Tensor, x_shape: Union[torch.Size, tuple, list]) -> torch.Tensor:

    batch_size, *_ = indices.shape
    assert x_shape[0] == batch_size
    out_arr = arr.gather(-1, indices)
    assert out_arr.shape == torch.Size([batch_size])
    return out_arr.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def get_device(device: str) -> torch.device:

    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError(f'Cuda is not available.')
        return torch.device('cuda:0')
    elif 'cuda:' in device:
        if not torch.cuda.is_available():
            raise ValueError(f'Cuda is not available.')
        device_id = int(device.split(':')[1])
        n_devices = torch.cuda.device_count()
        if device_id >= n_devices:
            raise ValueError(f'Cuda device {device} is not available.')
        return torch.device(device)
    elif device == 'cpu':
        return torch.device('cpu')
    else:
        raise ValueError(f'Pytorch Device {device} not available.')


def set_device(device: Union[str, torch.device]) -> torch.device:

    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return get_device(device)
    else:
        raise ValueError(f'device must be either a string or a torch.device instance, no {type(device)}.')



@torch.no_grad()





def dummy(inp):
    return inp