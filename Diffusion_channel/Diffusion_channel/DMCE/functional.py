from DMCE import utils

import torch
import torch.nn as nn
import numpy as np
import scipy.linalg
from typing import Tuple


def awgn(x: torch.Tensor, snr: float, multiplier: float = 1.) -> torch.Tensor:

    return x + (1 / snr ** 0.5) * multiplier * torch.randn_like(x)


def compute_covariance(x: np.ndarray, var_dim: int = 1) -> np.ndarray:

    assert x.ndim == 2
    if var_dim == 0:
        x = x.T
    (n_samples, n_features) = x.shape

    cov = np.zeros((n_features, n_features), dtype=x.dtype)
    for i in range(n_samples):
        cov += np.expand_dims(x[i], 1) @ np.expand_dims(x[i].conj(), 0)
    cov /= n_samples
    return cov


@torch.no_grad()
def compute_feature_statistics(x: torch.Tensor, feature_func: callable = None,
                               requires_np=False) -> Tuple[np.ndarray, np.ndarray]:

    if utils.exists(feature_func):
        if requires_np:
            features = feature_func(utils.torch2np(x))
        else:
            features = utils.torch2np(feature_func(x))
    else:
        features = utils.torch2np(x)

    #print(features.ndim)
    assert features.ndim == 2
    mu = np.mean(features, axis=0)
    sigma = np.cov(features.T)
    return mu, sigma


@torch.no_grad()
def compute_fid_score(x_real: torch.Tensor, x_generated: torch.Tensor,
                      feature_func: callable = None, requires_np=False) -> float:

    mu1, sigma1 = compute_feature_statistics(x_real, feature_func, requires_np)
    mu2, sigma2 = compute_feature_statistics(x_generated, feature_func, requires_np)

    fid_score = compute_frechet_distance(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
    return float(fid_score)


@torch.no_grad()
def feature_func2d(x: torch.Tensor, inception: nn.Module) -> torch.Tensor:

    data_shape = x.shape
    if data_shape[1] == 1:
        # expand the channel dimension to 3 if x is a gray-scaled image.
        shape_expanded = list(data_shape)
        shape_expanded[1] = 3
        x = x.expand(*shape_expanded)
    features = inception(x)
    features = torch.squeeze(features[0])
    return features


def compute_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Real and generated mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Real and generated covariances have different dimensions'

    diff = mu1 - mu2

    tmp = sigma1 @ sigma2
    tmp, _ = scipy.linalg.sqrtm(tmp, disp=False)
    tmp = np.real(tmp)
    d = np.linalg.norm(diff) ** 2 + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(tmp)
    return float(np.real(d))


@torch.no_grad()
def nmse_torch(x: torch.Tensor, x_hat: torch.Tensor, norm_per_sample: bool = False) -> float:


    assert x.shape == x_hat.shape

    # assumes that first dimension contains the different data samples
    norm_dim = tuple(range(1, x.ndim))
    numerator = torch.linalg.vector_norm(x - x_hat, dim=norm_dim) ** 2
    #x_cplx = x[:, 0] + 1j * x[:,1]
    #x_hat_cplx = x_hat[:, 0] + 1j * x_hat[:,1]
    #test = torch.linalg.vector_norm(x_cplx - x_hat_cplx, dim=1) ** 2
    assert numerator.ndim == 1

    denominator = torch.linalg.vector_norm(x, dim=norm_dim) ** 2
    assert denominator.ndim == 1
    if not norm_per_sample:
        denominator = torch.mean(denominator)

    return float(torch.mean(numerator / denominator))


def nmse_np(x: np.ndarray, x_hat: np.ndarray, norm_per_sample: bool = False) -> float:

    assert x.shape == x_hat.shape
    assert x.ndim == 2
    numerator = np.linalg.norm(x - x_hat, axis=1) ** 2
    assert numerator.ndim == 1

    denominator = np.linalg.norm(x, axis=1) ** 2
    assert denominator.ndim == 1
    if not norm_per_sample:
        denominator = np.mean(denominator)

    return float(np.mean(numerator / denominator))


@torch.no_grad()
def calculate_power_torch(x: torch.Tensor) -> float:
    norm_dim = tuple(range(1, x.ndim))
    return float(torch.mean(torch.linalg.vector_norm(x, dim=norm_dim) ** 2))


@torch.no_grad()
def pairwise_dist2(x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
 
    x_norm = torch.linalg.vector_norm(x, dim=1) ** 2
    if utils.exists(y):
        y_norm = torch.linalg.vector_norm(y, dim=1) ** 2
    else:
        y = x
        y_norm = x_norm
    dist = x_norm[:, None] + y_norm[None, :] - 2.0 * torch.real(x @ y.T.conj())

    # Account for numerical instabilities
    dist[dist < 0] = 0
    return dist


@torch.no_grad()
def rbf_kernel(x: torch.Tensor, n_kernels: int = 5, mul_factor: float = 2.) -> torch.Tensor:

    assert len(x.shape) == 2
    n_samples = x.shape[0]

    # get pairwise distances
    l2_distances = pairwise_dist2(x)

    # computes the variances of the different kernels depending on the pairwise distances
    bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
    bandwidths = bandwidth_multipliers * (l2_distances.sum() / (n_samples ** 2 - n_samples))

    # apply Gaussian kernel function for different variances indicated by bandwidths
    return torch.exp(-l2_distances[None, ...] / bandwidths[..., None, None]).sum(0)


@torch.no_grad()
def calculate_mmd(x: torch.Tensor, y: torch.Tensor, real2cmplx: bool = False, **kwargs) -> float:


    assert x.ndim == y.ndim
    assert utils.equal_iterables(x.shape[1:], y.shape[1:])
    x_size = x.shape[0]
    y_size = y.shape[0]

    if real2cmplx:
        x = utils.real2cmplx(x, squeezed=True)
        y = utils.real2cmplx(y, squeezed=True)

    # Reshape the input tensors such that they are 2-dimensional
    shape_expected = x.shape[0], int(np.prod(x.shape[1:]))
    x = torch.reshape(x, shape=shape_expected)
    shape_expected = y.shape[0], int(np.prod(y.shape[1:]))
    y = torch.reshape(y, shape=shape_expected)

    # stack both input tensors in the first dimension. This enables to compare data sets of different sizes
    x_y_stacked = torch.vstack([x, y])

    kernels = rbf_kernel(x_y_stacked, **kwargs)

    # extract the different kernels from the overall kernel matrix
    kernel_x = kernels[:x_size, :x_size]
    kernel_y = kernels[x_size:, x_size:]
    kernel_mixed1 = kernels[x_size:, :x_size]
    kernel_mixed2 = kernels[:x_size, x_size:]

    # derive unbiased sample means of all kernels
    kernel_x_term = (torch.sum(kernel_x) - torch.sum(torch.diag(kernel_x))) / (x_size * (x_size - 1))
    kernel_y_term = (torch.sum(kernel_y) - torch.sum(torch.diag(kernel_y))) / (y_size * (y_size - 1))
    if x_size == y_size:
        # Simpler version of unbiased MMD^2 estimator
        kernel_mixed1_term = (torch.sum(kernel_mixed1) - torch.sum(torch.diag(kernel_mixed1))) / (x_size * (y_size - 1))
        kernel_mixed2_term = (torch.sum(kernel_mixed2) - torch.sum(torch.diag(kernel_mixed2))) / (y_size * (x_size - 1))
    else:
        kernel_mixed1_term = kernel_mixed1.mean()
        kernel_mixed2_term = kernel_mixed2.mean()

    # calculate MMD
    return float(kernel_x_term + kernel_y_term - kernel_mixed1_term - kernel_mixed2_term)
