import math
import os
import os.path as path
import warnings
from functools import partial
from typing import Tuple, Union, Dict

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.utils.data import DataLoader
from modules import utils as ut
from tqdm.auto import tqdm

from DMCE import utils, networks, functional


class DiffusionModel(nn.Module):
    def __init__(self,
                 model: networks.CNN,
                 *,
                 data_shape: Union[Tuple, list],
                 complex_data: bool = True,
                 loss_type: str = 'l2',
                 which_schedule: str = 'linear',
                 num_timesteps: int = 300,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.035,
                 loss_weighting: bool = False,
                 clipping: bool = False,
                 objective: str = 'pred_noise',
                 reverse_method: str = 'reverse_mean',
                 reverse_add_random: bool = False):
       
        super().__init__()

        self.device = model.device
        self.num_timesteps = num_timesteps
        self.data_shape = tuple(data_shape)

        # for complex data, we have to multiply the real and imaginary normal noise parts with 1/sqrt(2)
        self.noise_multiplier = 1 / (2 ** 0.5) if complex_data else 1.
        self.model = model
        self.loss_weighting = loss_weighting
        self.clipping = clipping
        self.reverse_add_random = reverse_add_random

        if objective not in ['pred_noise', 'pred_post_mean', 'pred_x_0']:
            raise ValueError(f'Objective \'{objective}\' is not supported.')
        self.objective = objective

        if reverse_method not in ['reverse_mean', 'ground_truth']:
            raise ValueError(f'Reverse process method \'{reverse_method}\' is not supported.')
        self.reverse_method = reverse_method

        if loss_type not in ['l1', 'l2']:
            raise ValueError(f'Invalid loss type \'{loss_type}\'.')
        self.loss_type = loss_type

        # calculate hyperparameter scheduling related parameters
        betas = self.get_beta_schedule(which_schedule, num_timesteps, beta_start, beta_end).to(self.device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        snrs = alphas_cumprod / (1. - alphas_cumprod)
        snrs_db = 10 * torch.log10(snrs)

        # register parameters for various diffusion model calculations

        def register_buffer(name, val):
            self.register_buffer(name, val.to(torch.float32))

        # alphas_cumprod_shifted[0] is manually set to 1 for convenience
        alphas_cumprod_shifted = torch.cat((torch.as_tensor([1], device=self.device), alphas_cumprod[:-1]))
        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_shifted', alphas_cumprod_shifted)
        register_buffer('snrs', snrs)
        register_buffer('snrs_db', snrs_db)

        # calculations for diffusion q(x_t | x_{t-1}) and q(x_t | x_0)

        register_buffer('sqrt_betas', torch.sqrt(betas))
        register_buffer('sqrt_one_minus_betas', torch.sqrt(1 - betas))
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0) and the different formulations of the posterior mean

        posterior_variance = betas * ((1. - alphas_cumprod_shifted) / (1. - alphas_cumprod))
        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_mean_coef_x_0', betas * torch.sqrt(alphas_cumprod_shifted) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef_x_t',
                        torch.sqrt(alphas) * (1. - alphas_cumprod_shifted) / (1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recip_minus1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        register_buffer('sqrt_alphas', torch.sqrt(alphas))
        register_buffer('sqrt_recip_alphas', torch.sqrt(1 / alphas))
        register_buffer('post_mean_from_noise_coef',
                        self.sqrt_recip_alphas * betas / self.sqrt_one_minus_alphas_cumprod)
        register_buffer('noise_from_post_mean_coef', self.sqrt_one_minus_alphas_cumprod / betas)

        # loss weights for the different loss functions
        loss_weight = 1
        if self.objective == 'pred_post_mean':
            loss_weight = 1 / posterior_variance
        elif self.objective == 'pred_noise':
            loss_weight = (betas ** 2) / (posterior_variance * alphas * (1. - alphas_cumprod))
        elif self.objective == 'pred_x_0':
            loss_weight = (alphas_cumprod_shifted * betas ** 2) / (posterior_variance * (1. - alphas_cumprod) ** 2)
        register_buffer('loss_weights', torch.clamp(loss_weight, 1e-5, 10))

        self.num_parameters = utils.count_params(self, only_trainable=True)

    @staticmethod
    def get_beta_schedule(which_schedule: str, num_timesteps: int, beta_start: float = 0.0001,
                          beta_end: float = 0.035) -> torch.Tensor:


        if which_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        elif which_schedule == 'quad':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float64) ** 2
        elif which_schedule == 'sqrt':
            betas = torch.linspace(beta_start ** 2, beta_end ** 2, num_timesteps, dtype=torch.float64) ** 0.5
        elif which_schedule == 'const':
            betas = beta_end * torch.ones(num_timesteps, dtype=torch.float64)
        elif which_schedule == 'recip':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = beta_end * (1. / torch.linspace(num_timesteps, 1, num_timesteps, dtype=torch.float64))
        elif which_schedule == 'log':
            betas = 2 ** torch.linspace(math.log2(beta_start), math.log2(beta_end), num_timesteps, dtype=torch.float64)
        elif which_schedule == 'exp':
            betas = torch.log2(torch.linspace(2 ** beta_start, 2 ** beta_end, num_timesteps, dtype=torch.float64))
        else:
            raise NotImplementedError(which_schedule)
        assert betas.shape == (num_timesteps,)
        return betas

    @torch.no_grad()
    def forward_step(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
 
        noise = utils.default(noise, lambda: self.noise_multiplier * torch.randn_like(x_t))
        assert noise.shape == x_t.shape
        return (
                utils.extract(self.sqrt_one_minus_betas, t, x_t.shape) * x_t +
                utils.extract(self.sqrt_betas, t, x_t.shape) * noise
        )

    def forward_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:

        noise = utils.default(noise, lambda: self.noise_multiplier * torch.randn_like(x_0))
        assert noise.shape == x_0.shape
        return (
                utils.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
                utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def get_posterior_mean_from_x_0(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:


        assert x_t.shape == x_0.shape
        return (
                utils.extract(self.posterior_mean_coef_x_0, t, x_0.shape) * x_0 +
                utils.extract(self.posterior_mean_coef_x_t, t, x_t.shape) * x_t
        )

    @torch.no_grad()
    def get_posterior_mean_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        assert x_t.shape == noise.shape
        return (
                utils.extract(self.sqrt_recip_alphas, t, x_t.shape) * x_t -
                utils.extract(self.post_mean_from_noise_coef, t, x_t.shape) * noise
        )

    @torch.no_grad()
    def get_noise_from_posterior_mean(self, x_t: torch.Tensor, t: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:

        assert x_t.shape == mu.shape
        return (
                utils.extract(self.noise_from_post_mean_coef, t, x_t.shape) *
                (x_t - utils.extract(self.sqrt_alphas, t, x_t.shape) * mu)
        )

    @torch.no_grad()
    def get_posterior_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        return utils.extract(self.posterior_variance, t, x_t.shape)

    @torch.no_grad()
    def predict_x_0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        assert x_t.shape == noise.shape
        return (
                utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                utils.extract(self.sqrt_recip_minus1_alphas_cumprod, t, noise.shape) * noise
        )

    @torch.no_grad()
    def predict_noise_from_x_0(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:

        assert x_t.shape == x_0.shape
        return (
                (x_t - utils.extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0) /
                utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def reverse_step(self, x_t: torch.Tensor, t: int, *, add_random: bool = False) -> torch.Tensor:

        b, *_ = x_t.shape
        assert utils.equal_iterables(x_t.shape[1:], self.data_shape)

        # construct a 1D Tensor with the current timestep in each entry
        batched_times = torch.full((b,), t, device=self.device, dtype=torch.long)

        if self.reverse_method == 'reverse_mean':
            # Reverse Mean Forwarding process
            if self.objective == 'pred_noise':
                pred_noise = self.model(x_t, batched_times)
                posterior_mean = self.get_posterior_mean_from_noise(x_t, batched_times, pred_noise)
            elif self.objective == 'pred_x_0':
                x_0 = self.model(x_t, batched_times)
                posterior_mean = self.get_posterior_mean_from_x_0(x_t, batched_times, x_0)
            elif self.objective == 'pred_post_mean':
                posterior_mean = self.model(x_t, batched_times)
            else:
                raise ValueError(f'Objective {self.objective} is not supported.')

            if add_random:
                # Reverse PDF Sampling process
                posterior_variance = self.get_posterior_variance(x_t, batched_times)
                noise = self.noise_multiplier * torch.randn_like(x_t) if t > 0 else 0.  # don't add noise in last step
                x_pred = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                # Reverse Mean Forwarding process
                x_pred = posterior_mean

        elif self.reverse_method == 'ground_truth':
            # Ground-Truth Prediction process
            if self.objective == 'pred_noise':
                pred_noise = self.model(x_t, batched_times)
                x_0 = self.predict_x_0_from_noise(x_t, batched_times, pred_noise)
            elif self.objective == 'pred_x_0':
                x_0 = self.model(x_t, batched_times)
                pred_noise = self.predict_noise_from_x_0(x_t, batched_times, x_0)
            elif self.objective == 'pred_post_mean':
                posterior_mean = self.model(x_t, batched_times)
                pred_noise = self.get_noise_from_posterior_mean(x_t, batched_times, posterior_mean)
                x_0 = self.predict_x_0_from_noise(x_t, batched_times, pred_noise)
            else:
                raise ValueError(f'Objective {self.objective} is not supported.')

            if add_random:
                # Ground-Truth Prediction with random forward sampling process
                noise = self.noise_multiplier * torch.randn_like(x_t)
            else:
                # Ground-Truth Prediction process
                noise = pred_noise
            x_pred = self.forward_sample(x_0, batched_times - 1, noise) if t > 0 else x_0  # no forward in last step
        else:
            raise ValueError(f'Reverse process method {self.reverse_method} is not supported.')
        return x_pred

    @torch.no_grad()
    def reverse_sample_loop(self, x_t: torch.Tensor, t_start: int,
                            *, return_all_timesteps: bool = False, add_random: bool = False) -> torch.Tensor:
 

        assert t_start <= self.num_timesteps
        assert utils.equal_iterables(x_t.shape[1:], self.data_shape)
        x_all = [x_t]
        for t in reversed(range(t_start)):
            x_t = self.reverse_step(x_t, t, add_random=add_random)
            if return_all_timesteps:
                x_all.append(x_t)

        # clip the final samples for image data to the range [-1, 1]
        if self.clipping:
            x_all = [torch.clamp(x, -1, 1) for x in x_all]
            x_t = torch.clamp(x_t, -1, 1)
        if return_all_timesteps:
            return torch.stack(x_all, dim=1)
        else:
            return x_t

    @torch.no_grad()
    def generate_new_samples(self, n_samples: int, *, noise: torch.Tensor = None, return_all_timesteps: bool = False,
                             add_random: bool = None) -> torch.Tensor:
        

        add_random = utils.default(add_random, self.reverse_add_random)
        x_t = utils.default(noise, lambda: self.noise_multiplier * torch.randn((n_samples, *self.data_shape),
                                                                               device=self.device))
        x_0 = self.reverse_sample_loop(x_t, self.num_timesteps, return_all_timesteps=return_all_timesteps,
                                       add_random=add_random)
        return x_0

    @torch.no_grad()
    def generate_estimate(self, y: torch.Tensor, snr: float, *, add_random: bool = None,
                          return_all_timesteps: bool = False) -> torch.Tensor:
        

        add_random = utils.default(add_random, self.reverse_add_random)

        # estimate t_hat, the time step that corresponds to the correct SNR
        t = int(torch.abs(self.snrs - snr).argmin())

        # normalize the input data accordingly (this might differ for other data than normalized channels)
        norm_multiplier = (snr / (1 + snr)) ** 0.5
        x_t = norm_multiplier * y

        x_hat = self.reverse_sample_loop(x_t, t, return_all_timesteps=return_all_timesteps, add_random=add_random)
        return x_hat

    @property
    def loss_fn(self):
    
        if self.loss_type == 'l1':
            return nn.functional.l1_loss
        elif self.loss_type == 'l2':
            return nn.functional.mse_loss
        else:
            raise ValueError(f'Invalid loss type \'{self.loss_type}\'.')

    def forward(self, x_0: torch.Tensor, noise: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
    
        # check input sizes, devices, and data types
        b, c, *_ = x_0.shape
        #assert c == self.model.ch_data
        assert x_0.device == self.device
        assert x_0.dtype == torch.float32
        if utils.exists(noise):
            assert x_0.shape == noise.shape
            assert noise.device == self.device
            assert noise.dtype == torch.float32
        if utils.exists(t):
            assert t.ndim == 1
            assert len(t) == b
            assert t.device == self.device
            assert t.dtype == torch.long

        # DM forward process
        t = utils.default(t, lambda: torch.randint(0, self.num_timesteps, (b,), device=self.device).long())
        noise = utils.default(noise, lambda: self.noise_multiplier * torch.randn_like(x_0))
        x_t = self.forward_sample(x_0, t, noise=noise)

        # perform model forward pass
        model_out = self.model(x_t, t)

        # Calculate the loss depending on the training strategy
        reduction = 'none' if self.loss_weighting else 'mean'
        if self.objective == 'pred_post_mean':
            posterior_mean = self.get_posterior_mean_from_x_0(x_t, t, x_0)
            loss = self.loss_fn(model_out, posterior_mean, reduction=reduction)
        elif self.objective == 'pred_noise':
            loss = self.loss_fn(model_out, noise, reduction=reduction)
        elif self.objective == 'pred_x_0':
            loss = self.loss_fn(model_out, x_0, reduction=reduction)
        else:
            raise ValueError(f'Objective {self.objective} is not supported.')

        # if the loss pre-factor is included, first scale each loss term and afterward take the mean
        if self.loss_weighting:
            loss = torch.mean(loss, dim=tuple(range(1, len(loss.shape))))
            loss = loss * utils.extract(self.loss_weights, t, loss.shape)
            loss = torch.mean(loss)
        return loss


class Trainer(object):
    def __init__(self,
                 model: DiffusionModel,
                 data_train: torch.Tensor,
                 data_val: torch.Tensor,
                 *,
                 batch_size: int = 128,
                 lr_init: float = 1e-3,
                 lr_step_multiplier: float = 0.5,
                 epochs_until_lr_step: int = 150,
                 num_epochs: int = 500,
                 val_every_n_batches: int = None,
                 mode: str = '1D',
                 track_fid_score: bool = False,
                 track_val_loss: bool = True,
                 track_mmd: bool = False,
                 use_fixed_gen_noise: bool = True,
                 save_mode: str = 'best',
                 dir_result: str = '../results',
                 use_ray: bool = False,
                 complex_data: bool = True,
                 num_min_epochs: int = 1,
                 num_epochs_no_improve: int = 1,
                 fft_pre: bool = False,
                 ):
      
        self.model = model
        self.device = model.device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_every_n_batches = val_every_n_batches
        self.mode = mode
        self.track_fid_score = track_fid_score
        self.track_val_loss = track_val_loss
        self.track_mmd = track_mmd
        self.use_fixed_gen_noise = use_fixed_gen_noise
        self.save_mode = save_mode
        self.dir_result = dir_result
        os.makedirs(self.dir_result, exist_ok=True)
        self.use_ray = use_ray
        self.epoch = 0
        self.checkpoint = 0
        self.complex_data = complex_data
        self.num_min_epochs = num_min_epochs
        self.num_min_epochs_no_improve = num_epochs_no_improve
        self.fft_pre = fft_pre

        # instantiate optimizer and lr scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=epochs_until_lr_step,
                                                            gamma=lr_step_multiplier, verbose=True)

        # training data preparation
        self.num_samples, *_ = data_train.shape
        if fft_pre:
            data_train = ut.complex_1d_fft(data_train, ifft=False, mode=mode)
        data_train = data_train.to(dtype=torch.float32)
        self.dataloader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.num_batches = len(self.dataloader)

        # validation data preparation
        self.num_val_samples, *_ = data_val.shape
        if track_val_loss:
            if fft_pre:
                data_val = ut.complex_1d_fft(data_val, ifft=False, mode=self.mode)
            self.data_val = data_val.to(dtype=torch.float32)
            self.noise_val = self.model.noise_multiplier * torch.randn_like(self.data_val)
            self.t_val = torch.randint(0, self.model.num_timesteps, (self.num_val_samples,)).long()


        self.num_gen_samples = 0
        self.num_fid_samples = 0
        if track_fid_score:
            if mode == '1D' and self.complex_data:
                self.num_fid_samples = min(1000, self.num_val_samples)
                feature_func = partial(utils.real2cmplx, dim=1, squeezed=True)
            elif mode == '1D' and not self.complex_data:
                self.num_fid_samples = min(1000, self.num_val_samples)
                feature_func = np.squeeze
            elif mode == '2D':
                self.num_fid_samples = min(100, self.num_val_samples)
                inception = InceptionV3(normalize_input=False, requires_grad=False)
                inception.to(device=self.device)
                inception.eval()
                feature_func = partial(functional.feature_func2d, inception=inception)
            else:
                raise ValueError(f'Data mode {self.mode} is not supported.')
            self.generation_metric = partial(functional.compute_fid_score, feature_func=feature_func)
            self.num_gen_samples = self.num_fid_samples

            # data used for FID calculation is a random subset of the validation data
            self.data_fid = utils.get_random_subset(self.data_val, num_samples=self.num_fid_samples)

        # prepare everything required for MMD calculation
        self.num_mmd_samples = 0
        if track_mmd:
            if mode == '1D':
                self.num_mmd_samples = min(2000, self.num_val_samples)
            elif mode == '2D':
                self.num_mmd_samples = min(2000, self.num_val_samples)
            else:
                raise ValueError(f'Data mode {self.mode} is not supported.')
            self.num_gen_samples = max(self.num_gen_samples, self.num_mmd_samples)

            # data used for MMD calculation is a random subset of the validation data
            self.data_mmd = utils.get_random_subset(self.data_val, num_samples=self.num_mmd_samples).cpu()

        # sample starting noise if new data samples should be generated from same noise in each validation iteration
        if (track_fid_score or track_mmd) and self.use_fixed_gen_noise:
            self.gen_noise = self.model.noise_multiplier * torch.randn_like(
                self.data_mmd if self.num_mmd_samples >= self.num_fid_samples else self.data_fid)
            self.gen_noise = self.gen_noise.to(dtype=torch.float32)

    def get_checkpoint_dict(self, **metrics: dict) -> dict:


        checkpoint_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'batch_size': self.batch_size,
            'mode': self.mode,
            'dir_result': self.dir_result,
            'checkpoint': self.checkpoint,
        }
        checkpoint_dict.update(**metrics)
        return checkpoint_dict

    def save_model(self, **metrics: dict):


        new_dict = self.get_checkpoint_dict(**metrics)

        # generate path to the new model file
        dir_model = path.join(self.dir_result, 'train_models')
        os.makedirs(dir_model, exist_ok=True)
        filepath = path.join(dir_model, f'model-{self.checkpoint}.pt')
        if self.save_mode == 'all':
            # all model files are stored (requires a lot of memory)
            torch.save(new_dict, filepath)

        elif self.save_mode == 'best':
            # only the model file with the best validation loss is stored
            old_files = os.listdir(dir_model)
            if not old_files:
                torch.save(new_dict, filepath)
            else:
                try:
                    save_new = True
                    for old_file in old_files:
                        old_dict = torch.load(path.join(dir_model, str(old_file)), map_location=self.device)
                        if new_dict['val_loss'] > old_dict['val_loss']:
                            save_new = False
                            break
                    if save_new:
                        torch.save(new_dict, filepath)
                        for old_file in old_files:
                            os.remove(path.join(dir_model, str(old_file)))
                except OSError as error:
                    warnings.warn(f'\n{error}\nFalling back to save_mode = \'all\'!')
                    self.save_mode = 'all'

        elif self.save_mode == 'newest':
            # always save the newest model
            old_files = os.listdir(dir_model)
            if not old_files:
                torch.save(new_dict, filepath)
            else:
                try:
                    torch.save(new_dict, filepath)
                    for file in old_files:
                        os.remove(path.join(dir_model, str(file)))
                except OSError as error:
                    warnings.warn(f'\n{error}\nFalling back to save_mode = \'all\'!')
                    self.save_mode = 'all'

        else:
            raise NotImplementedError(self.save_mode)

    def load_model(self, checkpoint: int = None, filepath: str = None):

        if not utils.exists(checkpoint) and not utils.exists(filepath):
            raise ValueError('Either checkpoint or filepath required for model loading')
        filepath = utils.default(filepath, path.join(self.dir_result, 'train_models', f'model-{checkpoint}.pt'))
        if not path.isfile(filepath):
            raise ValueError('Model file does not exist.')

        load_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(load_dict['model'])
        self.optimizer.load_state_dict(load_dict['optimizer'])
        self.lr_scheduler.load_state_dict(load_dict['lr_scheduler'])
        self.epoch = load_dict['epoch']
        self.batch_size = load_dict['batch_size']
        self.mode = load_dict['mode']
        self.dir_result = load_dict['dir_result']
        self.checkpoint = load_dict['checkpoint']
        self.model.to(device=self.device)

    def validate(self, loss: Union[float, torch.Tensor]) -> Tuple[float, float, float]:
        

        self.model.eval()
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            val_loss = loss
            if self.track_val_loss:
                val_loss = self.model(self.data_val.to(device=self.device),
                                      self.noise_val.to(device=self.device),
                                      self.t_val.to(device=self.device))
            val_loss = float(val_loss)

            if self.num_gen_samples != 0:

                data_sampled = []
                diff = self.num_gen_samples
                idx = 0
                while diff > 0:
                    n_samples = min(diff, 512)
                    if self.use_fixed_gen_noise:
                        gen_noise = self.gen_noise[idx:idx + n_samples].to(device=self.device)
                    else:
                        gen_noise = None
                    data_sampled.append(self.model.generate_new_samples(n_samples=n_samples, noise=gen_noise,
                                                                        return_all_timesteps=False))
                    diff -= n_samples
                    idx += n_samples
                data_sampled = torch.cat(data_sampled, dim=0)

            fid_score = None
            if self.track_fid_score:
                fid_score = self.generation_metric(self.data_fid.to(device=self.device),
                                                   data_sampled[:self.num_fid_samples])
                fid_score = float(fid_score)

            mmd = None
            if self.track_mmd:
                with utils.set_num_threads_context(num_threads=int(os.cpu_count() // 2)):
                    mmd = functional.calculate_mmd(self.data_mmd.cpu(), data_sampled[:self.num_mmd_samples].cpu())
                mmd = float(mmd)

            metrics = {}
            metrics.update({'val_loss': val_loss}) if self.track_val_loss else None
            metrics.update({'fid_score': fid_score}) if self.track_fid_score else None
            metrics.update({'mmd': mmd}) if self.track_mmd else None
            self.save_model(**metrics)
            self.print_validation_msg(**metrics)
        self.model.train()
        torch.cuda.empty_cache()
        return val_loss, fid_score, mmd

    def print_validation_msg(self, **metrics):

        msg = f'Epoch {self.epoch}/{self.num_epochs + 1}:'
        for key in metrics.keys():
            msg += f' {key} = {metrics[key]} |'
        msg = msg[:-1]
        print(msg)

    def train(self) -> Dict:


        curr_num_batches = 0

        # Initial validation
        fid_scores = []
        val_losses = []
        train_losses = []
        mmds = []
        loss = 0
        val_loss, fid_score, mmd = self.validate(loss)
        val_losses.append(val_loss)
        train_losses.append(val_loss)
        fid_scores.append(fid_score)
        mmds.append(mmd)
        self.checkpoint += 1
        self.epoch += 1
        early_stopping = Early_stopping(min_epochs=self.num_min_epochs,
                                        num_epochs_no_improve=self.num_min_epochs_no_improve)

        # Training loop
        while self.epoch <= self.num_epochs:
            train_losses_epochs = []
            # use a progress bar from tqdm package if ray-tune package is not used
            #for batch, data_batch in enumerate(tqdm(self.dataloader) if not self.use_ray else self.dataloader):
            for batch, data_batch in enumerate(self.dataloader if not self.use_ray else self.dataloader):
                loss = self.model(data_batch.to(device=self.device))
                train_losses_epochs.append(float(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                curr_num_batches += 1

                #if utils.exists(self.val_every_n_batches):
                #    if curr_num_batches % self.val_every_n_batches == 0:
            train_losses.append(np.mean(train_losses_epochs))
            # validation branch
            with torch.no_grad():
                val_loss, fid_score, mmd = self.validate(loss)
                val_losses.append(val_loss)
                fid_scores.append(fid_score)
                mmds.append(mmd)
                self.checkpoint += 1
            stopping = early_stopping(val_loss=val_loss, epoch=self.epoch)
            if stopping:
                print('Early stopping. End of training.')
                break

            self.lr_scheduler.step()
            self.epoch += 1


        # Construct the results dictionary
        result_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'fid_scores': fid_scores,
            'mmds': mmds,
            'snrs': self.model.snrs.tolist(),
            'snrs_db': self.model.snrs_db.tolist(),
            'num_trained_batches': curr_num_batches,
            'trained_epochs': self.epoch,
        }

        # emptying the cache might be unnecessary, but nvidia-smi memory data is more informative
        torch.cuda.empty_cache()
        return result_dict


class Tester(object):
    def __init__(self,
                 model: DiffusionModel,
                 data: torch.Tensor,
                 *,
                 batch_size: int = 512,
                 criteria: Union[list, Tuple] = None,
                 complex_data: bool = True,
                 return_all_timesteps: bool = False,
                 fft_pre: bool = False,
                 mode: str = '1D',
                 ):
   
        self.model = model
        self.device = self.model.device
        self.complex_data = complex_data
        self.return_all_timesteps = return_all_timesteps
        self.fft_pre = fft_pre
        self.mode = mode

        # prepare test data
        self.num_samples, *data_shape = data.shape
        assert utils.equal_iterables(data_shape, self.model.data_shape)
        if self.fft_pre:
            # Transform for the network input
            data = ut.complex_1d_fft(data, ifft=False, mode=self.mode)
        self.data = data.to(dtype=torch.float32)
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=False, pin_memory=True)
        if self.fft_pre:
            # Transform back for the MSE evaluation
            self.data = ut.complex_1d_fft(data, ifft=True, mode=self.mode)

        # register all test functions for the requested criteria
        self.criteria = criteria
        self.test_funcs = [self._register_test_func(criterion) for criterion in criteria]

    def _register_test_func(self, criterion: str) -> callable:
 
        if criterion == 'nmse':
            return self._test_nmse
        elif criterion == 'fid':
            return self._test_fid
        else:
            raise NotImplementedError(criterion)

    # TODO: implement testing routine for FID score and MMD score
    def _test_fid(self):
     
        raise NotImplementedError
        pass

    def _test_mmd(self):
       
        raise NotImplementedError
        pass

    @torch.no_grad()
    def _test_nmse(self) -> dict:
     
        # specify which SNRs should be evaluated
        snr_db_range = torch.arange(-10, 45, 5, dtype=torch.float32, device=self.device)
        #snr_db_range = torch.arange(20, 30, 5, dtype=torch.float32, device=self.device)
        snr_range = 10 ** (snr_db_range / 10)

        #nmse_per_sample_list = []
        nmse_total_power_list = []

        with torch.no_grad():
            for snr in tqdm(iterable=snr_range):
                # test each SNR value
                x_hat = []
                for data_batch in self.dataloader:
                    data_batch = data_batch.to(device=self.device)

                    # add noise to the test data
                    y = functional.awgn(data_batch, snr, multiplier=self.model.noise_multiplier)

                    # calculate channel estimate
                    x_est = self.model.generate_estimate(y.to(device=self.device), snr, return_all_timesteps=self.return_all_timesteps)
                    if self.fft_pre:
                        if self.return_all_timesteps:
                            x_est = ut.complex_1d_fft(x_est, ifft=True, mode=self.mode, _4d_array=True)
                        else:
                            x_est = ut.complex_1d_fft(x_est, ifft=True, mode=self.mode)
                    x_hat.append(x_est)
                x_hat = torch.cat(x_hat, dim=0).cpu()

                if self.return_all_timesteps:
                    nmse_total_power_list.append([])
                    n_timesteps = x_hat.shape[1]
                    if len(self.data.shape) == 5:
                        dim = int(self.data.shape[-1] * self.data.shape[-2])
                        x_hat = ut.reshape_fortran(x_hat, (-1, n_timesteps, dim))
                        for t in range(n_timesteps):
                            nmse_total_power_list[-1].append(functional.nmse_torch(ut.reshape_fortran(torch.squeeze(self.data),
                                                                  (-1, dim)), x_hat[:, t], norm_per_sample=False))
                    else:
                        for t in range(n_timesteps):
                            nmse_total_power_list[-1].append(
                                functional.nmse_torch(torch.squeeze(self.data), torch.squeeze(x_hat[:, t]), norm_per_sample=False))
                else:

                    if len(self.data.shape) == 4:
                        #print('Reshaping...')
                        dim = int(self.data.shape[-1] * self.data.shape[-2])
                        x_hat = ut.reshape_fortran(x_hat, (-1, dim))
                        nmse_total_power_list.append(functional.nmse_torch(ut.reshape_fortran(torch.squeeze(self.data), (-1, dim)), x_hat, norm_per_sample=False))
                    else:
                        # calculate NMSE from estimated channels
                        nmse_total_power_list.append(functional.nmse_torch(torch.squeeze(self.data), torch.squeeze(x_hat), norm_per_sample=False))

        return {'SNRs': snr_db_range.tolist(),
                'NMSEs_total_power': nmse_total_power_list,
                }

    @torch.no_grad()
    def test(self) -> dict:


        test_dict = {}
        self.model.eval()
        for criterion, test_func in zip(self.criteria, self.test_funcs):
            print(f'Testing criterion: \"{criterion}\"')
            test_dict[criterion] = test_func()

        return test_dict


class Early_stopping:
    def __init__(self, min_epochs: int=1, num_epochs_no_improve: int=1):
        self.min_epochs = min_epochs
        self.num_epochs_no_improve = num_epochs_no_improve
        self.best_val_loss = np.inf
        self.counter = num_epochs_no_improve

    def __call__(self, val_loss, epoch):
        if epoch > self.min_epochs:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = self.num_epochs_no_improve
            else:
                self.counter -= 1
        if self.counter < 1:
            return True
        else:
            return False

