"""
Train and test script for the DMCE.
"""
from DMCE import utils, DiffusionModel, Trainer, Tester, CNN
import os
import os.path as path
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DMCE.utils import cmplx2real

CUDA_DEFAULT_ID = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cpu', type=str)
    args = parser.parse_args()
    device = args.device
    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  

    n_dim = 64 # RX antennas
    n_dim2 = 16 # TX antennas
    num_train_samples = 100_000
    num_val_samples = 10_000  
    num_test_samples = 10_000
    seed = 453451

    return_all_timesteps = False
    fft_pre = True #  channel distribution in angular domain

   
    ch_type = '3gpp' # {quadriga_LOS}
    n_path = 3
    if n_dim2 > 1:
        mode = '2D'
    else:
        mode = '1D'
    complex_data = True

    data_train, data_val, data_test = ut.load_or_create_data(ch_type=ch_type, n_path=n_path, n_antennas_rx=n_dim,
                                     n_antennas_tx=n_dim2, n_train_ch=num_train_samples, n_val_ch=num_val_samples,
                                     n_test_ch=num_test_samples, return_toep=False)
    if ch_type.startswith('3gpp') and n_dim2 > 1:
        data_train = np.reshape(data_train, (-1, n_dim, n_dim2), 'F')
        data_test = np.reshape(data_test, (-1, n_dim, n_dim2), 'F')
        data_val = np.reshape(data_val, (-1, n_dim, n_dim2), 'F')
    data_train = torch.from_numpy(np.asarray(data_train[:, None, :]))
    data_train = cmplx2real(data_train, dim=1, new_dim=False).float()
    data_val = torch.from_numpy(np.asarray(data_val[:, None, :]))
    data_val = cmplx2real(data_val, dim=1, new_dim=False).float()
    data_test = torch.from_numpy(np.asarray(data_test[:, None, :]))
    data_test = cmplx2real(data_test, dim=1, new_dim=False).float()
    if ch_type.startswith('3gpp'):
        ch_type += f'_path={n_path}'

    # set data params
    cwd = os.getcwd()
    bin_dir = path.join(cwd, 'bin')
    data_shape = tuple(data_train.shape[1:])

    # data parameter dictionary, which is saved in 'sim_params.json'
    data_dict = {
        'bin_dir': str(bin_dir),
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'num_test_samples': num_test_samples,
        'train_dataset': ch_type,
        'test_dataset': ch_type,
        'n_antennas': n_dim,
        'mode': mode,
        'data_shape': data_shape,
        'complex_data': complex_data
    }

    # set Diffusion model params
    num_timesteps = 100 
    loss_type = 'l2'
    which_schedule = 'linear'

    max_snr_dB = 40
    beta_start = 1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))
    if num_timesteps == 5:
        beta_end = 0.95  
    elif num_timesteps == 10:
        beta_end = 0.7  
    elif num_timesteps == 50:
        beta_end = 0.2  
    elif num_timesteps == 100:
        beta_end = 0.1 
    elif num_timesteps == 300:
        beta_end = 0.035 
    elif num_timesteps == 500:
        beta_end = 0.02 
    elif num_timesteps == 1_000:
        beta_end = 0.01 
    elif num_timesteps == 10_000:
        beta_end = 0.001 
    else:
        beta_end = 0.035
    objective = 'pred_noise'  
    loss_weighting = False 
    clipping = False
    reverse_method = 'reverse_mean'  
    reverse_add_random = False  
   
    diff_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
        'loss_type': loss_type,
        'which_schedule': which_schedule,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'objective': objective,
        'loss_weighting': loss_weighting,
        'clipping': clipping,
        'reverse_method': reverse_method,
        'reverse_add_random': reverse_add_random
    }

    kernel_size = (3, 3)
    n_layers_pre = 2
    max_filter = 64
    ch_layers_pre = np.linspace(start=1, stop=max_filter, num=n_layers_pre+1, dtype=int)
    ch_layers_pre[0] = 2
    ch_layers_pre = tuple(ch_layers_pre)
    ch_layers_pre = tuple(int(x) for x in ch_layers_pre)
    n_layers_post = 3
    ch_layers_post = np.linspace(start=1, stop=max_filter, num=n_layers_post+1, dtype=int)
    ch_layers_post[0] = 2
    ch_layers_post = ch_layers_post[::-1]
    ch_layers_post = tuple(ch_layers_post)
    ch_layers_post = tuple(int(x) for x in ch_layers_post)
    n_layers_time = 1
    ch_init_time = 16
    batch_norm = False
    downsamp_fac = 1

    cnn_dict = {
        'data_shape': data_shape,
        'n_layers_pre': n_layers_pre,
        'n_layers_post': n_layers_post,
        'ch_layers_pre': ch_layers_pre,
        'ch_layers_post': ch_layers_post,
        'n_layers_time': n_layers_time,
        'ch_init_time': ch_init_time,
        'kernel_size': kernel_size,
        'mode': mode,
        'batch_norm': batch_norm,
        'downsamp_fac': downsamp_fac,
        'device': device,
    }
    batch_size = 128
    lr_init = 1e-4
    lr_step_multiplier = 1.0
    epochs_until_lr_step = 150
    num_epochs = 500
    val_every_n_batches = 2000
    num_min_epochs = 50
    num_epochs_no_improve = 20
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # newest, all
    dir_result = path.join(cwd, 'results')
    timestamp = utils.get_timestamp()
    dir_result = path.join(dir_result, timestamp)

    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
        'fft_pre': fft_pre,
    }


    batch_size_test = 512
    criteria = ['nmse']

    tester_dict = {
        'batch_size': batch_size_test,
        'criteria': criteria,
        'complex_data': complex_data,
        'return_all_timesteps': return_all_timesteps,
        'fft_pre': fft_pre,
        'mode': mode,
    }

    os.makedirs(dir_result, exist_ok=True)

    cnn = CNN(**cnn_dict)
    diffusion_model = DiffusionModel(cnn, **diff_model_dict)
    trainer = Trainer(diffusion_model, data_train, data_val, **trainer_dict)
    tester = Tester(diffusion_model, data_test, **tester_dict)

    # Training
    train_dict = trainer.train()
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_' \
                f'T={num_timesteps}_loss.png'
    plt.figure()
    plt.semilogy(range(1, len(train_dict['train_losses'])+1), train_dict['train_losses'], label='train-loss')
    plt.semilogy(range(1, len(train_dict['val_losses'])+1), train_dict['val_losses'], label='val-loss')
    plt.legend(['train-loss', 'val-loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(file_name)

    #Testing
    test_dict = tester.test()
    mse_list = list()
    mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
    mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
    mse_list[-1].insert(0, 'nmse_dm')
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_est/{date_time}_{ch_type}_dim={n_dim}x{n_dim2}_valdata={num_val_samples}_T={num_timesteps}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)

    utils.save_params(dir_result=dir_result, filename='test_results', params=test_dict)


if __name__ == '__main__':
    main()
