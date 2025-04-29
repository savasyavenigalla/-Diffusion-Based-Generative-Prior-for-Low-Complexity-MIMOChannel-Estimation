import numpy as np
import torch
import h5py


def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))


def get_observation(h, snr, A=None):
    if A is None:
        return h + 10 ** (-snr / 20) * crandn(*h.shape)
    else:
        y = np.squeeze(np.matmul(A, np.expand_dims(h, 2)))
        if h.shape[1] == 1:
            y = np.expand_dims(y, 1)
        y += 10 ** (-snr / 20) * crandn(*y.shape)
        return y


def load_or_create_data(ch_type='3gpp', n_path=1, n_antennas_rx=64, n_antennas_tx=1, n_train_ch=100_000, n_val_ch=10_000, n_test_ch=10_000, return_toep=False):
    n_channels = n_train_ch + n_val_ch + n_test_ch
    if ch_type.startswith('quadriga'):
        print('Load channels...')
        if n_antennas_tx == 1:
            channels = h5py.File(
                f'bin/210000UEs_500m_1x{n_antennas_rx}BS_1x1MS_1carr_1symb_3kmh_diff=0_{ch_type}.mat', 'r')
        else:
            channels = h5py.File(
                f'bin/120000UEs_500m_1x{n_antennas_rx}BS_1x{n_antennas_tx}MS_1carr_1symb_diff=0_{ch_type}.mat', 'r')
        channels = channels['H_all']
        channels = np.array(channels)
        channels = np.transpose(channels['real'] + channels['imag'] * 1j)
        channels_train = channels[:n_train_ch]
        channels_test = channels[n_train_ch:n_train_ch + n_test_ch]
        channels_val = channels[n_train_ch + n_test_ch:n_train_ch + n_test_ch + n_val_ch]
        print('done.')
        if return_toep:
            return channels_train, None, channels_val, None, channels_test, None
        else:
            return channels_train, channels_val, channels_test
    else:
        if n_antennas_tx == 1:
            file_name_train = f'bin/{ch_type}_path={n_path}_dim={n_antennas_rx}_samp={n_train_ch}_train.npy'
            file_name_val = f'bin/{ch_type}_path={n_path}_dim={n_antennas_rx}_samp={n_val_ch}_val.npy'
            file_name_test = f'bin/{ch_type}_path={n_path}_dim={n_antennas_rx}_samp={n_test_ch}_test.npy'
            try:
                data_train, toep_train = np.load(file_name_train)
                data_val, toep_val = np.load(file_name_val)
                data_test, toep_test = np.load(file_name_test)
            except FileNotFoundError:
                print('Dataset not found.')
        else:
            file_name_train = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_train_ch}_train.npy'
            file_name_val = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_val_ch}_val.npy'
            file_name_test = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_test_ch}_test.npy'
            file_name_train_toeptx = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_train_ch}_train_toeptx.npy'
            file_name_val_toeptx = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_val_ch}_val_toeptx.npy'
            file_name_test_toeptx = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_test_ch}_test_toeptx.npy'
            file_name_train_toeprx = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_train_ch}_train_toeprx.npy'
            file_name_val_toeprx = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_val_ch}_val_toeprx.npy'
            file_name_test_toeprx = f'bin/{ch_type}_path={n_path}_dimrx={n_antennas_rx}_dimtx={n_antennas_tx}_samp={n_test_ch}_test_toeprx.npy'
            try:
                data_train = np.load(file_name_train)
                data_val = np.load(file_name_val)
                data_test = np.load(file_name_test)
                toep_train_rx = np.load(file_name_train_toeprx)
                toep_test_rx = np.load(file_name_test_toeprx)
                toep_val_rx = np.load(file_name_val_toeprx)
                toep_train_tx = np.load(file_name_train_toeptx)
                toep_test_tx = np.load(file_name_test_toeptx)
                toep_val_tx = np.load(file_name_val_toeptx)
                toep_train = (toep_train_rx, toep_train_tx)
                toep_test = (toep_test_rx, toep_test_tx)
                toep_val = (toep_val_rx, toep_val_tx)
            except FileNotFoundError:
                print('Dataset not found.')
        if return_toep:
            return data_train, toep_train, data_val, toep_val, data_test, toep_test
        else:
            return data_train, data_val, data_test


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def rand_exp(left: float, right: float, shape: tuple[int, ...]=(1,), seed=None):

    if left <= 0:
        raise ValueError('left needs to be positive but is {}'.format(left))
    if right <= left:
        raise ValueError(f'right needs to be larger than left but we have left: {left} and right: {right}')
    rng = np.random.default_rng(seed)
    return np.exp(np.log(left) + rng.random(*shape) * (np.log(right) - np.log(left)))



def toeplitz(c, r=None):
    
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()

    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n)).copy()


def as_strided(x, shape=None, strides=None, subok=False, writeable=True):

    x = np.array(x, copy=False, subok=subok)
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = np.asarray(DummyArray(interface, base=x))
    array.dtype = x.dtype

    view = _maybe_view_as_subclass(x, array)

    if view.flags.writeable and not writeable:
        view.flags.writeable = False

    return view

class DummyArray:

    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base

def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):

        new_array = new_array.view(type=type(original_array))

        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


def complex_1d_fft(input_tensor, ifft=False, _4d_array=False, mode='1D'):

    if _4d_array:
        complex_numbers = torch.view_as_complex(torch.stack([input_tensor[:, :, 0], input_tensor[:, :, 1]], dim=-1))
    else:
        complex_numbers = torch.view_as_complex(torch.stack([input_tensor[:, 0], input_tensor[:, 1]], dim=-1))

    if ifft:
        if mode == '1D':
            fft_result = torch.fft.ifft(complex_numbers, dim=-1, norm="ortho")
        else:
            fft_result = torch.fft.ifft2(complex_numbers, dim=(-2, -1), norm="ortho")
    else:
        if mode == '1D':
            fft_result = torch.fft.fft(complex_numbers, dim=-1, norm="ortho")
        else:
            fft_result = torch.fft.fft2(complex_numbers, dim=(-2, -1), norm="ortho")

    if _4d_array:
        fft_result = torch.stack([fft_result.real, fft_result.imag], dim=2)
    else:
        fft_result = torch.stack([fft_result.real, fft_result.imag], dim=1)

    return fft_result