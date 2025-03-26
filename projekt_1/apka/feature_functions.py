import numpy as np
import scipy.signal as signal

def acf(data):
    ACF = signal.fftconvolve(data, data, mode='full')
    ACF = ACF[len(ACF)//2:]
    return ACF

def calculate_zcr(data):
    length = len(data)
    crosses = 0
    for i in range(1, length):
        if data[i - 1] * data[i] < 0:
            crosses += 1
    return crosses / (2 * length)


def amdf(data):
    N = len(data)


    signal_sq = data ** 2
    sum_x2 = np.cumsum(signal_sq[::-1])[::-1]
    sum_x2_shifted = np.roll(sum_x2, -1)


    fft_signal = np.fft.fft(data, 2 * N)
    auto_corr = np.fft.ifft(fft_signal * np.conj(fft_signal)).real[:N]

    amdf = (sum_x2[:N] + sum_x2_shifted[:N] - 2 * auto_corr) / N

    return amdf




def find_average_minima_spacing(amdf_vals, prominence):
    minima_indices, _ = signal.find_peaks(-amdf_vals, prominence=prominence)

    # Ensure we have enough minima
    if len(minima_indices) < 2:
        return None  # Not enough minima to estimate frequency
    # Use up to 'num_minima' minima for averaging

    # Compute differences between consecutive minima
    spacings = np.diff(minima_indices)

    # Compute the average spacing (period estimate)
    avg_period = np.mean(spacings)

    return avg_period

def amplitude(data):
    return np.max(data) - np.min(data)


def calculate_f0(data, sample_rate):
    AMDF = amdf(data)
    amp = amplitude(AMDF)
    spacing = find_average_minima_spacing(AMDF, 0.85*amp)
    if not spacing:
        spacing = 1
    return sample_rate / spacing

def calculate_max_vol(full_data, nframes=256):
    frames = np.array_split(full_data, nframes)
    maxv = -3
    for frame in frames:
        vol = np.sqrt(np.mean(frame ** 2))
        if vol > maxv:
            maxv = vol
    return maxv

def calculate_silence_ratio_voiced_unvoiced(full_data, nframes = 256):
    ZCR_thresh = 0.03
    maxv = calculate_max_vol(full_data, nframes)
    silent = 0
    frames = np.array_split(full_data, nframes)
    silent_idxs =[]
    voiced_idxs = []
    unvoiced_idxs = []
    curr_idx = 0
    for frame in frames:
        vol = np.sqrt(sum(frame ** 2))
        if calculate_zcr(frame) < ZCR_thresh and vol < 0.05*maxv:
            silent += 1
            silent_idxs.append((curr_idx, curr_idx + len(frame)))
        elif calculate_zcr(frame) < ZCR_thresh and vol > 0.05*maxv:
            voiced_idxs.append((curr_idx, curr_idx + len(frame)))
        else:
            unvoiced_idxs.append((curr_idx, curr_idx + len(frame)))
        curr_idx += len(frame)
    return (silent/nframes), silent_idxs, voiced_idxs, unvoiced_idxs

