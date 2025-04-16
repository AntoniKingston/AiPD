import numpy as np
import math
import cmath

from networkx import dfs_edges
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
import random

#pamiętaj chemiku młody zawsze wlewaj listę dlugosci 2^n do tej funkcji
def FFT(list):
    n = len(list)
    if n == 1:
        #stała jest wartością wielomianu stopnia 0 w w_0
        return list
    w = cmath.exp(2*math.pi*1j/n)
    e_val = FFT(list[::2])
    o_val = FFT(list[1::2])
    ret = [0] * n
    for i in range(n//2):
        ret[i] = e_val[i] + (w**i * o_val[i])
        ret[i+n//2] = e_val[i] - (w **i* o_val[i])
    return np.array(ret)

def meaningful_FFT(signal):
    fefete = FFT(signal)[::2]
    return np.abs(fefete)


def frame_length(sr, min_frame_dur):
    T = 1 / sr
    l = 1
    while (T * l < min_frame_dur):
        l*=2
    return l

def dtft(frames, sr):
    fft = np.fft.rfft(frames, n=len(frames[0]), axis=1)
    freqs = np.fft.rfftfreq(len(frames[0]), 1 / sr)
    return fft, freqs

def frame_signal(signal, frame_length, overlap):
    hop_length = int(frame_length * (1-overlap))
    overhang = (len(signal) - frame_length) % hop_length
    pad_length = hop_length - overhang if overhang!=0 else 0
    padded_signal = np.pad(signal, (0, pad_length), 'constant')

    frames = np.lib.stride_tricks.sliding_window_view(padded_signal, window_shape=frame_length)[::hop_length]
    return frames

def window_signal(signal, window_type):
    if window_type == 'rectangular':
        return signal
    N = len(signal)
    if window_type == 'triangular':
        window = np.bartlett(N)
    elif window_type == 'hamming':
        window = np.hamming(N)
    elif window_type == 'hanning':
        window = np.hanning(N)
    elif window_type == 'blackman':
        window = np.blackman(N)
    else:
        raise ValueError('Unknown window type')
    return window * signal



def plot_spectrogram(fig, sr, signal, overlap=0.5, min_frame_dur = 0.2, window = 'rectangular', max_freq=2000):
    #first determine frame_length
    l = frame_length(sr, min_frame_dur)
    frames = frame_signal(signal, l, overlap)

    fig.clear()

    windowed_frames = np.array([window_signal(frame, window) for frame in frames])

    spec, freqs = dtft(windowed_frames, sr)
    magnitude = np.abs(spec)

    num_freq_bins = spec.shape[1]
    max_bin = int(max_freq / (sr / 2) * (num_freq_bins - 1))

    spec_db = 20 * np.log10(magnitude[:, :max_bin + 1] + 1e-10)
    spec_db = np.flipud(spec_db.T)



    num_frames = spec_db.shape[1]
    num_freqs = spec_db.shape[0]


    total_duration = len(signal) / sr
    # times = np.arange(len(signal))  / sr

    ax = fig.add_subplot(111)

    # plt.figure(figsize=(10, 10))
    sns.heatmap(spec_db, xticklabels=100, yticklabels=20, cmap='mako', cbar_kws={'label': 'dB'}, ax=ax)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram (Window: {window}, Frame: {min_frame_dur * 1000:.0f}ms, Overlap: {overlap:.2f})")

    # Set custom tick labels
    ax.set_xticks(np.linspace(0, num_frames, 10))
    ax.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, total_duration, 10)])
    ax.set_yticks(np.linspace(0, num_freqs, 5))
    ax.set_yticklabels([f"{f:.0f}Hz" for f in np.linspace(max_freq, 0, 5)])

    fig.tight_layout()
    # plt.xlabel("Time frame")
    # plt.ylabel("Frequency bin (flipped)")
    # plt.title("Spectrogram (Seaborn Matrix View)")
    # plt.xticks(np.linspace(0, num_frames, 10), labels=[f"{t:.2f}s" for t in np.linspace(0, times[-1], 10)])
    # plt.yticks(np.linspace(0, num_freqs, 10), labels=[f"{f:.0f}Hz" for f in np.linspace(max_freq, 0, 10)])
    # plt.tight_layout()
    # plt.show()


def f0_from_cepstrum(signal, sr):
    windowed_signal = np.hanning(len(signal))*signal
    spectrum = np.fft.fft(windowed_signal)
    log_spectrum = np.log(np.abs(spectrum) + 10e-10)
    cepstrum = np.fft.ifft(log_spectrum).real

    min_quefrency = int(sr/500)
    max_quefrency = int(sr/50)

    cepstrum = cepstrum[min_quefrency:max_quefrency]
    peak_idx = np.argmax(cepstrum) + min_quefrency
    f0 = sr / peak_idx
    return f0

def plot_f0_from_cepstrum(fig, signal, sr):
    fig.clear()
    l = frame_length(sr, 0.02)
    frames = frame_signal(signal, l, 0.5)
    windowed_frames = np.array([window_signal(frame, 'hamming') for frame in frames])
    f0s = [f0_from_cepstrum(frame, sr) for frame in windowed_frames]

    total_duration = len(signal) / sr
    sns.set_theme(style="darkgrid")

    ax = fig.add_subplot(111)

    sns.lineplot(f0s, ax=ax)
    ax.set_xlabel("time[s]")
    ax.set_ylabel("F0[Hz]")
    ax.set_title(f"F0 extracted using cepstrum")

    ax.set_xticks(np.linspace(0, len(f0s), 10))
    ax.set_xticklabels([f"{t:.2f}s" for t in np.linspace(0, total_duration, 10)])

    fig.tight_layout()


# def plot_f0_from_cepstrum(signal, sr):
#
#     l = frame_length(sr, 0.02)
#     frames = frame_signal(signal, l, 0.10)
#     windowed_frames = np.array([window_signal(frame, 'hamming') for frame in frames])
#     f0s = [f0_from_cepstrum(frame, sr) for frame in windowed_frames]
#
#     total_duration = len(signal) / sr
#
#
#     plt.plot(np.linspace(0, total_duration, len(f0s)), f0s)
#     plt.xlabel("time[s]")
#     plt.ylabel("F0[Hz]")
#     plt.title(f"F0 extracted using cepstrum")
#
#     plt.show()



def debug(sr, signal, min_frame_dur = 0.2, max_freq=2000):
    #first determine frame_length
    l = frame_length(sr, min_frame_dur)
    frames = frame_signal(signal, l)

    spec, freqs = dtft(frames, sr)
    return (spec, freqs)

def main():
    coefs = [i*(-1)**i for i in range(32)]
    print(full_IFFT(FFT(coefs)))


def IFFT(list):
    n = len(list)
    if n == 1:
        # stała jest wartością wielomianu stopnia 0 w w_0
        return list
    w = cmath.exp(-2 * math.pi*1j / n)

    e_val = IFFT(list[::2])
    o_val = IFFT(list[1::2])
    ret = [0] * n
    for i in range(n // 2):
        ret[i] = e_val[i] + (w**i * o_val[i])
        ret[i + n // 2] = e_val[i] - (w**i * o_val[i])

    return np.array(ret)

def full_IFFT(list):
    n = len(list)
    return np.array(IFFT(list))/n

def get_normalized_mono(path):
    sr, signal = wavfile.read(path)
    if signal.dtype != np.float32:
        signal = signal / np.max(np.abs(signal))
    if signal.ndim == 2:
        signal = signal[:, 0]
    return (sr, signal)

def frequency_centroid(fft, frequencies):
    den = np.sum(fft)
    to_sum_up = fft * frequencies
    up = sum(to_sum_up)
    return up/den
def effective_bandwidth(fft, frequencies):
    fc = frequency_centroid(fft, frequencies)
    den = np.sum(fft)
    to_sum_up = (frequencies - fc)**2 * fft**2
    up = sum(to_sum_up)
    return up/den
#function above it will filter fft to desired frequencies
def ESRB(N, fft, vol):
    den = N
    be = np.sum(fft**2)/den
    return be/vol
def SFM(fft):
    N = len(fft)
    return np.prod(fft**2)**(1/N)*N/np.sum(fft**2)

def SCF(fft, N):
    N = len(fft)
    return np.max(fft**2)*N/np.sum(fft**2)
def vol(fft):
    np.mean(fft**2)





if __name__ == '__main__':
    main()