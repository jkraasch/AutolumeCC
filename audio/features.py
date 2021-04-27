import librosa
import numpy as np


def chroma(seq, sr=44100, num_keys=12, is_spect=True):
    if is_spect:
        return librosa.feature.chroma_stft(S=seq, sr=sr, n_chroma=num_keys)
    return librosa.feature.chroma_stft(y=seq, sr=sr, n_chroma=num_keys)


def magnitude(seq, is_fft=True):
    if not is_fft:
        seq = librosa.stft(seq, hop_length=len(seq)+1)
    return np.abs(seq)


def hp_decomp(seq, is_fft=True):
    if not is_fft:
        seq = librosa.stft(seq, hop_length=len(seq)+1)
    return librosa.decompose.hpss(seq)

