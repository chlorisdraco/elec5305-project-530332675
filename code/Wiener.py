def estimate_noise_psd(noise: np.ndarray, fs: int, frame_len=FRAME, hop_len=HOP):
    """Estimate noise Power Spectral Density (PSD) from a known noise signal."""
    n_fft = int(round(frame_len * fs))
    hop = int(round(hop_len * fs))
    if n_fft % 2 == 1:
        n_fft += 1
    _, _, N = stft(
        noise,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        window="hann",
        boundary="zeros",
    )
    Pn = np.mean(np.abs(N) ** 2, axis=1)
    return Pn


def wiener_enhance(noisy: np.ndarray, fs: int, frame_len=FRAME, hop_len=HOP, Pn=None):
    """
    Perform Wiener filtering in the STFT domain.

    If `Pn` (noise PSD) is provided, it is used to compute the Wiener gain.
    Otherwise, a simple minimum-statistics approximation is used:
    per-frequency minimum energy over time as a crude noise estimate.
    """
    n_fft = int(round(frame_len * fs))
    hop = int(round(hop_len * fs))
    if n_fft % 2 == 1:
        n_fft += 1

    _, _, Y = stft(
        noisy,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        window="hann",
        boundary="zeros",
    )
    Syy = np.abs(Y) ** 2 + 1e-12

    if Pn is None:
        # Minimum-statistics estimate
        Pn = np.maximum(np.min(Syy, axis=1, keepdims=True), 1e-12)
    else:
        Pn = Pn[:, None] if Pn.ndim == 1 else Pn

    # Wiener gain (power subtraction clipped)
    G = np.clip((Syy - Pn) / Syy, 0.0, 1.0)
    Xhat = G * Y

    _, x_hat = istft(
        Xhat,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        window="hann",
        input_onesided=True,
        boundary=True,
    )
    return x_hat.astype(np.float32)
