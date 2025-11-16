def mix_at_snr(x: np.ndarray, n: np.ndarray, snr_db: float):
    """Mix clean and noise at a target SNR (in dB)."""
    N = min(len(x), len(n))
    x = x[:N]
    n = n[:N]
    Px = np.mean(x**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12
    target_Pn = Px / (10 ** (snr_db / 10.0))
    scale = math.sqrt(target_Pn / Pn)
    n2 = n * scale
    y = x + n2
    return y, n2


def si_sdr(reference: np.ndarray, estimation: np.ndarray, eps=1e-8) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)."""
    ref = reference - np.mean(reference)
    est = estimation - np.mean(estimation)
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    e_noise = est - s_target
    ratio = (np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps)
    return 10 * np.log10(ratio)


def seg_snr(reference: np.ndarray, estimation: np.ndarray, fs: int, frame_len=0.02, eps=1e-8) -> float:
    """Compute Segmental SNR over short overlapping frames."""
    N = len(reference)
    L = int(round(frame_len * fs))
    if L <= 0:
        return float("nan")
    K = max(1, N // L)
    snrs = []
    for k in range(K):
        s = reference[k * L : (k + 1) * L]
        e = reference[k * L : (k + 1) * L] - estimation[k * L : (k + 1) * L]
        if len(s) < L or len(e) < L:
            break
        Ps = np.mean(s**2) + eps
        Pe = np.mean(e**2) + eps
        snrs.append(10 * np.log10(Ps / Pe))
    return float(np.mean(snrs)) if snrs else float("nan")


def try_optional_metrics(clean: np.ndarray, enhanced: np.ndarray, fs: int):
    """Optionally compute PESQ and STOI if corresponding libraries are available."""
    out = {}
    try:
        from pesq import pesq
        out["PESQ_wb"] = float(pesq(fs, clean, enhanced, "wb"))
    except Exception:
        out["PESQ_wb"] = None
    try:
        from pystoi import stoi
        out["STOI"] = float(stoi(clean, enhanced, fs, extended=False))
    except Exception:
        out["STOI"] = None
    return out
