"""
Babble-Noise Speech Enhancement Mini Demo (All-in-One)

This demo generates synthetic "clean speech-like" and "babble noise" signals,
mixes them at multiple SNR levels (0 / 5 / 10 dB), and applies two STFT-domain
baseline methods for enhancement:

1. Wiener filtering (reference-based and minimum-statistics)
2. Spectral subtraction (minimum-statistics type)

Objective metrics (SI-SDR, Segmental SNR, and optional PESQ/STOI if available)
are computed and saved along with output WAV files in ./results/.

Usage:
    python babble_demo.py
"""

import os, json, math
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, lfilter, butter

# ================================================================
# Global configuration
# ================================================================
FS = 16000                # Sampling rate (Hz)
DUR = 6.0                 # Signal duration in seconds
SNRS = [0, 5, 10]         # Target SNR levels (dB)
FRAME = 0.02              # Frame length for STFT (20 ms)
HOP = 0.01                # Hop length (10 ms)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# 1. Signal synthesis (clean speech-like + babble noise)
# ================================================================
def synth_clean_speech_like(fs: int, dur: float) -> np.ndarray:
    """
    Generate a synthetic "speech-like" signal using harmonic stacking
    and two resonant bandpass filters (roughly simulating formants).
    """
    t = np.arange(int(fs * dur)) / fs
    f0 = 120 + 10 * np.sin(2 * np.pi * 0.5 * t)  # Slight pitch modulation
    sig = np.zeros_like(t)
    acc = np.cumsum(f0) / fs

    # Add harmonics up to the 15th
    for k in range(1, 15):
        sig += (1.0 / k) * np.sin(2 * np.pi * k * acc)
    sig *= 0.15

    # Two formant-like bands
    def bandpass(x, low, high, fs, order=2):
        b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
        return lfilter(b, a, x)

    v1 = bandpass(sig, 300, 900, fs)
    v2 = bandpass(sig, 1100, 1900, fs)
    out = 0.6 * v1 + 0.4 * v2
    out = out / (np.max(np.abs(out)) + 1e-8) * 0.5
    return out.astype(np.float32)


def synth_babble(fs: int, dur: float, n_talkers: int = 10) -> np.ndarray:
    """
    Generate synthetic "babble noise" by summing multiple band-limited,
    amplitude-modulated noise sources (simulating multiple talkers).
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * dur)) / fs
    babble = np.zeros_like(t)

    for _ in range(n_talkers):
        env = np.clip(rng.normal(0.6, 0.2, size=t.shape), 0.1, 1.2)
        noise = rng.normal(0, 1, size=t.shape)
        b, a = butter(3, 3000 / (fs / 2), btype="low")
        noise = lfilter(b, a, noise)
        mod = 0.5 * (1 + np.sin(2 * np.pi * 4 * t + rng.uniform(0, 2 * np.pi)))
        babble += env * mod * noise

    babble = babble / (np.max(np.abs(babble)) + 1e-8) * 0.5
    return babble.astype(np.float32)


# ================================================================
# 2. Mixing and objective metrics
# ================================================================
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
    """Compute Segmental SNR over short non-overlapping frames."""
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


# ================================================================
# 3. STFT-domain Wiener baseline
# ================================================================
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


# ================================================================
# 4. Spectral Subtraction baseline
# ================================================================
def spectral_subtraction(noisy: np.ndarray, fs: int, frame_len=FRAME, hop_len=HOP, alpha: float = 1.0):
    """
    Simple STFT-domain Spectral Subtraction baseline.

    - alpha: over-subtraction factor.
      alpha = 1.0 is the basic case.
      alpha > 1.0 will remove more noise but can introduce musical noise
      and speech distortion.

    This implementation:
    - estimates noise power via a minimum-statistics style approach
      (per-frequency minimum power across time),
    - subtracts scaled noise power from the noisy spectrum,
    - clips negative values to a small floor,
    - reconstructs signal using noisy phase.
    """
    n_fft = int(round(frame_len * fs))
    hop = int(round(hop_len * fs))
    if n_fft % 2 == 1:
        n_fft += 1

    # STFT of the noisy signal
    _, _, Y = stft(
        noisy,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        window="hann",
        boundary="zeros",
    )
    mag = np.abs(Y)
    phase = np.angle(Y)

    # Estimate noise power via minimum statistics
    Pn = np.minimum.reduce(mag**2, axis=1, keepdims=True)  # (F,1)

    # Spectral power subtraction
    S_hat = mag**2 - alpha * Pn
    S_hat = np.maximum(S_hat, 1e-12)  # avoid negative / zero power

    # Rebuild complex spectrum with original phase
    Z = np.sqrt(S_hat) * np.exp(1j * phase)

    # Inverse STFT
    _, enhanced = istft(
        Z,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        window="hann",
        input_onesided=True,
        boundary=True,
    )
    return enhanced.astype(np.float32)


# ================================================================
# 5. Main execution
# ================================================================
def main():
    """Main pipeline: generate signals, mix, enhance, and evaluate."""

    # Step 1: Generate clean and noise signals
    clean = synth_clean_speech_like(FS, DUR)
    babble = synth_babble(FS, DUR, n_talkers=10)
    sf.write(os.path.join(RESULTS_DIR, "clean.wav"), clean, FS)
    sf.write(os.path.join(RESULTS_DIR, "noise_babble.wav"), babble, FS)

    summary = {}

    for snr in SNRS:
        tag = f"SNR{snr}dB"

        # Step 2: Mix at target SNR
        noisy, noise_used = mix_at_snr(clean, babble, snr)
        sf.write(os.path.join(RESULTS_DIR, f"noisy_{tag}.wav"), noisy, FS)

        # Step 3a: Wiener with reference noise PSD (ideal case)
        Pn_ref = estimate_noise_psd(noise_used, FS, FRAME, HOP)
        enh_ref = wiener_enhance(noisy, FS, FRAME, HOP, Pn=Pn_ref)
        sf.write(os.path.join(RESULTS_DIR, f"enhanced_ref_{tag}.wav"), enh_ref, FS)

        # Step 3b: Wiener with min-statistics (no noise reference)
        enh_ms = wiener_enhance(noisy, FS, FRAME, HOP, Pn=None)
        sf.write(os.path.join(RESULTS_DIR, f"enhanced_ms_{tag}.wav"), enh_ms, FS)

        # Step 3c: Spectral subtraction (min-statistics, alpha=1.0)
        enh_ss = spectral_subtraction(noisy, FS, FRAME, HOP, alpha=1.0)
        sf.write(os.path.join(RESULTS_DIR, f"enhanced_ss_{tag}.wav"), enh_ss, FS)

        # Step 4: Evaluate metrics for all methods
        N = min(len(clean), len(noisy), len(enh_ref), len(enh_ms), len(enh_ss))
        x = clean[:N]
        y = noisy[:N]
        z_ref = enh_ref[:N]
        z_ms = enh_ms[:N]
        z_ss = enh_ss[:N]

        metrics = {
            "SI_SDR_noisy": float(si_sdr(x, y)),
            "SI_SDR_enh_ref": float(si_sdr(x, z_ref)),
            "SI_SDR_enh_ms": float(si_sdr(x, z_ms)),
            "SI_SDR_enh_ss": float(si_sdr(x, z_ss)),
            "SegSNR_noisy": float(seg_snr(x, y, FS)),
            "SegSNR_enh_ref": float(seg_snr(x, z_ref, FS)),
            "SegSNR_enh_ms": float(seg_snr(x, z_ms, FS)),
            "SegSNR_enh_ss": float(seg_snr(x, z_ss, FS)),
        }

        # Improvements relative to noisy
        metrics["SI_SDR_impr_ref"] = metrics["SI_SDR_enh_ref"] - metrics["SI_SDR_noisy"]
        metrics["SI_SDR_impr_ms"] = metrics["SI_SDR_enh_ms"] - metrics["SI_SDR_noisy"]
        metrics["SI_SDR_impr_ss"] = metrics["SI_SDR_enh_ss"] - metrics["SI_SDR_noisy"]
        metrics["SegSNR_impr_ref"] = metrics["SegSNR_enh_ref"] - metrics["SegSNR_noisy"]
        metrics["SegSNR_impr_ms"] = metrics["SegSNR_enh_ms"] - metrics["SegSNR_noisy"]
        metrics["SegSNR_impr_ss"] = metrics["SegSNR_enh_ss"] - metrics["SegSNR_noisy"]

        # Optional PESQ/STOI
        opt_ref = try_optional_metrics(x, z_ref, FS)
        opt_ms = try_optional_metrics(x, z_ms, FS)
        opt_ss = try_optional_metrics(x, z_ss, FS)
        metrics.update({f"ref_{k}": v for k, v in opt_ref.items()})
        metrics.update({f"ms_{k}": v for k, v in opt_ms.items()})
        metrics.update({f"ss_{k}": v for k, v in opt_ss.items()})

        summary[tag] = metrics

        # Print summary for each SNR
        print(f"\n=== {tag} ===")
        for k, v in metrics.items():
            print(f"{k:>18}: {v}")

    # Save metrics to JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print(f"WAV files saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
