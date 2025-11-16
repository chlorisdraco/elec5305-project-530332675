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
