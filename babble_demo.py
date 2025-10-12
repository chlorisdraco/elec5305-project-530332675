"""
Babble-Noise Speech Enhancement Mini Demo (All-in-One)

- 生成“干净语音风格”的合成信号 + 合成 babble 噪声
- 以多档 SNR（默认 0/5/10 dB）混合，保存 noisy.wav
- 基于 STFT 的 Wiener 基线增强（既支持提供 noise_ref，也支持无参考的最小统计法）
- 计算 SI-SDR、Segmental SNR（可选：PESQ/STOI，如果安装）
- 所有输出写入 ./results/ 目录

运行:
    python babble_demo.py
"""

import os, json, math
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, lfilter, butter

# ---------------------------
# 全局参数
# ---------------------------
FS = 16000
DUR = 6.0               # 秒
SNRS = [0, 5, 10]       # 混合SNR档
FRAME = 0.02            # 20 ms
HOP = 0.01              # 10 ms

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# 合成音频（干净语音风格 & babble）
# ---------------------------
def synth_clean_speech_like(fs: int, dur: float) -> np.ndarray:
    """粗略的人声样式：谐波叠加 + 两个“共振峰”带通"""
    t = np.arange(int(fs*dur))/fs
    f0 = 120 + 10*np.sin(2*np.pi*0.5*t)  # 轻微颤动
    sig = np.zeros_like(t)
    acc = np.cumsum(f0)/fs
    for k in range(1, 15):
        sig += (1.0/k) * np.sin(2*np.pi*k*acc)
    sig *= 0.15

    def bandpass(x, low, high, fs, order=2):
        b,a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
        return lfilter(b,a,x)

    v1 = bandpass(sig, 300, 900, fs)
    v2 = bandpass(sig, 1100, 1900, fs)
    out = 0.6*v1 + 0.4*v2
    out = out / (np.max(np.abs(out)) + 1e-8) * 0.5
    return out.astype(np.float32)

def synth_babble(fs: int, dur: float, n_talkers: int = 10) -> np.ndarray:
    """合成 babble：多路带限噪声 + 低频包络调制"""
    rng = np.random.default_rng(0)
    t = np.arange(int(fs*dur))/fs
    babble = np.zeros_like(t)
    for _ in range(n_talkers):
        env = np.clip(rng.normal(0.6, 0.2, size=t.shape), 0.1, 1.2)
        noise = rng.normal(0, 1, size=t.shape)
        b,a = butter(3, 3000/(fs/2), btype='low')
        noise = lfilter(b,a,noise)
        mod = 0.5*(1+np.sin(2*np.pi*4*t + rng.uniform(0, 2*np.pi)))
        babble += (env * mod * noise)
    babble = babble / (np.max(np.abs(babble)) + 1e-8) * 0.5
    return babble.astype(np.float32)

# ---------------------------
# 混合与指标
# ---------------------------
def mix_at_snr(x: np.ndarray, n: np.ndarray, snr_db: float):
    N = min(len(x), len(n))
    x = x[:N]; n = n[:N]
    Px = np.mean(x**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12
    target_Pn = Px / (10**(snr_db/10.0))
    scale = math.sqrt(target_Pn / Pn)
    n2 = n * scale
    y = x + n2
    return y, n2

def si_sdr(reference: np.ndarray, estimation: np.ndarray, eps=1e-8) -> float:
    ref = reference - np.mean(reference)
    est = estimation - np.mean(estimation)
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    e_noise = est - s_target
    ratio = (np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps)
    return 10*np.log10(ratio)

def seg_snr(reference: np.ndarray, estimation: np.ndarray, fs: int, frame_len=0.02, eps=1e-8) -> float:
    N = len(reference)
    L = int(round(frame_len*fs))
    if L <= 0: return float("nan")
    K = max(1, N // L)
    snrs = []
    for k in range(K):
        s = reference[k*L:(k+1)*L]
        e = reference[k*L:(k+1)*L] - estimation[k*L:(k+1)*L]
        if len(s) < L or len(e) < L:
            break
        Ps = np.mean(s**2) + eps
        Pe = np.mean(e**2) + eps
        snrs.append(10*np.log10(Ps/Pe))
    return float(np.mean(snrs)) if snrs else float("nan")

def try_optional_metrics(clean: np.ndarray, enhanced: np.ndarray, fs: int):
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

# ---------------------------
# STFT 域 Wiener 基线
# ---------------------------
def estimate_noise_psd(noise: np.ndarray, fs: int, frame_len=FRAME, hop_len=HOP):
    n_fft = int(round(frame_len*fs))
    hop = int(round(hop_len*fs))
    if n_fft % 2 == 1: n_fft += 1
    _, _, N = stft(noise, fs=fs, nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft,
                   window='hann', boundary='zeros')
    Pn = np.mean(np.abs(N)**2, axis=1)  # (F,)
    return Pn

def wiener_enhance(noisy: np.ndarray, fs: int, frame_len=FRAME, hop_len=HOP, Pn=None):
    """
    - 若提供 Pn（噪声PSD，形状(F,)或(F,1)），用它计算 Wiener 增益；
    - 否则用最小统计法(min-stat)的简单近似：每频率取整段最小能量作为噪声估计。
    """
    n_fft = int(round(frame_len*fs))
    hop = int(round(hop_len*fs))
    if n_fft % 2 == 1: n_fft += 1

    # boundary='zeros' 保证 iSTFT 可逆长度
    _, _, Y = stft(noisy, fs=fs, nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft,
                   window='hann', boundary='zeros')
    Syy = np.abs(Y)**2 + 1e-12

    if Pn is None:
        Pn = np.maximum(np.min(Syy, axis=1, keepdims=True), 1e-12)  # (F,1)
    else:
        Pn = Pn[:, None] if Pn.ndim == 1 else Pn

    G = np.clip((Syy - Pn) / Syy, 0.0, 1.0)  # Wiener 增益（功率域剪裁）
    Xhat = G * Y

    _, x_hat = istft(Xhat, fs=fs, nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft,
                     window='hann', input_onesided=True, boundary=True)
    return x_hat.astype(np.float32)

# ---------------------------
# 主流程
# ---------------------------
def main():
    # 1) 生成合成音频
    clean = synth_clean_speech_like(FS, DUR)
    babble = synth_babble(FS, DUR, n_talkers=10)
    sf.write(os.path.join(RESULTS_DIR, "clean.wav"), clean, FS)
    sf.write(os.path.join(RESULTS_DIR, "noise_babble.wav"), babble, FS)

    summary = {}
    for snr in SNRS:
        tag = f"SNR{snr}dB"
        # 2) 混合
        noisy, noise_used = mix_at_snr(clean, babble, snr)
        noisy_path = os.path.join(RESULTS_DIR, f"noisy_{tag}.wav")
        sf.write(noisy_path, noisy, FS)

        # 3a) 有噪声参考的 Wiener（更接近“已知噪声统计”的理想情况）
        Pn_ref = estimate_noise_psd(noise_used, FS, FRAME, HOP)
        enh_ref = wiener_enhance(noisy, FS, FRAME, HOP, Pn=Pn_ref)
        enh_ref_path = os.path.join(RESULTS_DIR, f"enhanced_ref_{tag}.wav")
        sf.write(enh_ref_path, enh_ref, FS)

        # 3b) 无噪声参考的 Wiener（最小统计近似）
        enh_ms = wiener_enhance(noisy, FS, FRAME, HOP, Pn=None)
        enh_ms_path = os.path.join(RESULTS_DIR, f"enhanced_ms_{tag}.wav")
        sf.write(enh_ms_path, enh_ms, FS)

        # 4) 计算指标
        # 对齐长度
        N = min(len(clean), len(noisy), len(enh_ref), len(enh_ms))
        x = clean[:N]; y = noisy[:N]; z_ref = enh_ref[:N]; z_ms = enh_ms[:N]

        metrics = {
            "SI_SDR_noisy": float(si_sdr(x, y)),
            "SI_SDR_enh_ref": float(si_sdr(x, z_ref)),
            "SI_SDR_enh_ms": float(si_sdr(x, z_ms)),
            "SegSNR_noisy": float(seg_snr(x, y, FS)),
            "SegSNR_enh_ref": float(seg_snr(x, z_ref, FS)),
            "SegSNR_enh_ms": float(seg_snr(x, z_ms, FS)),
        }
        metrics["SI_SDR_impr_ref"] = metrics["SI_SDR_enh_ref"] - metrics["SI_SDR_noisy"]
        metrics["SI_SDR_impr_ms"]  = metrics["SI_SDR_enh_ms"]  - metrics["SI_SDR_noisy"]
        metrics["SegSNR_impr_ref"] = metrics["SegSNR_enh_ref"] - metrics["SegSNR_noisy"]
        metrics["SegSNR_impr_ms"]  = metrics["SegSNR_enh_ms"]  - metrics["SegSNR_noisy"]

        # 可选 PESQ/STOI（若安装）
        opt_ref = try_optional_metrics(x, z_ref, FS)
        opt_ms  = try_optional_metrics(x, z_ms, FS)
        metrics.update({f"ref_{k}": v for k, v in opt_ref.items()})
        metrics.update({f"ms_{k}": v for k, v in opt_ms.items()})

        summary[tag] = metrics

        # 打印摘要
        print(f"\n=== {tag} ===")
        for k,v in metrics.items():
            print(f"{k:>16}: {v}")

    # 保存指标
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print(f"WAV files saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
