# <span style="color:#FF6F00;">Real-Time Speech Enhancement under Babble Noise Using Wiener Filtering</span>

**Author:** Jiahao Xu  
**SID:** 530332675  
**Date:** 06/09/2025  

---

## <span style="color:#1565C0;">Branch Structure Explanation</span>
This repository is organized into two main branches to clearly separate source code and documentation:

**Main Branch**
Contains the project proposal, progress report, and overall documentation.
Serves as the public-facing summary of the project, including objectives, background, results, and future work.
Suitable for quick review without the need to run any code.

**Code Branch**
Contains all Python source code, data files, and generated results (.wav, .json).
Main script: babble_demo.py
 — runs the full pipeline for speech enhancement under babble noise.
All files in this branch are runnable and reproducible.
Output files will be saved in the results/ directory.

## <span style="color:#1976D2;">Project Overview</span>

This project focuses on developing a lightweight and reproducible baseline for real-time speech enhancement under babble noise, which is one of the most challenging non-stationary noise conditions.
A classical Short-Time Fourier Transform (STFT)-based Wiener filtering approach is implemented and evaluated at multiple input signal-to-noise ratios (SNRs).
The aim is to establish an early functional pipeline that generates measurable objective improvements in intelligibility and quality while maintaining a low computational cost suitable for real-time processing.

---

## <span style="color:#2E7D32;">Background and Motivation</span>

Speech enhancement plays a vital role in applications such as online meetings, hearing aids, and mobile communication, where speech quality and intelligibility are often degraded by environmental noise.
Among various noise types, babble noise—caused by overlapping speech from multiple speakers—is particularly difficult to handle because it shares similar spectral and temporal features with the target voice.
Traditional spectral subtraction or stationary noise estimation methods fail to track such dynamic variations.
Therefore, this work aims to (i) construct a clear baseline using Wiener filtering, (ii) quantify its limitations under babble noise, and (iii) lay the groundwork for later comparison with causal learning-based models such as CRN-lite or DCCRN.

---

## <span style="color:#388E3C;">Proposed Methodology</span>

Synthetic Data Generation
Clean speech-like and babble noise signals are procedurally generated to simulate controlled multi-talker environments at 0 dB, 5 dB, and 10 dB SNRs.

STFT-Domain Wiener Filtering
The noisy signal is decomposed using STFT (20 ms frame, 10 ms hop, Hann window).
Two versions of the Wiener filter are tested:

Reference-based: noise power spectral density (PSD) estimated from a known noise sample (upper-bound case).

Min-statistic: noise PSD estimated directly from the noisy mixture (realistic case).

Evaluation Metrics
Objective measures include SI-SDR, Segmental SNR, and optionally PESQ/STOI if available.
Experiments are conducted on all three SNR levels, and results are logged in JSON and WAV formats for reproducibility.

Implementation and Runtime
The system is implemented in pure Python using numpy, scipy, and soundfile, ensuring real-time feasibility (processing time < 1× audio length).

---

## <span style="color:#E65100;">Method Implementation</span>

The system was implemented entirely in Python 3 using open-source libraries (`numpy`, `scipy`, `soundfile`).  
A modular structure was designed with four core scripts:

1. **`mix_babble.py`** – Generates mixtures of clean and babble noise at predefined SNRs (0 / 5 / 10 dB).  
2. **`baseline_wiener.py`** – Performs STFT-domain Wiener filtering. Supports both reference-based and min-stat noise estimation.  
3. **`eval_metrics.py`** – Computes SI-SDR and Segmental SNR, with optional PESQ/STOI if installed.  
4. **`run_demo.py` / `babble_demo.py`** – End-to-end driver script that synthesizes toy data, performs enhancement, and logs results.

Each stage runs automatically and saves intermediate `.wav` files in the `results/` folder.  
The total runtime for 6 s of audio is below real time (< 6 s on a typical laptop CPU).

## <span style="color:#F57C00;">Expected Outcomes</span>

A verified baseline pipeline capable of enhancing speech under babble noise with measurable improvements.

Demonstrated objective gains of +2–3 dB in SI-SDR and +2–4 dB in Segmental SNR across 0–10 dB SNR conditions.

A GitHub project site hosting code, results, and demo audio samples for feedback and further extension.

Insights highlighting the limitations of classical Wiener filtering in non-stationary, speech-like noise—motivating future work on decision-directed estimators or causal CRN-lite networks.

---

## <span style="color:#1565C0;">Timeline (Weeks 6–13)</span>

| Weeks  | Tasks                                   |
|--------|-----------------------------------------|
| 6–7    | Review research and collect datasets    |
| 8–9    | Build first model and test baseline methods |
| 10–11  | Improve model and run evaluation        |
| 12–13  | Write report, prepare GitHub and demos  |

---

## <span style="color:#0D47A1;">Results and Discussion</span>

| SNR (dB) | Method Type | SI-SDR (dB) | ΔSI-SDR | SegSNR (dB) | ΔSegSNR |
|-----------|-------------|-------------|--------:|-------------|--------:|
| 0 dB | Noisy | −0.00 | – | 0.03 | – |
| | Wiener-ref | 2.37 | **+2.38** | 3.99 | **+3.96** |
| | Wiener-ms | −0.05 | −0.05 | 0.05 | +0.01 |
| 5 dB | Noisy | 5.00 | – | 5.03 | – |
| | Wiener-ref | 7.52 | **+2.53** | 8.20 | **+3.16** |
| | Wiener-ms | 4.90 | −0.10 | 5.04 | +0.00 |
| 10 dB | Noisy | 10.00 | – | 10.03 | – |
| | Wiener-ref | 12.21 | **+2.21** | 12.51 | **+2.47** |
| | Wiener-ms | 9.85 | −0.15 | 10.01 | −0.02 |

The reference-based Wiener filter consistently improves both SI-SDR and Segmental SNR by 2–4 dB across all SNR levels.  
In contrast, the minimum-statistics version shows limited or negative gains under babble noise, confirming that traditional noise-tracking struggles with highly non-stationary, speech-like interference.

## <span style="color:#455A64;">References</span>
1.Loizou, P. C. (2013). Speech Enhancement: Theory and Practice. CRC Press.

2.Reddy, C. K. et al. (2021). “DNS Challenge: Improving Noise Suppression Models.” Proc. Interspeech.

3.Ephraim, Y., & Malah, D. (1984). “Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator.” IEEE Trans. Acoustics, Speech, and Signal Processing, 32(6), 1109–1121.

4.Tan, K., & Wang, D. (2019). “Learning Complex Spectral Mapping with a Convolutional Recurrent Network for Speech Enhancement.” IEEE/ACM Trans. Audio, Speech, and Language Processing, 27(12), 1996–2008.
