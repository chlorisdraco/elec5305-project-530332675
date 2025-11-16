# <span style="color:#FF6F00;">Real-Time Speech Enhancement under Babble Noise Using Wiener Filtering and Spectral Subtraction</span>

**Author:** Jiahao Xu  
**SID:** 530332675  
**Date:** 06/09/2025  

---

## <span style="color:#1565C0;">Branch Structure Explanation</span>

This repository is organised into two main branches to clearly separate source code and documentation.

### **Main Branch**
Contains the project proposal, progress report, and high-level documentation.  
Provides a summary of project objectives, background, results, and future work.  
Suitable for quick academic review without running code.

### **Code Branch**
Contains all Python source code, synthetic data generation scripts, and generated results (`.wav`, `.json`).  
Main script: **`babble_demo.py`**, which runs the entire experiment pipeline.  
All files in this branch are runnable and reproducible.  
Outputs are saved in the `results/` directory.

---

## <span style="color:#1976D2;">Project Overview</span>

This project builds a simple and reproducible baseline for real-time speech enhancement under babble noise.  
Babble noise is difficult because it changes quickly and resembles real speech.  
Three classical STFT-based enhancement methods are implemented:

1. **Wiener filtering (reference noise PSD)**  
2. **Wiener filtering (minimum-statistics)**  
3. **Spectral Subtraction (minimum-statistics)**  

All methods are tested at **0 dB, 5 dB, and 10 dB SNR**.  
The goal is to measure how these traditional filters perform and to identify their limitations.  
This baseline also prepares the system for later comparison with causal neural models.

---

## <span style="color:#2E7D32;">Background and Motivation</span>

Speech enhancement is important in daily applications such as online meetings, hearing aids, and mobile communication.  
However, babble noise — created by many overlapping speakers — is hard to remove because it shares similar frequency and timing patterns with the target voice.

Traditional spectral subtraction and minimum-statistics methods assume the noise is stable.  
This assumption often fails for babble noise.  
Therefore, this project aims to:

- Build a clean and transparent baseline using classical filters  
- Evaluate the stability and limits of these methods  
- Support later development of more adaptive or learning-based approaches  

---

## <span style="color:#2E7D32;">Research Question</span>

**Which classical baseline method performs better under babble noise: Wiener filtering or Spectral Subtraction?**

This question guides the evaluation and helps highlight the weaknesses of traditional noise estimation.

---

## <span style="color:#388E3C;">Proposed Methodology</span>

### **Synthetic Data Generation**
Speech-like signals and babble noise are synthetically generated.  
Mixtures are created at **0 dB, 5 dB, and 10 dB SNR**.

### **STFT-Domain Enhancement Methods**
Three baseline methods are evaluated:

**1. Wiener Filtering (Reference PSD)**  
Noise PSD is computed from a clean noise sample.  
This represents an upper-bound for classical algorithms.

**2. Wiener Filtering (Minimum-Statistics)**  
Noise PSD is estimated directly from the mixture using frequency-wise minima.  
This method is simple but unstable in non-stationary noise.

**3. Spectral Subtraction (Minimum-Statistics)**  
Subtracts an estimated noise spectrum from the noisy signal.  
This method is commonly used as a classical baseline.

### **Evaluation Metrics**
- SI-SDR  
- Segmental SNR  
- Optional: PESQ and STOI  

All results are saved in `.wav` and `.json` formats.

### **Implementation and Runtime**
The system is implemented using Python (`numpy`, `scipy`, and `soundfile`).  
Processing time for 6 s audio is below real time.

---

## <span style="color:#E65100;">Method Implementation</span>

Four main scripts are included:

1. **`mix_babble.py`** – Generates clean + babble mixtures.  
2. **`baseline_wiener.py`** – Implements reference and min-stat Wiener filtering.  
3. **`baseline_spectral_subtraction.py`** – Implements magnitude spectral subtraction.  
4. **`babble_demo.py`** – Runs the full pipeline and logs metrics.

Intermediate `.wav` files are stored in the `results/` folder.  
All components run automatically.

---

## <span style="color:#F57C00;">Expected Outcomes</span>

- A verified classical baseline for babble-noise speech enhancement.  
- Objective improvements of **+2–4 dB** using reference Wiener filtering.  
- A GitHub project with reproducible code and audio demos.  
- Insights into why traditional methods struggle with fast-changing, speech-like noise.  
- A foundation for future models such as decision-directed Wiener filtering or causal CRN-lite.

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

### **Results**

| SNR (dB) | Method Type | SI-SDR (dB) | ΔSI-SDR | SegSNR (dB) | ΔSegSNR |
|-----------|-------------|-------------|--------:|-------------|--------:|
| 0 dB | Noisy | −0.00 | – | 0.03 | – |
| | Wiener-ref | 2.37 | **+2.38** | 3.99 | **+3.96** |
| | Wiener-ms | −0.05 | −0.05 | 0.05 | +0.01 |
| | Spectral Subtraction | −0.03 | −0.03 | 0.04 | +0.01 |
| 5 dB | Noisy | 5.00 | – | 5.03 | – |
| | Wiener-ref | 7.52 | **+2.53** | 8.20 | **+3.16** |
| | Wiener-ms | 4.90 | −0.10 | 5.04 | +0.00 |
| | Spectral Subtraction | 4.95 | −0.05 | 5.04 | +0.00 |
| 10 dB | Noisy | 10.00 | – | 10.03 | – |
| | Wiener-ref | 12.21 | **+2.21** | 12.51 | **+2.47** |
| | Wiener-ms | 9.85 | −0.15 | 10.01 | −0.02 |
| | Spectral Subtraction | 9.93 | −0.07 | 10.03 | −0.00 |

### **Discussion**

The reference Wiener filter delivered the strongest improvements across all SNR levels.  
Because it has access to clean noise PSD, it produces stable gain estimates and reduces distortion.

The minimum-statistics Wiener filter and Spectral Subtraction both struggled.  
Babble noise changes quickly, and minimum-statistics noise estimates become unstable.  
As a result, both methods provide very limited gains and may even reduce SI-SDR.  
Spectral Subtraction performs slightly better than Wiener-ms, but still does not improve SI-SDR in most cases.

These results show that classical non-adaptive enhancement methods face clear limitations in speech-like noise.  
More adaptive or learning-based approaches are likely to perform better.

---

## <span style="color:#455A64;">References</span>

1. Loizou, P. C. (2013). *Speech Enhancement: Theory and Practice*. CRC Press.  
2. Reddy, C. K. et al. (2021). “DNS Challenge: Improving Noise Suppression Models.” *Proc. Interspeech*.  
3. Ephraim, Y., & Malah, D. (1984). “Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator.” *IEEE TASSP*.  
4. Boll, S. (1979). “Suppression of acoustic noise in speech using spectral subtraction.” *IEEE TASSP*.  


