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
        noisy_path = os.path.join(RESULTS_DIR, f"noisy_{tag}.wav")
        sf.write(noisy_path, noisy, FS)

        # Step 3a: Wiener with reference noise PSD (ideal case)
        Pn_ref = estimate_noise_psd(noise_used, FS, FRAME, HOP)
        enh_ref = wiener_enhance(noisy, FS, FRAME, HOP, Pn=Pn_ref)
        sf.write(os.path.join(RESULTS_DIR, f"enhanced_ref_{tag}.wav"), enh_ref, FS)

        # Step 3b: Wiener with min-statistics (no noise reference)
        enh_ms = wiener_enhance(noisy, FS, FRAME, HOP, Pn=None)
        sf.write(os.path.join(RESULTS_DIR, f"enhanced_ms_{tag}.wav"), enh_ms, FS)

        # Step 4: Evaluate metrics
        N = min(len(clean), len(noisy), len(enh_ref), len(enh_ms))
        x = clean[:N]
        y = noisy[:N]
        z_ref = enh_ref[:N]
        z_ms = enh_ms[:N]

        metrics = {
            "SI_SDR_noisy": float(si_sdr(x, y)),
            "SI_SDR_enh_ref": float(si_sdr(x, z_ref)),
            "SI_SDR_enh_ms": float(si_sdr(x, z_ms)),
            "SegSNR_noisy": float(seg_snr(x, y, FS)),
            "SegSNR_enh_ref": float(seg_snr(x, z_ref, FS)),
            "SegSNR_enh_ms": float(seg_snr(x, z_ms, FS)),
        }

        # Improvements relative to noisy
        metrics["SI_SDR_impr_ref"] = metrics["SI_SDR_enh_ref"] - metrics["SI_SDR_noisy"]
        metrics["SI_SDR_impr_ms"] = metrics["SI_SDR_enh_ms"] - metrics["SI_SDR_noisy"]
        metrics["SegSNR_impr_ref"] = metrics["SegSNR_enh_ref"] - metrics["SegSNR_noisy"]
        metrics["SegSNR_impr_ms"] = metrics["SegSNR_enh_ms"] - metrics["SegSNR_noisy"]

        # Optional PESQ/STOI
        opt_ref = try_optional_metrics(x, z_ref, FS)
        opt_ms = try_optional_metrics(x, z_ms, FS)
        metrics.update({f"ref_{k}": v for k, v in opt_ref.items()})
        metrics.update({f"ms_{k}": v for k, v in opt_ms.items()})

        summary[tag] = metrics

        # Print summary for each SNR
        print(f"\n=== {tag} ===")
        for k, v in metrics.items():
            print(f"{k:>16}: {v}")

    # Save metrics to JSON
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'metrics.json')}")
    print(f"WAV files saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
