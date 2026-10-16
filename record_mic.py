"""
record_mic.py - record laptop mic samples and extract FFT features (testing only).

Setup:
    pip install sounddevice numpy scipy matplotlib

Usage:
    python record_mic.py --label normal --seconds 20
    python record_mic.py --label fan_on --seconds 20 --plot
    python record_mic.py --list-devices

Outputs (in ./mic_data):
    <label>_<timestamp>.wav            raw audio, 16 kHz mono
    <label>_<timestamp>_features.csv   per-window features:
                                       t_s, peak_hz, band_00..band_15, label

Sample rate and window size match the planned INMP441 setup (16 kHz), so the
same feature code can later run on real hardware recordings.
"""
import argparse
import csv
import os
import time

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

RATE = 16000
WINDOW = 1024          # samples per FFT window (64 ms at 16 kHz)
HOP = 512              # 50% overlap
N_BANDS = 16           # equal-width bands from 0 Hz to Nyquist (8 kHz)


def record(seconds, device=None):
    print("Recording %d s... make some noise (fan, tapping, clapping)." % seconds)
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    print("Done.")
    return audio[:, 0]


def extract_features(x):
    """Per-window peak frequency and band energies (dB)."""
    window = np.hanning(WINDOW)
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / RATE)
    edges = np.linspace(0, len(freqs), N_BANDS + 1, dtype=int)
    rows = []
    for start in range(0, len(x) - WINDOW + 1, HOP):
        seg = x[start:start + WINDOW] * window
        spec = np.abs(np.fft.rfft(seg)) ** 2
        peak_hz = freqs[np.argmax(spec[1:]) + 1]          # skip DC bin
        bands = [10 * np.log10(spec[edges[i]:edges[i + 1]].sum() + 1e-12)
                 for i in range(N_BANDS)]
        rows.append((start / RATE, peak_hz, bands))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="normal")
    p.add_argument("--seconds", type=int, default=5)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--outdir", default="mic_data")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.outdir, "%s_%s" % (args.label, stamp))

    x = record(args.seconds, args.device)
    print("Peak level: %.3f (1.0 = clipping), RMS: %.4f" %
          (np.max(np.abs(x)), np.sqrt(np.mean(x ** 2))))

    wavfile.write(base + ".wav", RATE, (x * 32767).astype(np.int16))

    rows = extract_features(x)
    with open(base + "_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "peak_hz"] +
                   ["band_%02d" % i for i in range(N_BANDS)] + ["label"])
        for t, peak, bands in rows:
            w.writerow(["%.3f" % t, "%.1f" % peak] +
                       ["%.2f" % b for b in bands] + [args.label])
    print("Saved %s.wav and %s_features.csv (%d windows)" %
          (base, base, len(rows)))

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 6))
        ax[0].plot(np.arange(len(x)) / RATE, x)
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("amplitude")
        ax[1].specgram(x, NFFT=WINDOW, Fs=RATE, noverlap=WINDOW - HOP)
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("frequency (Hz)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()