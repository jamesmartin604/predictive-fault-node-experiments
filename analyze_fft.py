"""
analyze_fft.py - FFT spectra and simple features for recorded samples.

Setup:
    py -m pip install numpy scipy pandas matplotlib

Usage (labels are taken from the file names):
    py analyze_fft.py --audio mic_data\\quiet_2026....wav mic_data\\snap_2026....wav
    py analyze_fft.py --accel normal_01.csv imbalance_1.csv --rot-hz 50
    py analyze_fft.py --audio a.wav b.wav --accel c.csv d.csv --rot-hz 50

Audio:  16 kHz WAV files from record_mic.py.
Accel:  CSV files from logger.py (columns t_us,ax,ay,az,label). The sample
        rate is estimated from t_us, and the data is split into blocks
        wherever there is a timing gap (between printed blocks), so each
        spectrum only uses continuous data.

Outputs (in ./analysis_out):
    spectra.png            spectrum overlay, one panel per sensor type
    features_summary.csv   one row per input file
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

AUDIO_BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000),
               (2000, 4000), (4000, 8000)]
VIB_BANDS = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 500)]


# ---------------------------------------------------------------- loading
def load_wav(path):
    rate, data = wavfile.read(path)
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if data.dtype == np.int16:
        x /= 32768.0
    elif data.dtype == np.int32:
        x /= 2147483648.0
    return x, rate


def load_accel(path):
    """Return (list of Nx3 blocks in g, estimated sample rate in Hz)."""
    df = pd.read_csv(path, comment="#")
    t = df["t_us"].to_numpy(dtype=np.float64)
    dt = np.diff(t)
    med = np.median(dt[dt > 0])
    fs = 1e6 / med
    cuts = np.where(dt > 5 * med)[0] + 1          # gaps between blocks
    xyz = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
    blocks = [xyz[i] for i in np.split(np.arange(len(t)), cuts) if len(i) >= 128]
    if not blocks:
        raise ValueError("%s: no continuous block of 128+ samples" % path)
    return blocks, fs


# ---------------------------------------------------------------- spectra
def audio_spectrum(x, fs):
    return welch(x, fs=fs, nperseg=1024, detrend="constant")


def accel_spectrum(blocks, fs):
    """Average PSD over blocks; PSDs of the three axes are summed."""
    psds = []
    for b in blocks:
        total = 0
        for k in range(3):
            f, p = welch(b[:, k], fs=fs, nperseg=128, detrend="constant")
            total = total + p
        psds.append(total)
    return f, np.mean(psds, axis=0)


def accel_rms(blocks):
    sq = [np.sum((b - b.mean(axis=0)) ** 2, axis=1) for b in blocks]
    return float(np.sqrt(np.mean(np.concatenate(sq))))


# ---------------------------------------------------------------- features
def db(v):
    return 10 * np.log10(v + 1e-20)


def band_power(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return 0.0
    return float(np.sum(psd[m]) * (freqs[1] - freqs[0]))


def extract_features(freqs, psd, bands, rms, rot_hz=None):
    feats = {
        "rms": rms,
        "peak_hz": float(freqs[np.argmax(psd[1:]) + 1]),   # skip DC bin
        "centroid_hz": float(np.sum(freqs * psd) / (np.sum(psd) + 1e-20)),
        "total_db": float(db(band_power(freqs, psd, 0, freqs[-1] + 1))),
    }
    for lo, hi in bands:
        if lo < freqs[-1]:
            feats["band_%d_%d_db" % (lo, hi)] = float(db(band_power(freqs, psd, lo, hi)))
    if rot_hz:
        df = freqs[1] - freqs[0]
        for name, f0 in (("rot_1x", rot_hz), ("rot_2x", 2 * rot_hz)):
            w = max(0.1 * f0, 2 * df)
            if f0 + w < freqs[-1]:
                m = (freqs >= f0 - w) & (freqs <= f0 + w)
                feats[name + "_db"] = float(db(psd[m].max()))
    return feats


# ---------------------------------------------------------------- main
def label_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", nargs="*", default=[], help="WAV files")
    p.add_argument("--accel", nargs="*", default=[], help="accelerometer CSV files")
    p.add_argument("--rot-hz", type=float, default=None,
                   help="fan rotation frequency (RPM / 60), enables 1x/2x peak features")
    p.add_argument("--outdir", default="analysis_out")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if not args.audio and not args.accel:
        p.error("give at least one --audio or --accel file")
    os.makedirs(args.outdir, exist_ok=True)

    panels = [k for k, files in (("Sound", args.audio), ("Vibration", args.accel)) if files]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 4 * len(panels)), squeeze=False)
    axes = {name: axes[i][0] for i, name in enumerate(panels)}
    rows = []

    for path in args.audio:
        x, fs = load_wav(path)
        f, psd = audio_spectrum(x, fs)
        feats = extract_features(f, psd, AUDIO_BANDS, float(np.sqrt(np.mean(x ** 2))))
        rows.append({"source": path, "kind": "audio", "label": label_of(path), **feats})
        axes["Sound"].plot(f, db(psd), label=label_of(path))

    for path in args.accel:
        blocks, fs = load_accel(path)
        f, psd = accel_spectrum(blocks, fs)
        feats = extract_features(f, psd, VIB_BANDS, accel_rms(blocks), args.rot_hz)
        rows.append({"source": path, "kind": "accel", "label": label_of(path), **feats})
        axes["Vibration"].plot(f, db(psd), label="%s (fs~%.0f Hz)" % (label_of(path), fs))
        if args.rot_hz:
            axes["Vibration"].axvline(args.rot_hz, color="gray", ls=":", lw=1)

    for name, ax in axes.items():
        ax.set_title("%s spectrum" % name)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "spectra.png"), dpi=150)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "features_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.2f}".format):
        print(out.drop(columns=["source"]).to_string(index=False))
    print("\nSaved spectra.png and features_summary.csv in %s" % args.outdir)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()