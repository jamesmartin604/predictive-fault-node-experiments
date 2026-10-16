r"""
train_baseline.py - train simple classifiers on FFT features from labelled WAVs.

Trains "classroom" (every classroom_*.wav) against "quiet" (every quiet_*.wav).

Setup:
    py -m pip install numpy scipy scikit-learn matplotlib joblib

Usage:
    py train_baseline.py                      # reads .\mic_data\classroom_*.wav, quiet_*.wav
    py train_baseline.py --folder mic_data
    py train_baseline.py --classes classroom quiet coffee_shop

Labels come from the file names: the trailing recording number or timestamp
is stripped, so quiet_1.wav and quiet_2.wav are both class "quiet".

Evaluation holds out one WHOLE RECORDING at a time (leave-one-recording-out).
Never split windows from the same recording between train and test: 50%
overlapping windows are near-duplicates, so a random split gives inflated
accuracy that says nothing about new recordings.

Outputs (in ./analysis_out):
    confusion.png   confusion matrices for both models
    model.joblib    final model (logistic regression) trained on all recordings
"""
import argparse
import glob
import os
import re

import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

WINDOW = 1024
HOP = 512
N_BANDS = 16
FEATURE_NAMES = (["rms_db", "peak_hz", "centroid_hz"] +
                 ["band_%02d" % i for i in range(N_BANDS)])

# Classes to train on. Every <class>_<n>.wav in the folder is used, so new
# recordings are picked up automatically.
CLASSES = ["classroom", "quiet"]


def label_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"(_\d{8}_\d{6})?(_\d+)?$", "", stem)


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


def window_features(x, rate):
    """One row per window: rms_db, peak_hz, centroid_hz, 16 band energies (dB)."""
    win = np.hanning(WINDOW)
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / rate)
    edges = np.linspace(0, len(freqs), N_BANDS + 1, dtype=int)
    rows = []
    for start in range(0, len(x) - WINDOW + 1, HOP):
        raw = x[start:start + WINDOW]
        spec = np.abs(np.fft.rfft(raw * win)) ** 2
        rms_db = 10 * np.log10(np.mean(raw ** 2) + 1e-12)
        peak_hz = freqs[np.argmax(spec[1:]) + 1]
        centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-20)
        bands = [10 * np.log10(spec[edges[i]:edges[i + 1]].sum() + 1e-12)
                 for i in range(N_BANDS)]
        rows.append([rms_db, peak_hz, centroid] + bands)
    return np.array(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="mic_data")
    p.add_argument("--outdir", default="analysis_out")
    p.add_argument("--classes", nargs="+", default=CLASSES,
                   help="class names to train on (default: %(default)s)")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.folder, "*.wav"))
                   if label_of(f) in args.classes)
    if not files:
        p.error("no %s recordings found in %s" % ("/".join(args.classes), args.folder))

    X, y, groups, rates = [], [], [], set()
    for path in files:
        x, rate = load_wav(path)
        rates.add(rate)
        feats = window_features(x, rate)
        if len(feats) == 0:
            p.error("%s is shorter than one window (%d samples)" % (path, WINDOW))
        X.append(feats)
        y += [label_of(path)] * len(feats)
        groups += [os.path.basename(path)] * len(feats)
    X = np.vstack(X)
    y = np.array(y)
    groups = np.array(groups)

    if len(rates) > 1:
        print("WARNING: files have different sample rates %s; band edges differ." % rates)

    print("Recordings and windows per class:")
    classes = sorted(set(y))
    for c in classes:
        recs = sorted(set(groups[y == c]))
        print("  %-14s %d recordings, %5d windows" % (c, len(recs), np.sum(y == c)))
        if len(recs) < 2:
            print("    WARNING: only one recording, so it can't be held out fairly.")
    if len(classes) < 2:
        p.error("need at least 2 classes")

    # class_weight="balanced": unequal numbers of recordings per class would
    # otherwise bias the models toward the bigger class.
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision tree (depth 4)": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=0),
    }

    os.makedirs(args.outdir, exist_ok=True)
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)

    for ax, (name, model) in zip(axes[0], models.items()):
        pred = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut())
        acc = np.mean(pred == y)
        print("\n=== %s: leave-one-recording-out accuracy %.1f%% ===" % (name, 100 * acc))
        for g in sorted(set(groups)):
            m = groups == g
            print("  %-34s -> %.1f%% windows correct" % (g, 100 * np.mean(pred[m] == y[m])))

        cm = confusion_matrix(y, pred, labels=classes)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        print("Confusion matrix (rows = true, cols = predicted):")
        print("  classes:", classes)
        print(cm)

        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, "%d%%" % round(100 * cmn[i, j]), ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("%s\n%.1f%% accuracy" % (name, 100 * acc))

        if isinstance(model, DecisionTreeClassifier):
            model.fit(X, y)
            order = np.argsort(model.feature_importances_)[::-1][:5]
            print("Top features:", ", ".join(
                "%s (%.2f)" % (FEATURE_NAMES[i], model.feature_importances_[i]) for i in order))

    plt.tight_layout()
    out = os.path.join(args.outdir, "confusion.png")
    fig.savefig(out, dpi=150)
    print("\nSaved %s" % out)

    # Final model: fit on all 7 recordings and save for later use.
    final = models["Logistic regression"]
    final.fit(X, y)
    model_path = os.path.join(args.outdir, "model.joblib")
    joblib.dump({"model": final, "classes": list(final.classes_),
                 "feature_names": FEATURE_NAMES,
                 "window": WINDOW, "hop": HOP, "n_bands": N_BANDS}, model_path)
    print("Saved %s" % model_path)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()