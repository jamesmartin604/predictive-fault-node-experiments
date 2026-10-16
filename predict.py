"""
predict.py - classify a WAV file using the model saved by train_baseline.py.

Usage:
    py predict.py some_recording.wav
    py predict.py a.wav b.wav c.wav
    py predict.py some_recording.wav --model analysis_out\model.joblib

Keep this file in the same folder as train_baseline.py.
"""
import argparse
import os

import joblib
import numpy as np

from train_baseline import load_wav, window_features


def classify(model, path):
    x, rate = load_wav(path)
    feats = window_features(x, rate)
    if len(feats) == 0:
        raise ValueError("file is shorter than one analysis window")

    pred = model.predict(feats)                  # one label per window
    proba = model.predict_proba(feats)           # one probability row per window
    classes = list(model.classes_)

    # Per-recording answer: average the window probabilities, pick the best.
    mean_p = proba.mean(axis=0)
    best = int(np.argmax(mean_p))
    return classes[best], dict(zip(classes, mean_p)), pred, rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="one or more .wav files")
    p.add_argument("--model", default=os.path.join("analysis_out", "model.joblib"))
    args = p.parse_args()

    saved = joblib.load(args.model)
    model = saved["model"]

    for path in args.files:
        label, probs, window_preds, rate = classify(model, path)
        print("\n%s  (%d Hz, %d windows)" % (path, rate, len(window_preds)))
        print("  Prediction: %s" % label)
        for c, pr in sorted(probs.items(), key=lambda kv: -kv[1]):
            print("    %-12s %5.1f%%" % (c, 100 * pr))
        agree = np.mean(window_preds == label)
        print("  %.0f%% of windows agree with this label" % (100 * agree))


if __name__ == "__main__":
    main()