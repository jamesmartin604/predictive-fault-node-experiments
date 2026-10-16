# Predictive Fault Node: Experiments

Sample tests and prototype scripts for my Final Year Project. **The main project lives in a separate repository, [`predictive-fault-node`](https://github.com/<your-username>/predictive-fault-node).** This repo holds the early laptop-side experiments I ran while waiting for the hardware to arrive.

> **Status:** throwaway prototyping. The scripts here run on placeholder sound recordings, not real fan-fault data. Anything that proves useful will be moved into the main repo.

## What the main project is

A battery-powered ESP32 sensor node that predicts early-stage mechanical faults in a small motor or fan, instead of only detecting a fault after it has happened. It combines vibration (MPU6050) and sound (INMP441 microphone), extracts frequency-domain features with an FFT, and is designed to run the trained model on-device so the node only transmits an alert when it flags a problem. See the main repo for the hardware, firmware and write-up.

## Why this repo exists

To try out the data pipeline before the hardware is available:

1. Record labelled audio from a laptop microphone
2. Compute FFT spectra and simple features
3. Train a basic classifier and check it can separate placeholder classes

The same steps will later run on real vibration and sound recordings from the fan rig.

## Contents

| File | Purpose |
|---|---|
| `audio-extraction.py` | Records audio from a laptop microphone (16 kHz mono), saves a WAV and a per-window FFT feature CSV |
| `analyze_fft.py` | Loads WAV (sound) and CSV (vibration) files, plots spectra, and extracts simple features (peak, centroid, band power, rotation-frequency peaks) |
| `train_baseline.py` | Trains logistic regression and a decision tree on window features, evaluated by leaving out one whole recording at a time |

Recordings go in `mic_data/`; plots and summaries are written to `analysis_out/`.

## Setup

Requires Python 3.10+.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Record a labelled sample (list inputs with `--list-devices`, pick one with `--device`):

```
python audio-extraction.py --label quiet --seconds 20 --plot
```

Plot spectra and extract features:

```
python analyze_fft.py --audio mic_data\quiet_1.wav mic_data\snapping_1.wav
```

Train and evaluate the baseline classifier (class names come from file names, so `quiet_1.wav` is class `quiet`):

```
python train_baseline.py --folder mic_data
```

## Findings so far

- **Evaluation is by recording, not by window.** Neighbouring windows overlap by 50% and are near-duplicates, so splitting windows from the same recording across train and test would inflate accuracy.
- **First run was at chance level** (about 39 to 49% on two classes) using two quiet and two finger-snap recordings. Snaps are transients, so most windows in a "snapping" file contain no snap and the window-level labels are mostly wrong. The decision tree's top feature was loudness (`rms_db`), which suggests the files differ in microphone gain more than in the sound.
- **Next:** record steady sounds (for example fan off vs on), keep microphone distance constant, turn off laptop audio enhancements, and record at least 5 separate recordings per class before trusting any score.

## Limitations

- Laptop microphones apply automatic gain and noise suppression, and their frequency response differs from the INMP441, so these recordings test the code, not the final sensor.
- The vibration path in `analyze_fft.py` has only been checked for file format. There is no real accelerometer data yet.
- Data from the Wokwi simulator is not real vibration data.

## Author

James Martin, Final Year Project, BSc Computer Science and Software Engineering, Maynooth University (2026-27).