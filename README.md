# Aequor

**A Real-Time Adaptive Cursor Stabilization and Visualization System for Hand Tremors**

> *"Intent is like still water, while tremor is like wind blowing ripples. We cannot stop the wind from blowing, but we can build a harbor between fingertips and screens to calm the waves."*

---

## Overview

**Aequor** (derived from the Latin term for *calm, still water*) is a digital cursor stabilization system specifically designed for individuals experiencing physiological or pathological hand tremors (4–12 Hz).

In human-computer interaction, hand tremors often lead to cursor drift and misclicks. Aequor models the user's raw input trajectory as a combination of intended signal and physical noise. Through real-time sampling, frequency-domain analysis, and adaptive state estimation, it precisely extracts the user's genuine **intent** from unstable physical movements, delivering smooth, low-latency system-level cursor control.

---

## Core Features

* **Adaptive Composite Filtering**:
Integrates a Least Mean Squares (LMS) adaptive notch filter with a Kalman state estimator. It dynamically tracks and attenuates the dominant 4–12 Hz tremor frequency, striking an optimal balance between filtering smoothness and real-time responsiveness.
* **Discrete Wavelet Denoising**:
Optionally incorporates a `db4` discrete wavelet transform (DWT) soft-thresholding module to effectively remove unstructured, high-frequency Gaussian noise.
* **Click-Phase Dynamic Suppression**:
Addresses severe cursor displacement caused by physical squeezing during mouse presses. The system automatically tightens state-estimation constraints upon detecting click events, significantly improving click accuracy.
* **Real-Time Spectrum Diagnostics Console**:
Built on PyQt5 and pyqtgraph, providing millisecond-level visual feedback for cursor trajectories, frequency response charts, and dominant frequency tracking.
* **System-Level Cursor Injection**:
Provides a low-latency coordinate write-back mechanism that overrides the operating system's underlying input stream, ensuring seamless compatibility with desktop applications.

---

## Quick Start

### 1. Prerequisites

Python 3.8 or higher is required. Clone the repository and install the dependencies:

```bash
git clone https://github.com/austinwangdev319/Aequor.git
cd Aequor
pip install -r requirements.txt

```

### 2. Launch the Application

```bash
python aequor_app.py

```

### 3. Basic Usage

1. Click **Start Filtering** on the control console to activate the processing engine.
2. Check **Apply to system cursor** to enable system-wide cursor takeover.
3. Adjust the **Sensitivity** slider to match your tremor characteristics and find the ideal balance between responsiveness and smoothness.

---

## System Architecture & Algorithm Pipeline

```
[Raw Input Trajectory (X, Y)]
              │
              ▼
┌───────────────────────────┐      ┌────────────────────────────────────┐
│   Sliding-Window FFT      │ ───> │     LMS Adaptive Notch Filter      │
│ (Dominant Frequency Est.) │      │ (4–12 Hz Tremor Energy Attenuation)│
└───────────────────────────┘      └────────────────────────────────────┘
                                                     │
                                                     ▼
                                   ┌────────────────────────────────────┐
                                   │      Kalman State Estimator        │
                                   │ (State Pred. & Process Noise Adapt)│
                                   └────────────────────────────────────┘
                                                     │
                                                     ▼
┌───────────────────────────┐      ┌───────────────────────────────────┐
│   Click Event Trigger     │ ───> │     Discrete Wavelet Denoising    │
│  (Tightened Constraints)  │      │     (DWT db4 Soft-Thresholding)   │
└───────────────────────────┘      └───────────────────────────────────┘
                                                     │
                                                     ▼
                                     [Smoothed System Cursor Output]

```

---

## Technical Details

1. **Online Dominant Frequency Estimation**: Computes the Power Spectral Density (PSD) of input movement sequences using a sliding-window Fast Fourier Transform (FFT) to lock onto the peak tremor frequency.
2. **Adaptive Notch Filtering**: The LMS algorithm updates filter coefficients in real time based on the estimated dominant frequency, precisely eliminating tremor components while preserving intentional movement.
3. **Kalman State Estimation**: Dynamically adjusts the process noise covariance matrix ($Q$) based on cursor speed. It increases responsiveness during rapid movements and enforces stronger smoothing constraints when stationary or clicking.

---

## Limitations & Known Issues

* **Non-Uniform Event Sampling**: Due to OS event-loop mechanics, mouse input sampling intervals may fluctuate slightly, which could introduce minor microsecond-level latency under specific conditions.
* **System Permissions**: Injecting hardware coordinates into the OS input stream requires elevated permissions. On macOS or Windows with strict UAC settings, the application must be run as Administrator or granted Accessibility permissions.

---

## Future Roadmap

* [ ] **Auto-Profile Calibration**: Introduce an initial 5-second baseline sampling phase to automatically generate a personalized tremor frequency profile.
* [ ] **Multi-Harmonic Notch Filter Array**: Expand multi-peak detection capabilities to simultaneously suppress primary frequencies and secondary harmonics.
* [ ] **Kernel-Level Input Driver**: Develop Windows/Linux kernel-level input drivers to bypass user-space latency and deliver a completely native input experience.
