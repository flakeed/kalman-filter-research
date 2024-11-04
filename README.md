# Kalman Filter Research
**Status**: *Completed*

## Project Overview
This repository contains an implementation and research of the Kalman filter algorithm for signal processing and noise reduction. The project demonstrates how different parameters affect the filter's performance in smoothing noisy signals.

The Kalman filter is a powerful mathematical tool used for:
- Filtering measurement noise from signals
- Improving measurement accuracy
- State estimation in dynamic systems

This implementation specifically focuses on applying the Kalman filter to a sinusoidal signal with added Gaussian noise to demonstrate its noise reduction capabilities.
![image](https://github.com/user-attachments/assets/2305a247-e985-496c-93b7-9f5c01f1a3ea)

## Features

- Implementation of a basic Kalman filter class in Python
- Visualization of filter performance using matplotlib
- Analysis of various filter parameters:
  - Process noise covariance (Q)
  - Measurement noise covariance (R)
  - Initial estimation error covariance (P)
  - State transition matrix (F)
  - Measurement matrix (H)
- Comparison of noise variance before and after filtering
- Interactive plots showing:
  - Original signal
  - Noisy measurements
  - Filtered signal

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

1. Run the main script:
```bash
python kalman_filter.py
```

2. The script will:
   - Generate a sinusoidal signal with noise
   - Apply the Kalman filter
   - Display plots comparing the original, noisy, and filtered signals
   - Print noise variance statistics
