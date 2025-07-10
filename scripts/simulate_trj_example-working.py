#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example for simulating the recording of a moving source with a microphone array.
You need to have a 'LDC93S6A.wav' audio file as input.
The script will generate:
    - source_signal.jpeg: Plot of the original source signal
    - filtered_signal_chan0.wav: Simulated mic recording (channel 0)
    - filtered_signal_chan0.jpeg: Plot of the simulated output
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import gpuRIR

# Disable mixed precision (default is fine too)
gpuRIR.activateMixedPrecision(False)

# Load source signal
fs, source_signal = wavfile.read('LDC93S6A.wav')

# # Normalize and convert to float32 if needed
# if source_signal.dtype == np.int16:
#     source_signal = source_signal.astype(np.float32) / 32768.0
# elif source_signal.dtype == np.int32:
#     source_signal = source_signal.astype(np.float32) / 2147483648.0

# # Apply fade-in and fade-out to avoid sharp transients
# fade_len = int(0.05 * fs)  # 50ms fade
# window = np.ones_like(source_signal)
# window[:fade_len] = np.linspace(0, 1, fade_len)
# window[-fade_len:] = np.linspace(1, 0, fade_len)
# source_signal *= window

# Plot source signal
plt.figure(figsize=(10, 6))
plt.plot(source_signal, linestyle='-', color='b')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Source Signal")
plt.tight_layout()
plt.savefig('source_signal.jpeg', format='jpeg', dpi=300)
plt.close()

# Define room and array parameters
room_sz = [3, 4, 2.5]  # Room dimensions [m]
traj_pts = 64
pos_traj = np.tile(np.array([0.0, 3.0, 1.0]), (traj_pts, 1))
pos_traj[:, 0] = np.linspace(0.1, 2.9, traj_pts)  # X axis movement

nb_rcv = 2
pos_rcv = np.array([[1.4, 1, 1.5], [1.6, 1, 1.5]])
# orV_rcv = np.array([[-1, 0, 0], [1, 0, 0]])
mic_pattern = "omni"

# Reverberation parameters
T60 = 0.4         # RT60
att_diff = 10.0   # Diffuse model start
att_max = 50.0    # Diffuse model end

# Compute simulation parameters
beta = gpuRIR.beta_SabineEstimation(room_sz, T60) # estimate beta is reflection coeff using Sabine reverberation formula 
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60) # point where distinct echos (IS model) is replaced by a diffuse reverberation tail
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
nb_img = gpuRIR.t2n(Tdiff, room_sz)

# Simulate RIRs and apply to trajectory
# RIRs = gpuRIR.simulateRIR(
#     room_sz, beta, pos_traj, pos_rcv, nb_img, Tmax, fs,
#     Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
# )
RIRs = gpuRIR.simulateRIR(
    room_sz, beta, pos_traj, pos_rcv, nb_img, Tmax, fs,
    Tdiff=Tdiff, mic_pattern=mic_pattern
)
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)

# Normalize output to avoid clipping
max_val = np.max(np.abs(filtered_signal))
if max_val > 0:
    filtered_signal /= max_val

# Save channel 0
wavfile.write('filtered_signal_chan0.wav', fs, filtered_signal[:, 0])

# Plot channel 0
plt.figure(figsize=(10, 6))
plt.plot(filtered_signal[:, 0], linestyle='-', color='b')
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.title("Filtered Signal (Channel 0)")
plt.tight_layout()
plt.savefig('filtered_signal_chan0.jpeg', format='jpeg', dpi=300)
plt.close()
