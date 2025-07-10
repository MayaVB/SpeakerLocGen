import numpy as np
import matlab.engine
from scipy.io import wavfile
from scipy.io.wavfile import write as wavwrite

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Add the MEX function directory
eng.addpath('/home/dsi/mayavb/PythonProjects/projet_year4/', nargout=0)
eng.addpath('/home/dsi/mayavb/Signal-Generator', nargout=0)

# Verify MATLAB can find the function
mex_path = eng.which('signal_generator', nargout=1)
print(f"MATLAB found signal_generator at: {mex_path}")

# ✅ Fix Inputs
fs = float(16000)  # Sampling frequency (Hz)
duration = 2  # Seconds
samples = int(fs * duration)  # Total samples

# 🔹 Ensure `in_signal` is a **row vector** (1, samples)
in_signal_orig = np.random.randn(1, samples)  # (1, N) → Row vector
in_signal_orig = matlab.double(in_signal_orig.tolist())  # Convert to MATLAB format

wav_path = "LDC93S6A.wav"
fs_wav, wav_data = wavfile.read(wav_path)  # fs_wav is the sample rate, wav_data is the audio signal
in_signal = matlab.double(wav_data.tolist())  # Convert to MATLAB format
samples = int(len(wav_data))

# Room properties
c = float(343)  # Speed of sound (m/s)
L = matlab.double([4, 5, 6])  # Room dimensions (1x3)

# Source path (N, 3)
s_path = np.random.uniform(2, 3, size=(samples, 3))  # (time, xyz)
s_path = matlab.double(s_path.tolist())

# # Receiver path (N, 3, M)
M = 2  # Number of microphones
fixed_position = np.array([[1.5, 2.5-0.1, 3], [1.5, 2.5+0.1, 3]])  # Example positions for two mics
r_path = np.tile(fixed_position.T, (samples, 1, 1))  # Repeat over time dimension
r_path = matlab.double(r_path.tolist())

# Reflection coefficients (1x6)
# beta = matlab.double([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])  # (1,6) OR single value for T_60
beta = matlab.double(0.2)  # rt60

# Other parameters
nsamples = float(1024)  # RIR length in samples
mtype = 'o'  # Omnidirectional mic
order = float(2)  # Reflection order
dim = float(3)  # Room dimension (3D)
orientation = matlab.double([0, 0])  # Azimuth & Elevation (1x2)
hp_filter = False  # Disable high-pass filter

# ✅ Run the MEX function with corrected input shape
try:
    out, beta_hat = eng.signal_generator(
        in_signal, c, fs, r_path, s_path, L, beta, nsamples, mtype, order, dim, orientation, hp_filter, nargout=2
    )

    print("✅ MEX function executed successfully!")
    print(f"Output Shape: {np.array(out).shape}")
    print(f"Estimated Beta: {beta_hat}")

except matlab.engine.MatlabExecutionError as e:
    print("❌ Error executing MEX function:", str(e))

# Quit MATLAB engine
eng.quit()



# Convert MATLAB output to NumPy array
out_np = np.array(out).flatten()

# Optional: Normalize to -1.0 to 1.0 for safety
max_val = np.max(np.abs(out_np))
if max_val > 1.0:
    out_np = out_np / max_val

# Convert to 16-bit PCM format
out_int16 = np.int16(out_np * 32767)

# Save as .wav
output_path = "signal_generator_output.wav"
wavwrite(output_path, int(fs), out_int16)

print(f"🎵 Output saved to: {output_path}")