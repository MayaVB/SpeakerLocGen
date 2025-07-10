import numpy as np
import matlab.engine
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wavwrite

# === 1. Load anechoic input signal ===
wav_path = "LDC93S6A.wav"
fs, in_data = wavfile.read(wav_path)
if in_data.ndim > 1:
    in_data = in_data[0, :]  # use only 1st channel if stereo

# Take 4 seconds, duplicated to 2 channels like in MATLAB
# in_data = in_data[:4 * fs]
# in_data = np.concatenate([in_data, in_data])  # shape: (2N,)
in_data = in_data.astype(np.float64) / np.max(np.abs(in_data))  # Normalize
len_samples = len(in_data)

# === 2. Room setup ===
c = float(340)
L = matlab.double([4, 5, 6])
beta = matlab.double(0.2)  # reverberation coefficient
nsamples = float(1024)
order = float(2)
mtype = 'o'

# Receiver positions
rp = np.array([[1.5, 2.4, 3.0],
               [1.5, 2.6, 3.0]])
M = rp.shape[0]
cp = np.mean(rp, axis=0)

# === 3. Create moving source path ===
hop = 32
sp = np.array([2.5, 0.5, 3.0])  # initial source position
stop = np.array([2.5, 4.5, 3.0])  # linear movement

sp_path = np.zeros((len_samples, 3))
rp_path = np.zeros((len_samples, 3, M))

for ii in range(0, len_samples, hop):
    end = min(ii + hop, len_samples)
    frac = ii / len_samples
    new_sp = sp + frac * (stop - sp)
    sp_path[ii:end, :] = new_sp

    for m in range(M):
        rp_path[ii:end, :, m] = rp[m]

# === 4. Convert to MATLAB format ===
eng = matlab.engine.start_matlab()
eng.addpath('/home/dsi/mayavb/PythonProjects/projet_year4/', nargout=0)
eng.addpath('/home/dsi/mayavb/Signal-Generator', nargout=0)

# in_signal = matlab.double(in_data.tolist())  # shape (2, N)
in_signal = matlab.double(in_data.reshape(1, -1).tolist())
sp_path_mat = matlab.double(sp_path.tolist())
# rp_path_mat = matlab.double(rp_path.tolist())
rp_path_mat = matlab.double(rp_path.reshape(len_samples, 3 * M).tolist())

fixed_position = np.array([[1.5, 2.5-0.1, 3], [1.5, 2.5+0.1, 3]])  # Example positions for two mics
r_path = np.tile(fixed_position.T, (len_samples, 1, 1))  # shape (N, 3, M)
r_path = matlab.double(r_path.tolist())  # keep as-is


# === 5. Call MEX ===
try:
    out, beta_hat = eng.signal_generator(
        in_signal, c, float(fs), r_path, sp_path_mat,
        L, beta, nsamples, mtype, order, float(3), matlab.double([0, 0]),
        False, nargout=2
    )
    print("✅ MEX executed successfully!")
except matlab.engine.MatlabExecutionError as e:
    print("❌ MEX error:", str(e))
    eng.quit()
    exit()

# === 6. Convert output and save as .wav ===
out_np = np.array(out)  # shape: (M, N)
out_np = out_np / np.max(np.abs(out_np))  # Normalize to [-1, 1]
out_int16 = np.int16(out_np.T * 32767)  # shape: (N, M)

wavwrite('output_mics.wav', int(fs), out_int16)

# === 7. Plot ===
t = np.arange(len_samples) / fs
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, in_data)
plt.title("Input signal (channel 1)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
for m in range(M):
    plt.plot(t, out_np[m], label=f'Mic {m+1}')
plt.title("Output mic signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("mic_signals_plot.png", dpi=300, bbox_inches='tight')
plt.close()


# === 8. Quit MATLAB ===
eng.quit()
