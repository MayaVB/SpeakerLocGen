import matlab.engine
import matlab
import numpy as np

def signal_generator_wrapper(fs, in_data, scene, hop=32):

    # --- 1. Normalize input ---
    in_data = in_data.astype(np.float64)
    in_data = in_data / np.max(np.abs(in_data))
    len_samples = len(in_data)

    # --- 2. Room setup ---
    c = 340.0
    L = matlab.double(list(scene['room_dim']))
    beta = matlab.double([scene['RT60']])
    nsamples = float(1024)
    order = float(2)
    mtype = 'o'

    # --- 3. Mic positions ---
    rp = np.array(scene['mic_pos'])  # (M, 3)
    M = rp.shape[0]

    # # --- 4. Source movement ---
    sp = np.array(scene['src_pos'])[:, 0]       # Start position
    stop = np.array(scene['src_pos'])[:, -1]    # End position

    sp_pathV1 = np.zeros((len_samples, 3))
    for ii in range(0, len_samples, hop):
        end = min(ii + hop, len_samples)
        frac = ii / len_samples
        sp_pathV1[ii:end, :] = sp + frac * (stop - sp)

    # option B
    src_traj = np.array(scene['src_pos']).T  # shape: [N, 3]
    n_rirs = src_traj.shape[0]

    # Interpolate to match total number of samples
    sp_pathV2 = np.zeros((len_samples, 3))
    for i in range(3):  # for x, y, z
        sp_pathV2[:, i] = np.interp(
            np.linspace(0, n_rirs - 1, len_samples),
            np.arange(n_rirs),
            src_traj[:, i]
        )
    sp_path = sp_pathV2

    # --- 5. Receiver path (static) ---
    r_path = np.tile(rp.T, (len_samples, 1, 1))  # (N, 3, M)
    r_path = matlab.double(r_path.tolist())

    # --- 6. Convert to MATLAB types ---
    in_signal = matlab.double(in_data.reshape(1, -1).tolist())
    sp_path_mat = matlab.double(sp_path.tolist())

    # --- 7. Call MATLAB MEX function ---
    eng = matlab.engine.start_matlab()
    eng.addpath('/home/dsi/mayavb/PythonProjects/projet_year4/', nargout=0)
    eng.addpath('/home/dsi/mayavb/Signal-Generator', nargout=0)

    try:
        out, beta_hat = eng.signal_generator(
            in_signal, c, float(fs), r_path, sp_path_mat,
            L, beta, nsamples, mtype, order, float(3), matlab.double([0, 0]),
            False, nargout=2
        )
    except matlab.engine.MatlabExecutionError as e:
        print("❌ MEX error:", str(e))
        eng.quit()
        raise

    eng.quit()

    out_np = np.array(out)  # (M, N)
    out_np = out_np / np.max(np.abs(out_np))
    return out_np
