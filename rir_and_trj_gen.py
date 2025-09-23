import numpy as np
import pyroomacoustics as pra


def estimate_max_order(room_sz, T60, c=343.0):
    Tmax = 0.8 * T60
    avg_dist = np.mean(room_sz)
    return int(np.ceil((c * Tmax) / avg_dist))

def center_to_corner(pos, room_sz):
    """
    Convert positions from centered coordinate system to corner-origin.
    """
    return pos + np.array(room_sz) / 2

def generate_rirs_pyroom(room_sz, all_pos_src, index, pos_rcv, T60, fs, simulate_trj=False):
    """
    Generates Room Impulse Responses (RIRs) using pyroomacoustics for multiple source positions.

    :param room_sz: Room dimensions [L, W, H]
    :param all_pos_src: All source positions [3, num_sources] or [3, num_frames] if simulate_trj
    :param index: Index to select a specific frame (when simulate_trj=False)
    :param pos_rcv: Receiver positions [num_mics, 3]
    :param T60: Reverberation time
    :param fs: Sampling rate
    :param simulate_trj: If True, simulate all positions in trajectory
    :return: List of RIRs: rirs[mic][src]
    """
    # Determine source positions
    if simulate_trj:
        pos_src = all_pos_src.T  # shape: (N_sources, 3)
    else:
        pos_src = all_pos_src[:, index].reshape(-1, 3)  # shape: (N_sources, 3)

    # Shift positions from center-based to corner-based coordinates
    pos_src_shifted = pos_src
    pos_rcv_shifted = pos_rcv
    
    # print("pos_src min/max:", np.min(pos_src, axis=0), np.max(pos_src, axis=0))
    # print("room size:", room_sz)
    
    # print("Orig source positions:\n", pos_src)
    # print("Shifted source positions:\n", pos_src_shifted)
    # print("Room size:", room_sz)

    # reflection = pra.inverse_sabine(T60, room_sz)
    # absorption = 1 - reflection
    reflection, _ = pra.inverse_sabine(T60, room_sz)
    absorption = 1 - reflection

    room = pra.ShoeBox(
        room_sz,
        fs=fs,
        materials=pra.Material(absorption),
        max_order= 0,  # max_order= estimate_max_order(room_sz, T60) 30
        ray_tracing=True,
        air_absorption=True,
    )

    # Add source(s)
    for src in pos_src_shifted:
        room.add_source(src)

    # Add microphone array (shape must be 2 x M)
    room.add_microphone_array(np.array(pos_rcv_shifted).T)

    # Compute RIRs
    room.compute_rir()

    # Enforce fixed RIR length
    rir_len = int(fs * T60 * 0.8)
    for m in range(len(room.rir)):
        for s in range(len(room.rir[m])):
            rir = room.rir[m][s]
            if len(rir) > rir_len:
                room.rir[m][s] = rir[:rir_len]
            else:
                room.rir[m][s] = np.pad(rir, (0, rir_len - len(rir)))

    return room.rir  # rirs[mic][source]

