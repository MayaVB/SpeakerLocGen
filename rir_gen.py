
import numpy as np
import gpuRIR

def generate_rirs(room_sz, pos_src, pos_rcv, T60, fs):
    """
    Generates Room Impulse Responses (RIRs) for multiple source positions.
    
    :param room_sz: Dimensions of the room [length, width, height]
    :param pos_src: Array of source positions [[x1, y1, z1], [x2, y2, z2], ...]
    :param pos_rcv: Array of receiver positions [[x1, y1, z1], [x2, y2, z2], ...]
    :param T60: Reverberation time
    :param fs: Sampling frequency
    :return: RIRs for each source position
    """
    mic_pattern = "omni"  # Receiver polar pattern
    beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # Reflection coefficients
    Tmax = T60 * 0.8  # Time to stop the simulation [s]
    Tdiff = 0.1  # ISM stops here and the Gaussian noise with an exponential envelope starts
    nb_img = gpuRIR.t2n(T60, room_sz)  # Number of image sources in each dimension

    all_RIRs = []
    for src_pos in pos_src.T:
        RIR = gpuRIR.simulateRIR(room_sz, beta, src_pos, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, mic_pattern=mic_pattern)
        all_RIRs.append(RIR)
    
    return all_RIRs, RIR