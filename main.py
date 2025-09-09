from scipy.signal import lfilter
import soundfile as sf
import numpy as np

import argparse
import os
import random

from scene_gen import select_random_speaker, generate_scenes, generate_rirs_pyroom, plot_scene_interactive
from scene_gen import plot_scene_interactive, calculate_doa

from utils import save_wavs, write_scene_to_file, save_data_h5py
from utils import get_trj_DOA, interpolate_angles, inject_pink_noise_bursts

from processing import add_ar1_noise_to_signals
# import matplotlib.pyplot as plt


def generate_rev_speech(args):
    """
    :param args: From Parser
    :return: scene_agg: scence list of dict containing info about the scenario generated
    """
    scene_agg = []
    clean_speech_dir = args.clean_speech_dir
    num_scenes = args.num_scenes
    snr = args.snr
    simulate_matlab_trj = args.simulate_matlab_trj
    
    if args.output_folder:
        save_rev_speech_dir = os.path.join(args.split, args.output_folder)
    else:
        save_rev_speech_dir = args.split
    
    # Generate reverberant speech files
    for scene_idx in range(num_scenes):
        
        selected_speaker, wav_files = select_random_speaker(clean_speech_dir)

        # Generate a scene
        scene = generate_scenes(args)[0]
        mics_num = len(scene['mic_pos'])
        
        # plot an interactive scene plot for each scenarion
        plot_scene_interactive(scene, save_rev_speech_dir, scene_idx)
                
        for index, _ in enumerate(scene['src_pos'][0]):
            curr_wav_file_path = random.choice(wav_files)

            if len(wav_files) < len(scene['src_pos'][0]):
                raise Exception("speaker wav files are lower then requested speaker location- not enough files!") 

            # check wav length
            s, fs = sf.read(curr_wav_file_path) 
            if len(s)/fs > args.minimum_sentence_len:
                wav_verified = True
            else:
                wav_verified = False

            while not wav_verified:

                # Read a clean file
                s, fs = sf.read(curr_wav_file_path)  
                if len(s)/fs < args.minimum_sentence_len or len(s)/fs > args.maximum_sentence_len:
                    curr_wav_file_path = random.choice(wav_files)
                else:
                    wav_verified = True

            print('Processing Scene %d/%d. Speaker: %s, wav file processed: %s, wav file number: %d.' % (
                scene_idx + 1, num_scenes, selected_speaker, os.path.basename(curr_wav_file_path), index + 1))

            # Generate RIR for the current source position for all mic positons
            if args.simulate_trj and not simulate_matlab_trj:
                from gpuRIR import simulateTrajectory
                from rir_gen import generate_rirs

                RIRs = generate_rirs(scene['room_dim'], scene['src_pos'], index, scene['mic_pos'], scene['RT60'], fs, args.simulate_trj)
                # RIRsV2 = generate_rirs_pyroom(scene['room_dim'], scene['src_pos'], index, scene['mic_pos'], scene['RT60'], fs, args.simulate_trj)

                rev_signals = simulateTrajectory(s, RIRs)
                rev_signals = rev_signals.T
                scene['DOA_az_trj'] = get_trj_DOA(scene, rev_signals)
                
            elif simulate_matlab_trj:
                if args.simulate_sinus:
                    T_frames   = int(np.ceil(len(s) / fs * args.fps))  # total frames given desired fps
                    radius     = np.linalg.norm(scene['src_pos'][0][index]   # keep original distance
                                                - scene['mic_pos'][0])       # mic[0] as pivot
                    center_deg = 60.0
                    amp_deg    = 20.0

                    # precompute sine angles in radians
                    angles_deg = center_deg + amp_deg * np.sin(2 * np.pi * np.arange(T_frames) / T_frames)
                    angles_rad = np.deg2rad(angles_deg)

                    # convert DOA angle -> Cartesian (XZ‑plane, y=0)  relative to room center
                    room_cx, room_cy, room_cz = np.array(scene['room_dim']) / 2
                    src_trj = np.zeros((T_frames, 3))
                    src_trj[:, 0] = room_cx + radius * np.cos(angles_rad)   # x
                    src_trj[:, 1] = room_cy                                 # y (constant height)
                    src_trj[:, 2] = room_cz + radius * np.sin(angles_rad)   # z

                    scene['src_trj'] = [src_trj]  # gpuRIR / matlab wrapper expect list‑of‑arrays
                    scene['trajectory_type'] = 'smooth_sine'
                    
                from matlab_sg_wrapper import signal_generator_wrapper
                rev_signals = signal_generator_wrapper(fs, s, scene)
                scene['DOA_az_trj'] = get_trj_DOA(scene, rev_signals)
                
            else: 
                from rir_gen import generate_rirs
                RIRs = generate_rirs(scene['room_dim'], scene['src_pos'], index, scene['mic_pos'], scene['RT60'], fs, args.simulate_trj)

                # Generate reverberant speech for each mic
                rev_signals = np.zeros([mics_num, len(s)])
                for j, rir in enumerate(RIRs[0]):
                    rev_signals[j] = lfilter(rir, 1, s)
            
            # Normalize
            rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  
            # scene['DOA_az_trj'] = get_trj_DOA(scene, rev_signals)

            # # Loop through each mic and save
            # for mic in range(rev_signals.shape[1]):
            #     filename = f"rev_signal_mic{mic}.wav"
            #     wavfile.write(filename, fs, rev_signals[:, mic])
            #     print(f"Saved: {filename}")

            # Add noise
            if args.dataset == 'add_noise':
                rev_signals = add_ar1_noise_to_signals(rev_signals, decay=0.9, snr_db=args.snr)
                rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  # Normalize
                
            if args.inject_burst_noise:
                rev_signals = inject_pink_noise_bursts(
                    rev_signals,
                    fs=fs,
                    burst_duration=0.5,       # seconds
                    snr_db=-5,                # make bursts *louder* than signal
                    num_bursts=10              # 2 bursts at random times
                )
            # normalization- hurts the noise we add
            # rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1
            
            # clipping
            # peak = np.max(np.abs(rev_signals))
            # if peak > 1.0:
            #     rev_signals = rev_signals / peak / 1.1

            # Save data as wavs and create a directory in which wavs will be saved
            speaker_wav_dir = os.path.join(save_rev_speech_dir, 'RevMovingSrcDatasetWavs', f'scene_{scene_idx + 1}', selected_speaker, f'src_pos{index + 1}')
            os.makedirs(speaker_wav_dir, exist_ok=True)
            save_wavs(rev_signals, 0, speaker_wav_dir, fs)

            # Save data as h5py file
            save_data_h5py(rev_signals, scene, scene_idx, index, save_rev_speech_dir)
            
            if args.simulate_trj:
                break # for traj simulation we dont need to loop over src_pos the calculation is performed at "simulateTrajectory"
        
        scene_agg.append(scene)
        
        # save scene info txt file
        write_scene_to_file(scene_agg, os.path.join(save_rev_speech_dir, 'dataset_info.txt'))

    return scene_agg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a clean sound sample."
    )
    # general parameters
    # parser.add_argument("--split", choices=['train', 'val', 'test'], default='endfire_boreside_cp', help="Generate training, val or test")
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='circular_trajectory_inject_burst_noiseVSharon', help="Generate training, val or test")
    parser.add_argument("--dataset", choices=['None', 'add_noise'], default='add_noise')
    # parser.add_argument("--clean_speech_dir", type=str, default='../dataset_folder', help="Directory where the clean speech files are stored")
    parser.add_argument("--clean_speech_dir", type=str, default='/dsi/gannot-lab1/datasets/sharon_db/wsj0/Train', help="Directory where the clean speech files are stored")
    parser.add_argument("--output_folder", type=str, default='', help="Directory where the ourput is saved")

    # scene parameters
    parser.add_argument("--num_scenes", type=int, default=1, help="Number of scenarios to generate")
    parser.add_argument("--mics_num", type=int, default=5, help="Number of microphones in the array")
    parser.add_argument("--mic_min_spacing", type=float, default=0.08, help="Minimum spacing between microphones")
    parser.add_argument("--mic_max_spacing", type=float, default=0.08, help="Maximum spacing between microphones")
    parser.add_argument("--mic_height", type=float, default=1.7, help="Height of the microphone array from the floor")

    parser.add_argument("--room_len_x_min", type=float, default=4, help="Minimum length of the room in x-direction")
    parser.add_argument("--room_len_x_max", type=float, default=7, help="Maximum length of the room in x-direction")
    parser.add_argument("--aspect_ratio_min", type=float, default=1, help="Minimum aspect ratio of the room")
    parser.add_argument("--aspect_ratio_max", type=float, default=1.5, help="Maximum aspect ratio of the room")
    parser.add_argument("--room_len_z_min", type=float, default=2.3, help="Minimum length of the room in z-direction")
    parser.add_argument("--room_len_z_max", type=float, default=2.9, help="Maximum length of the room in z-direction")

    parser.add_argument("--T60_options", type=float, nargs='+', default=[0.3], help="List of T60 values for the room [0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.8]")
    parser.add_argument("--snr", type=float, default=20, help="added noise snr value [dB]")
    parser.add_argument("--noise_fc", type=float, default=1000, help="cufoff lowpass freq for added noise [Hz]")
    parser.add_argument("--noise_AR_decay", type=float, default=0.6, help="cufoff lowpass freq for added noise [Hz]")
    parser.add_argument("--minimum_sentence_len", type=float, default=4, help="default = 8 minimm required for sentence length in seconds")
    parser.add_argument("--maximum_sentence_len", type=float, default=6, help="max required for sentence length in seconds")

    parser.add_argument("--source_min_height", type=float, default=1.6, help="Minimum height of the source (1.5/1.65)")
    parser.add_argument("--source_max_height", type=float, default=1.8, help="Maximum height of the source (2/1.75)")
    parser.add_argument("--source_min_radius", type=float, default=1.5, help="Minimum radius for source localization")
    parser.add_argument("--source_max_radius", type=float, default=1.7, help="Maximum radius for source localization")
    parser.add_argument("--DOA_grid_lag", type=float, default=5, help="Degrees for DOA grid lag")
    
    parser.add_argument("--offgrid_angle", type=bool, default=False, help="generate off sampeling grid doas")
    parser.add_argument("--simulate_trj", type=bool, default=True, help="simulate moving speaker")
    parser.add_argument("--endfire_bounce", type=bool, default=False, help="simulate 'half circle' movment- endfire -> broadband - back to endfire")
    parser.add_argument("--simulate_matlab_trj", type=bool, default=True, help="activate python wrapper for signal generator")
    parser.add_argument("--simulate_sinus", type=bool, default=False, help="simulate sinus like trj- filter win UNET?")    
    parser.add_argument("--fps", type=float, default=125, help="frames per second: fs/(framesize*(1-overlap))")
    parser.add_argument("--inject_burst_noise", type=bool, default=True, help="frames per second: fs/(framesize*(1-overlap))")

    parser.add_argument("--margin", type=float, default=0.5, help="Margin distance between the source/mics to the walls")  
    args = parser.parse_args()
    
    
    ## ======= Main: Generate data for training/validation ======= ##
    scenes = generate_rev_speech(args)
        
