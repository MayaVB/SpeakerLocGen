from scipy.signal import lfilter
import soundfile as sf
import numpy as np
from scipy.io import wavfile

import argparse
import os
import random
# from datetime import datetime, timedelta
from gpuRIR import simulateTrajectory
from rir_gen import generate_rirs

from scene_gen import select_random_speaker, generate_scenes, plot_scene_interactive
from utils import save_wavs, write_scene_to_file, save_data_h5py
from processing import add_ar1_noise_to_signals
from scene_gen import calculate_doa

def interpolate_angles(start_deg, end_deg, num):
    """Interpolate angles correctly over circular range [0, 360)."""
    delta = (end_deg - start_deg + 180) % 360 - 180
    return (start_deg + np.linspace(0, 1, num) * delta) % 360

def generate_rev_speech(args):
    """
    :param args: From Parser
    :return: scene_agg: scence list of dict containing info about the scenario generated
    """
    scene_agg = []
    clean_speech_dir = args.clean_speech_dir
    num_scenes = args.num_scenes
    snr = args.snr
    
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
                if len(s)/fs < args.minimum_sentence_len:
                    curr_wav_file_path = random.choice(wav_files)
                else:
                    wav_verified = True

            print('Processing Scene %d/%d. Speaker: %s, wav file processed: %s, wav file number: %d.' % (
                scene_idx + 1, num_scenes, selected_speaker, os.path.basename(curr_wav_file_path), index + 1))

            # Generate RIR for the current source position for all mic positons
            # Basically- this RIR's can be generated for src_pos together- but lfilter cant filter all 
            # atm we keep this inside src_pos loop 
            RIRs = generate_rirs(scene['room_dim'], scene['src_pos'], index, scene['mic_pos'], scene['RT60'], fs, args.simulate_trj)

            if args.simulate_trj:
                rev_signals = simulateTrajectory(s, RIRs)
                rev_signals = rev_signals.T
                # # Normalize output to avoid clipping
                # max_val = np.max(np.abs(filtered_signal))
                # if max_val > 0:
                #     filtered_signal /= max_val
                
                # calculate trj DOA
                src_pos_trj = np.array(scene['src_pos'])  # shape [3, N]
                n_samples = rev_signals.shape[1]
                n_rirs = src_pos_trj.shape[1]
                samples_per_rir = n_samples // n_rirs
                az_DOA = scene['DOA_az']  # assumed shape: [n_rirs]

                doa_trj = np.zeros(n_samples)  # azimuth per sample

                for i in range(n_rirs - 1):
                    start = i * samples_per_rir
                    end = (i + 1) * samples_per_rir
                    doa_trj[start:end] = interpolate_angles(az_DOA[i], az_DOA[i + 1], end - start)

                # Final segment — hold the last DOA
                last_start = (n_rirs - 1) * samples_per_rir
                doa_trj[last_start:] = az_DOA[-1]
                scene['DOA_az_trj'] = doa_trj
                
            else: 
                # Generate reverberant speech for each mic
                rev_signals = np.zeros([mics_num, len(s)])
                for j, rir in enumerate(RIRs[0]):
                    rev_signals[j] = lfilter(rir, 1, s)
            
            # Normalize
            rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  
    
            # # Loop through each mic and save
            # for mic in range(rev_signals.shape[1]):
            #     filename = f"rev_signal_mic{mic}.wav"
            #     wavfile.write(filename, fs, rev_signals[:, mic])
            #     print(f"Saved: {filename}")

            # Add noise
            if args.dataset == 'add_noise':
                # furthest_mic = 2 # not calaulated
                
                # var_furthest = np.var(rev_signals[furthest_mic])
                # alpha = np.exp(-2 * np.pi * args.noise_fc  / fs) # Calculate filter coefficient (pole location)
                
                # g = np.sqrt(10 ** (-snr / 10) * var_furthest)
                # noise = g * np.random.randn(mics_num, len(s))
                # noise = lfilter([1], np.array([1, -alpha]), noise, axis=-1)  # Low-pass filter the noise at fcutoff = (fs / (2 * np.pi)) * np.arccos(pole)

                rev_signals = add_ar1_noise_to_signals(rev_signals, decay=0.9, snr_db=args.snr)

                # rev_signals += noise
                rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  # Normalize

                # Create a directory in which wavs will be saved
                speaker_wav_dir = os.path.join(save_rev_speech_dir, 'RevMovingSrcDatasetWavs', f'scene_{scene_idx + 1}', selected_speaker, f'src_pos{index + 1}')
                os.makedirs(speaker_wav_dir, exist_ok=True)
                
            # Save data as wavs
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
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='temp', help="Generate training, val or test")
    parser.add_argument("--dataset", choices=['None', 'add_noise'], default='add_noise')
    parser.add_argument("--clean_speech_dir", type=str, default='../dataset_folder', help="Directory where the clean speech files are stored")
    # parser.add_argument("--clean_speech_dir", type=str, default='/dsi/gannot-lab1/datasets/sharon_db/wsj0/Train', help="Directory where the clean speech files are stored")
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

    parser.add_argument("--T60_options", type=float, nargs='+', default=[0.8], help="List of T60 values for the room [0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.8]")
    parser.add_argument("--snr", type=float, default=10, help="added noise snr value [dB]")
    parser.add_argument("--noise_fc", type=float, default=1000, help="cufoff lowpass freq for added noise [Hz]")
    parser.add_argument("--noise_AR_decay", type=float, default=0.9, help="cufoff lowpass freq for added noise [Hz]")
    parser.add_argument("--minimum_sentence_len", type=float, default=8, help="minimum required for sentence length in seconds")

    parser.add_argument("--source_min_height", type=float, default=1.6, help="Minimum height of the source (1.5/1.65)")
    parser.add_argument("--source_max_height", type=float, default=1.8, help="Maximum height of the source (2/1.75)")
    parser.add_argument("--source_min_radius", type=float, default=1.5, help="Minimum radius for source localization")
    parser.add_argument("--source_max_radius", type=float, default=1.7, help="Maximum radius for source localization")
    parser.add_argument("--DOA_grid_lag", type=float, default=5, help="Degrees for DOA grid lag")
    parser.add_argument("--offgrid_angle", type=bool, default=False, help="generate off sampeling grid doas")
    parser.add_argument("--simulate_trj", type=bool, default=True, help="simulate moving speaker")
    parser.add_argument("--endfire_bounce", type=bool, default=True, help="simulate 'half circle' movment- endfire -> broadband - back to endfire")

    parser.add_argument("--margin", type=float, default=0.5, help="Margin distance between the source/mics to the walls")  
    args = parser.parse_args()
    
    
    ## ======= Main: Generate data for training/validation ======= ##
    scenes = generate_rev_speech(args)
        
