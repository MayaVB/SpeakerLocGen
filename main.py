from scipy.signal import lfilter
import soundfile as sf
import numpy as np

import argparse
import h5py
import os
import random
# from datetime import datetime, timedelta

from scene_gen import generate_scenes, plot_scene_interactive
from rir_gen import generate_rirs


def select_random_speaker(speakers_dir_clean):
    """
    Selects a random speaker directory and returns the speaker name and all .wav files in that directory.
    :param speakers_dir_clean: Path to the directory containing speaker directories
    :return: selected_dir: Name of the selected speaker directory
             wav_files: List of all .wav files in the selected speaker directory
    """
    # List all directories in the speakers_dir_clean
    speaker_dirs = [d for d in os.listdir(speakers_dir_clean) if os.path.isdir(os.path.join(speakers_dir_clean, d))]

    # Ensure there are directories in the speakers_dir_clean
    if not speaker_dirs:
        raise ValueError("No speaker directories found in the directory.")

    # Select a random directory
    selected_dir = random.choice(speaker_dirs)
    selected_dir_path = os.path.join(speakers_dir_clean, selected_dir)

    # List all .wav files in the selected directory
    wav_files = [os.path.join(selected_dir_path, f) for f in os.listdir(selected_dir_path) if f.endswith('.wav')]

    # Ensure there are .wav files in the selected directory
    if not wav_files:
        raise ValueError("No .wav files found in the selected speaker directory.")

    return selected_dir, wav_files


def save_data_h5py(rev_signals, scene, scene_idx, src_pos_index, speaker_out_dir=[]):
    """
    Save wav data and label DOA in two groups, this is done for each scene and for each source position
    :param rev_signals: reverberent signals
    :param scene: scene parameters
    :param scene_idx: scene index
    :param src_pos_index: source position index
    :param speaker_out_dir: directory where the dataset is stored (defults is home dir)
    """
    # os.makedirs(speaker_out_dir, exist_ok=True)
    dataset_path = os.path.join(speaker_out_dir, 'RevMovingSrcDataset.h5')

    with h5py.File(dataset_path, 'a') as h5f:
        group_name = f"sceneIndx_{scene_idx}_roomDim_{scene['room_dim']}_rt60_{scene['RT60']}_srcPosInd_{src_pos_index}"
        grp = h5f.create_group(group_name)
        grp.create_dataset(f'signals_', data=rev_signals)
        grp.create_dataset(f'speaker_DOA_', data=scene['DOA_az'][src_pos_index])
        grp.create_dataset(f'reverberation_time_', data=scene['RT60'])
                
        # Increment the dataset length
        if 'data_len' in h5f.attrs:
            h5f.attrs['data_len'] += 1
        else:
            h5f.attrs['data_len'] = 1
        

def save_wavs(rev_signals, file, speaker_out_dir, fs):
    for j, sig in enumerate(rev_signals, 1):
        out_file_name = '_ch{}'.format(j) + '.wav' 
        out_file_path = os.path.join(speaker_out_dir, out_file_name)
        sf.write(out_file_path, sig, fs)


def round_numbers(list_in):
    return list(map(lambda x: round(x, 4), list_in))


def write_scene_to_file(scenes, file_name):
    """
    print scene info to a txt file
    :param scenes: list of dict containing info about the scenario generated
    :param file_name: file name to save the info
    """
    with open (file_name, 'w') as f:
        for scene in scenes:
            f.write(f"Room size: {round_numbers(scene['room_dim'])}\n")
            f.write(f"Reverberation time: {round(scene['RT60'])}\n")
            f.write(f"Critical distance: {round(scene['critic_dist'], 4)}\n")
            pos = round_numbers(scene['src_pos'][0])
            f.write(f"Source position: {pos}\n")
            mics_num = len(scene['mic_pos'])
            for i in range(mics_num):
                pos = round_numbers(scene['mic_pos'][i])
                dist = round_numbers(scene['dists'][i])
                f.write(f"Mic{i} pos\t: {pos}, dist:{dist}\n")
            f.write('\n\n\n')
            f.flush()
    return


def generate_rev_speech(args):
    """
    :param args: From Parser
    :return: scene_agg: scence list of dict containing info about the scenario generated
    """
    scene_agg = []
    clean_speech_dir = args.clean_speech_dir
    num_scenes = args.num_scenes
    snr = args.snr
    
    save_rev_speech_dir = os.path.join(args.output_folder, args.split)
    # if os.path.exists(save_rev_speech_dir):
        # os.remove(save_rev_speech_dir)
    
    # Generate reverberant speech files
    for scene_idx in range(num_scenes):
        
        selected_speaker, wav_files = select_random_speaker(clean_speech_dir)

        # Generate a scene
        scene = generate_scenes(args)[0]
        mics_num = len(scene['mic_pos'])
        
        # plot an interactive scene plot for each scenarion
        plot_scene_interactive(scene, save_rev_speech_dir, scene_idx)
        
        # for index, wav_file in enumerate(wav_files):
        for index, _ in enumerate(scene['src_pos'][0]):
            curr_src_pos = scene['src_pos'][:, index]
            curr_wav_file_path = random.choice(wav_files)

            if len(wav_files) < len(scene['src_pos'][0]):
                raise Exception("speaker wav files are lower then requested speaker location- not enough files!") 

            print('Processing Scene %d/%d. Speaker: %s, wav file processed: %s, wav file number: %d.' % (
                scene_idx + 1, num_scenes, selected_speaker, os.path.basename(curr_wav_file_path), index + 1))

            # Read a clean file
            s, fs = sf.read(curr_wav_file_path)            

            # Generate RIR for the current source position for all mic positons
            # Basically- this RIR's can be generated for src_pos together- but lfilter cant filter all 
            # atm we keep this inside src_pos loop 
            RIRs = generate_rirs(scene['room_dim'], curr_src_pos, scene['mic_pos'], scene['RT60'], fs)

            # Generate reverberant speech for each mic
            rev_signals = np.zeros([mics_num, len(s)])
            for j, rir in enumerate(RIRs[0]):
                rev_signals[j] = lfilter(rir, 1, s)
                
            # Normalize
            rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  

            # Add noise
            if args.dataset == 'add_noise':
                # furthest_mic = np.argmax(scene['dists'])
                furthest_mic = 1 # not calaulated
                var_furthest = np.var(rev_signals[1])
                g = np.sqrt(10 ** (-snr / 10) * var_furthest)
                noise = g * np.random.randn(mics_num, len(s))
                noise = lfilter([1], np.array([1, -0.9]), noise, axis=-1)  # Low-pass filter the noise

                rev_signals += noise
                rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  # Normalize

                # Create a directory in which wavs will be saved
                speaker_wav_dir = os.path.join(save_rev_speech_dir, 'RevMovingSrcDatasetWavs', f'scene_{scene_idx + 1}', selected_speaker, f'src_pos{index + 1}')
                os.makedirs(speaker_wav_dir, exist_ok=True)
                
            # Save data as wavs
            save_wavs(rev_signals, 0, speaker_wav_dir, fs)

            # Save data as h5py file
            save_data_h5py(rev_signals, scene, scene_idx, index, save_rev_speech_dir)
                        
        scene_agg.append(scene)
        
        # save scene info txt file
        write_scene_to_file(scene_agg, os.path.join(save_rev_speech_dir, 'train_info.txt'))

    return scene_agg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a clean sound sample."
    )
    # general parameters
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='train', help="Generate training, val or test")
    parser.add_argument("--dataset", choices=['None', 'add_noise'], default='add_noise')
    parser.add_argument("--clean_speech_dir", type=str, default='../dataset_folder', help="Directory where the clean speech files are stored")
    parser.add_argument("--output_folder", type=str, default='', help="Directory where the ourput is saved")

    # scene parameters
    parser.add_argument("--num_scenes", type=int, default=3, help="Number of scenarios to generate")
    parser.add_argument("--mics_num", type=int, default=5, help="Number of microphones in the array")
    parser.add_argument("--mic_min_spacing", type=float, default=0.03, help="Minimum spacing between microphones")
    parser.add_argument("--mic_max_spacing", type=float, default=0.08, help="Maximum spacing between microphones")
    parser.add_argument("--mic_height", type=float, default=1.7, help="Height of the microphone array from the floor")

    parser.add_argument("--room_len_x_min", type=float, default=4, help="Minimum length of the room in x-direction")
    parser.add_argument("--room_len_x_max", type=float, default=7, help="Maximum length of the room in x-direction")
    parser.add_argument("--aspect_ratio_min", type=float, default=1, help="Minimum aspect ratio of the room")
    parser.add_argument("--aspect_ratio_max", type=float, default=1.5, help="Maximum aspect ratio of the room")
    parser.add_argument("--room_len_z_min", type=float, default=2.3, help="Minimum length of the room in z-direction")
    parser.add_argument("--room_len_z_max", type=float, default=2.9, help="Maximum length of the room in z-direction")

    parser.add_argument("--T60_options", type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8], help="List of T60 values for the room")
    parser.add_argument("--snr", type=float, default=20, help="added noise snr value [dB]")

    parser.add_argument("--source_min_height", type=float, default=1.5, help="Minimum height of the source")
    parser.add_argument("--source_max_height", type=float, default=2, help="Maximum height of the source")
    parser.add_argument("--source_min_radius", type=float, default=1.5, help="Minimum radius for source localization")
    parser.add_argument("--source_max_radius", type=float, default=1.7, help="Maximum radius for source localization")
    parser.add_argument("--DOA_grid_lag", type=float, default=5, help="Degrees for DOA grid lag")

    parser.add_argument("--margin", type=float, default=0.5, help="Margin distance between the source/mics to the walls")  
    args = parser.parse_args()
    
    
    ## ======= Main: Generate data for training/validation ======= ##
    if args.split == 'train':
        scenes = generate_rev_speech(args)
        
