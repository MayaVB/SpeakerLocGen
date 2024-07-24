from scene_gen import generate_scenes, plot_scene, plot_scene_interactive
from utils import read_hdf5_file
from rir_gen import generate_rirs
import numpy as np
import pathlib
from scipy.signal import lfilter
import soundfile as sf
import argparse
import h5py
import os
import random



def select_random_speaker(speakers_dir_clean):
    """
    Selects a random speaker directory and returns the speaker name and all .wav files in that directory.
    :param speakers_dir_clean: Path to the directory containing speaker directories.
    :return: A tuple (speaker_name, list of .wav file paths) where speaker_name is the name of the selected speaker directory.
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


def save_data_h5py(rev_signals, scene, speaker_out_dir=[]):
    """
    :param rev_signals: reverberent signals
    :param scene: scene parameters
    :param speaker_out_dir: directory where the dataset is stored (defults is home dir)
    """
    with h5py.File("reverbDataset.h5", 'a') as h5f:
        group_name = f"room_{scene['room_dim']}_rt60_{scene['RT60']}_pos_{scene['src_pos']}"
        grp = h5f.create_group(group_name)
        grp.create_dataset(f'signals_', data=rev_signals)
        grp.create_dataset(f'speaker_position_', data=scene['src_pos'])


def save_wavs(rev_signals, file, speaker_out_dir, fs):
    for j, sig in enumerate(rev_signals, 1):
        out_file_name = '_ch{}'.format(j) + '.wav' 
        out_file_path = speaker_out_dir / out_file_name
        sf.write(out_file_path, sig, fs)


def round_numbers(list_in):
    return list(map(lambda x: round(x, 4), list_in))


def write_scene_to_file(scenes, file_name):
    with open (file_name, 'w') as f:
        for scene in scenes:
            f.write(f"Room size: {round_numbers(scene['room_dim'])}\n")
            f.write(f"Critical distance: {round(scene['critic_dist'], 4)}\n")
            pos = round_numbers(scene['src_pos'][0])
            f.write(f"Source position: {pos}\n")
            mics_num = len(scene['mic_pos'])
            for i in range(mics_num):
                pos = round_numbers(scene['mic_pos'][i])
                dist = round(scene['dists'][i], 4)
                f.write(f"Mic{i} pos\t: {pos}, dist:{dist}\n")
            f.write('\n\n\n')
            f.flush()
    return


def generate_rev_speech(args, scene_type, clean_speech_dir, save_rev_speech_dir, num_scenes=5, snr=20):
    """
    :param args: From Parser
    :param scene_type: 'near' / 'far' / 'random' / 'winning_ticket'
    :param clean_speech_dir: Directory where the clean speech files are stored
    :param save_rev_speech_dir: Directory to save the generated files
    :param num_scenes: Number of scenes (rooms) to generate
    :param snr: SNR for BIUREV-N
    :return:
    """
    # Get all sub directories (speakers)
    scene_agg = []

    # Generate reverberant speech files
    for scene_idx in range(num_scenes):
        
        selected_speaker, wav_files = select_random_speaker(clean_speech_dir)
        
        # Generate a scene
        scene = generate_scenes(1, scene_type, len(wav_files))[0]
        mics_num = len(scene['mic_pos'])
        plot_scene_interactive(scene, 'Plots', scene_idx)
        
        for index, wav_file in enumerate(wav_files):
            if len(wav_files) < len(scene['src_pos'][0]):
                raise Exception("speaker wav files are lower then requested speaker location- not enough files!") 
                
            print('Processing Scene %d/%d. Speaker: %s, file %d/%d.' % (
                scene_idx + 1, num_scenes, selected_speaker, index + 1, len(wav_files)))

            # Read a clean file
            s, fs = sf.read(wav_file)            

            # Generate RIRs for the current source position
            RIRs, _ = generate_rirs(scene['room_dim'], scene['src_pos'], scene['mic_pos'], scene['RT60'], fs)

            # Generate reverberant speech
            rev_signals = np.zeros([mics_num, len(s)])
            for j, rir in enumerate(RIRs[0][0]):
                rev_signals[j] = lfilter(rir, 1, s)
                
            # Normalize
            rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  

            # BIUREV-N -- Add noise
            if args.dataset == 'BIUREV-N':
                # furthest_mic = np.argmax(scene['dists'])
                furthest_mic = 1 # not calaulated
                var_furthest = np.var(rev_signals[1])
                g = np.sqrt(10 ** (-snr / 10) * var_furthest)
                noise = g * np.random.randn(mics_num, len(s))
                noise = lfilter([1], np.array([1, -0.9]), noise, axis=-1)  # Low-pass filter the noise

                rev_signals += noise
                rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  # Normalize

                # Create a directory in which wav will be saved for examine
                speaker_out_dir = save_rev_speech_dir / 'BIUREV-N' / args.split / scene_type / selected_speaker/ f'scene_{scene_idx + 1}'
                speaker_out_dir.mkdir(parents=True, exist_ok=True)

            # Save wavs
            save_wavs(rev_signals, 0, speaker_out_dir, fs)               # Save signals to disk

            # Save signals to h5py file
            save_data_h5py(rev_signals, scene)
                        
        scene_agg.append(scene)
        
    return scene_agg




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample."
    )
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='train', help="Generate training, val or test")
    parser.add_argument("--dataset", choices=['BIUREV', 'BIUREV-N', 'both'], default='BIUREV-N', help="Generate BIUREV/BIUREV-N/both")
    args = parser.parse_args()

    #################################### Generate training data ####################################

    save_rev_speech_dir = pathlib.Path('/src/Results')
    if args.split == 'train':
        clean_speech_dir = pathlib.Path('../dataset_folder')
        scene_type = 'random'
        generate_rev_speech(args, scene_type, clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'train_random.txt')

    #################################### Generate validation data ####################################

    elif args.split == 'val':
        clean_speech_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_dt/data/cln_test/')

        scenes = generate_rev_speech(args, 'near', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'val_near.txt')

        generate_rev_speech(args, 'far', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'val_far.txt')

    #################################### Generate test data ####################################
    elif args.split == 'test':
        clean_speech_dir = pathlib.Path(
            '/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_et/data/cln_test/')

        scenes = generate_rev_speech(args, 'near', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_near.txt')

        scenes = generate_rev_speech(args, 'far', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_far.txt')

        scenes = generate_rev_speech(args, 'random', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_random.txt')

        scenes = generate_rev_speech(args, 'winning_ticket', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_winning_ticket.txt')